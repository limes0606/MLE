"""
Promotion gate analysis (standalone, NOT an Airflow task; does not change the champion or the model bank).

Question: how much of the challenger-vs-champion AUC lift is noise, and does the gate (lift >= +0.010 with the 95% CI above zero) make sense?
Evaluation loans: originations after the challenger's training window (neither model saw them), scored by both models.
  A  sampling noise of the cumulative lift at each monitoring look, and the lift of each cohort on its own
  B  retraining noise with zero true benefit: the champion's settings refitted on bootstrap resamples of the same 2023 loans
     (resampling lowers the average lift, so use its spread, not its mean) and on 10 different random seeds
  C  how often the gate fires when the true lift is zero (refitted models from B, five looks)
  D  paired swap test at the final look (each loan's two scores swapped at random)
  E  chance the gate fires for a range of true lifts (normal approximation, final look)
  F  rough value of +0.01 AUC per 1,000 applicants (margin scenarios; loss per bad loan 7,544, no recovery; best cutoff per model on the same loans)
Usage (project root, after the DAG has built the gold stores and both models):  python3 scripts/promotion_gate_analysis.py   (about 1 minute)
Outputs in datamart/model_diagnostics/ : promotion_gate_*.csv / .json
"""
import os, sys, pickle, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
ROOT = os.environ.get("PIPELINE_BASE", str(Path(__file__).resolve().parents[1])); sys.path.insert(0, ROOT)
from utils.model_utils import FEATURES, SEED, build_pipeline, load_training_frame
OUT = os.path.join(ROOT, "datamart", "model_diagnostics"); os.makedirs(OUT, exist_ok=True); t0 = time.time()
from math import erf, sqrt
def norm_cdf(z): return 0.5 * (1 + erf(z / sqrt(2)))
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)

df = load_training_frame("2025-06-01")
ch = pickle.load(open(f"{ROOT}/model_bank/credit_model_2024_05_01.pkl", "rb"))
rt = pickle.load(open(f"{ROOT}/model_bank/credit_model_2025_01_01.pkl", "rb"))
ev = df[df.orig_month >= "2024-09-01"].reset_index(drop=True)          # loans after the challenger's training window
y = ev.label.to_numpy(); months = [pd.Timestamp(m) for m in sorted(ev.orig_month.unique())]
s_ch = ch["pipeline"].predict_proba(ev[FEATURES])[:, 1]; s_rt = rt["pipeline"].predict_proba(ev[FEATURES])[:, 1]
log(f"eval loans {len(ev)} ({months[0]:%Y-%m}..{months[-1]:%Y-%m}); champion AUC {roc_auc_score(y,s_ch):.4f}, retrained {roc_auc_score(y,s_rt):.4f}")
looks = [ev.orig_month <= m for m in months]                            # cumulative looks = monitoring runs 2025-02..2025-06

def lift_ci(yy, a, b, n_boot=300, rng=None):
    rng = rng or np.random.default_rng(42); d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yy), len(yy))
        if yy[i].min() != yy[i].max(): d.append(roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]))
    return roc_auc_score(yy, b) - roc_auc_score(yy, a), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.std(d))

# A) sampling noise of the lift estimate at each look + per-cohort lift
rows = []
for m, lk in zip(months, looks):
    k = lk.to_numpy(); pt, lo, hi, se = lift_ci(y[k], s_ch[k], s_rt[k])
    c = (ev.orig_month == m).to_numpy(); pc = roc_auc_score(y[c], s_rt[c]) - roc_auc_score(y[c], s_ch[c])
    rows.append({"look_to_cohort": f"{m:%Y-%m}", "n_loans": int(k.sum()), "cum_lift": round(pt, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                 "se_lift": round(se, 4), "cohort_only_lift": round(pc, 4)})
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/promotion_gate_sampling_noise.csv", index=False); print(A.to_string(index=False)); log("A done")
se_final = A.se_lift.iloc[-1]

# B) retraining noise with ZERO true benefit: champion spec refitted on bootstrap resamples of the SAME 2023 training loans
train23 = df[df.orig_month <= "2023-12-01"]
N_NULL = int(os.environ.get("N_NULL", 30)); rng = np.random.default_rng(7); null_scores = []; rows = []
for r in range(N_NULL):
    ix = rng.integers(0, len(train23), len(train23)); tr = train23.iloc[ix]
    m = build_pipeline("xgboost"); m.set_params(model__random_state=SEED + 100 + r); m.fit(tr[FEATURES], tr.label)
    s = m.predict_proba(ev[FEATURES])[:, 1]; null_scores.append(s)
    rows.append({"run": r, "auc_eval": roc_auc_score(y, s), "lift_vs_champion": roc_auc_score(y, s) - roc_auc_score(y, s_ch)})
B = pd.DataFrame(rows).round(4); B.to_csv(f"{OUT}/promotion_gate_retrain_noise.csv", index=False)
print("retrain-noise lift (no true benefit): mean %+.4f SD %.4f min %+.4f max %+.4f" % (B.lift_vs_champion.mean(), B.lift_vs_champion.std(), B.lift_vs_champion.min(), B.lift_vs_champion.max())); log("B done")

# B2) seed-only noise (same data, different random seed)
sd_rows = []
for r in range(10):
    m = build_pipeline("xgboost"); m.set_params(model__random_state=SEED + 200 + r); m.fit(train23[FEATURES], train23.label)
    sd_rows.append(roc_auc_score(y, m.predict_proba(ev[FEATURES])[:, 1]) - roc_auc_score(y, s_ch))
print("seed-only lift SD %.4f (range %+.4f..%+.4f)" % (np.std(sd_rows, ddof=1), min(sd_rows), max(sd_rows)))

# C) gate false-fire rate with zero true benefit: the gate (lift>=0.010 and CI low>0) is checked at all 5 looks for each null model
def gate_fires(a, b, rng):
    out = []
    for lk in looks:
        k = lk.to_numpy(); pt, lo, hi, se = lift_ci(y[k], a[k], b[k], n_boot=200, rng=rng); out.append(pt >= 0.010 and lo > 0)
    return out
rng2 = np.random.default_rng(11); fires = [gate_fires(s_ch, s, rng2) for s in null_scores]
F = np.array(fires); C = {"null_models": len(F), "fire_any_look": float(F.any(1).mean()), "fire_at_final_look": float(F[:, -1].mean()),
                          "per_look": [float(v) for v in F.mean(0)]}
json.dump(C, open(f"{OUT}/promotion_gate_false_fire.json", "w"), indent=1); print("gate false-fire (zero true benefit):", C); log("C done")

# D) paired swap null at the final look (no retraining): randomly swap each loan's two scores
rng3 = np.random.default_rng(5); sw = []
for _ in range(1000):
    f = rng3.random(len(y)) < 0.5; a = np.where(f, s_rt, s_ch); b = np.where(f, s_ch, s_rt)
    sw.append(roc_auc_score(y, b) - roc_auc_score(y, a))
print("swap-null final look: SD %.4f; 95%% of null lifts within +-%.4f; P(lift>=0.010)=%.3f" % (np.std(sw), np.percentile(np.abs(sw), 95), np.mean(np.array(sw) >= 0.010)))

# E) power of the gate at the final look, normal approximation, for a range of true lifts (single look)
P = []
for d in [0.0, 0.005, 0.010, 0.015, 0.020, 0.030]:
    thr = max(0.010, 1.96 * se_final); P.append({"true_lift": d, "p_gate_fires": round(float(1 - norm_cdf((thr - d) / se_final)), 3)})
pd.DataFrame(P).to_csv(f"{OUT}/promotion_gate_power.csv", index=False); print(pd.DataFrame(P).to_string(index=False), f"\nSE at final look {se_final:.4f}; min lift for 80% power: {max(0.010,1.96*se_final)+0.84*se_final:.4f}")

# F) rough business value per 1,000 applicants: value of the best cutoff for models spanning the AUC range (same eval loans)
LOSS = 7544.0
def best_value(score, margin, rec=0.0):
    best = -1e18
    for c in np.quantile(score, np.linspace(0.02, 0.98, 49)):
        ap = score < c; v = np.where(y[ap] == 0, margin, -LOSS * (1 - rec)).sum() / len(y) * 1000; best = max(best, v)
    return best
mods = {"logreg": build_pipeline("logreg"), "random_forest": build_pipeline("random_forest"), "xgboost": build_pipeline("xgboost")}
sc = {}; 
for k, m in mods.items(): sc[k] = m.fit(train23[FEATURES], train23.label).predict_proba(ev[FEATURES])[:, 1]
rows = []
for k, s in sc.items():
    r = {"model": k, "auc_eval": round(roc_auc_score(y, s), 4)}
    for mg in (650, 1500, 3000): r[f"value_per_1000_margin{mg}"] = round(best_value(s, mg))
    rows.append(r)
r = {"model": "challenger_retrained", "auc_eval": round(roc_auc_score(y, s_rt), 4)}; r.update({f"value_per_1000_margin{mg}": round(best_value(s_rt, mg)) for mg in (650, 1500, 3000)}); rows.append(r)
r = {"model": "champion", "auc_eval": round(roc_auc_score(y, s_ch), 4)}; r.update({f"value_per_1000_margin{mg}": round(best_value(s_ch, mg)) for mg in (650, 1500, 3000)}); rows.append(r)
Fv = pd.DataFrame(rows); Fv.to_csv(f"{OUT}/promotion_gate_value.csv", index=False); print(Fv.to_string(index=False))
# value of +0.01 AUC: slope from a line through the models
SLOPE = {}
for mg in (650, 1500, 3000):
    x = Fv.auc_eval.to_numpy(); v = Fv[f"value_per_1000_margin{mg}"].to_numpy(); sl = np.polyfit(x, v, 1)[0] * 0.01; SLOPE[mg] = float(sl)
    print(f"margin {mg}: ~{sl:,.0f} per 1,000 applicants per +0.01 AUC (line through the 5 models)")
json.dump({"eval_loans": int(len(ev)), "eval_cohorts": [f"{m:%Y-%m}" for m in months], "champion_auc": float(roc_auc_score(y, s_ch)),
           "retrained_auc": float(roc_auc_score(y, s_rt)), "final_look_lift_se": float(se_final),
           "retrain_resample_lift_sd": float(B.lift_vs_champion.std()), "retrain_resample_lift_mean": float(B.lift_vs_champion.mean()),
           "seed_only_lift_sd": float(np.std(sd_rows, ddof=1)), "swap_null_sd": float(np.std(sw)),
           "swap_null_p_lift_ge_0.010": float(np.mean(np.array(sw) >= 0.010)),
           "min_true_lift_for_80pct_power": float(max(0.010, 1.96 * se_final) + 0.84 * se_final),
           "value_per_1000_applicants_per_0.01_auc": SLOPE}, open(f"{OUT}/promotion_gate_summary.json", "w"), indent=1)
log("all done")
