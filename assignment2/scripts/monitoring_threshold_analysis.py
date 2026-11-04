"""
Monitoring threshold analysis (standalone, NOT an Airflow task; changes nothing in the pipeline).

Question: how noisy is the rolling three-cohort AUC, and what do the AUC drop limits (amber 0.03, red 0.05 below baseline,
two consecutive runs) cost in false alarms and buy in detection?
Data: the champion scored on the 13 cohorts it never trained on (originations 2024-01..2025-01), about 500 loans each.
  1  measured noise: monthly cohort AUC, rolling three-cohort AUC (actual series and bootstrap)
  2  false alarms: simulated 22-run series drawn from the pooled loans (no decay), with the baseline known exactly and with its estimation error
  3  detection: from a chosen month the model's ranking quality is degraded by a set AUC drop (score distribution unchanged);
     reports the share detected and the median months from onset, including the 5-month label lag
Limits: cohorts are drawn independently (no drift or autocorrelation); step decay of ranking quality only; 13 cohorts give a rough noise estimate.
Usage (project root, after the DAG has built the gold stores and the champion):  python3 scripts/monitoring_threshold_analysis.py   (about 2 minutes)
Outputs in datamart/model_diagnostics/ : monitoring_threshold_*.csv / .json
"""
import os, sys, pickle, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.environ.get("PIPELINE_BASE", str(Path(__file__).resolve().parents[1])); sys.path.insert(0, ROOT)
from utils.model_utils import FEATURES, load_training_frame
OUT = os.path.join(ROOT, "datamart", "model_diagnostics"); os.makedirs(OUT, exist_ok=True); t0 = time.time()
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
N_SER = int(os.environ.get("N_SER", 1000)); H = 24                      # simulated series and months per series
rng = np.random.default_rng(2026)

df = load_training_frame("2025-06-01"); ch = pickle.load(open(f"{ROOT}/model_bank/credit_model_2024_05_01.pkl", "rb"))
base_hat = ch["baseline"]["auc"]; base_se = (ch["baseline"]["auc_ci_high"] - ch["baseline"]["auc_ci_low"]) / 3.92
pool = df[df.orig_month >= "2024-01-01"].reset_index(drop=True)          # champion has not seen these (13 cohorts)
y = pool.label.to_numpy(); s = ch["pipeline"].predict_proba(pool[FEATURES])[:, 1]; mo = pool.orig_month.to_numpy()
cohorts = sorted(pd.unique(mo)); n_c = int(np.median([ (mo == c).sum() for c in cohorts ]))
auc_c = np.array([roc_auc_score(y[mo == c], s[mo == c]) for c in cohorts])
true_auc = roc_auc_score(y, s)
log(f"{len(cohorts)} cohorts, {len(y)} loans, median {n_c}/cohort; pooled champion AUC {true_auc:.4f}; shipped baseline {base_hat:.4f} (SE {base_se:.4f})")
log("monthly cohort AUC: mean %.4f SD %.4f" % (auc_c.mean(), auc_c.std(ddof=1)))
# measured rolling-3 noise: (a) empirical SD of the actual rolling series, (b) bootstrap SE of each pooled window
roll = [roc_auc_score(y[np.isin(mo, cohorts[i:i+3])], s[np.isin(mo, cohorts[i:i+3])]) for i in range(len(cohorts) - 2)]
bs = []
for i in range(len(cohorts) - 2):
    k = np.isin(mo, cohorts[i:i+3]); yy, ss = y[k], s[k]; d = []
    for _ in range(300):
        j = rng.integers(0, len(yy), len(yy)); d.append(roc_auc_score(yy[j], ss[j]))
    bs.append(np.std(d))
log("rolling-3 AUC (actual, %d windows): SD %.4f; bootstrap SE per window mean %.4f (independent-cohort guess 0.022/sqrt3 = %.4f)" % (len(roll), np.std(roll, ddof=1), np.mean(bs), auc_c.std(ddof=1)/np.sqrt(3)))

# decayed scores: blend the score's rank with random noise until pooled AUC has fallen by delta (ranking quality decays, bad rate unchanged)
rk = pd.Series(s).rank(pct=True).to_numpy(); u = rng.random(len(s))
s_sorted = np.sort(s)
def decayed(lmb):                                   # same score distribution as the real model; only the ranking quality decays
    b = pd.Series((1 - lmb) * rk + lmb * u).rank(pct=True).to_numpy()
    return s_sorted[np.minimum((b * len(s)).astype(int), len(s) - 1)]
def lam_for(delta):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        m = (lo + hi) / 2
        if true_auc - roc_auc_score(y, decayed(m)) < delta: lo = m
        else: hi = m
    return (lo + hi) / 2
DELTAS = [0.02, 0.03, 0.05, 0.07, 0.10]; lam = {d: lam_for(d) for d in DELTAS}
log("blend weights for target AUC drops:", {d: round(l, 3) for d, l in lam.items()})
sd = {d: decayed(lam[d]) for d in DELTAS}

def series(onset=None, delta=None):
    """Monthly pooled-3 rolling AUC for H months of ~n_c loans each; from `onset` the model has decayed by delta."""
    yc, sc = [], []
    for t in range(H):
        ix = rng.integers(0, len(y), n_c); yc.append(y[ix])
        sc.append((sd[delta] if (onset is not None and t >= onset) else s)[ix])
    return np.array([roc_auc_score(np.concatenate(yc[t-2:t+1]), np.concatenate(sc[t-2:t+1])) for t in range(2, H)])
LIMITS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
def first_alarm(r, base, lim):                           # two consecutive runs at/below baseline - limit (the SOP persistence rule)
    b = r < base - lim; return next((i for i in range(1, len(b)) if b[i] and b[i-1]), None)

# 1) FALSE ALARMS: stable world (no decay). Baseline = truth + estimation error (or exact).
stable = [series() for _ in range(N_SER)]; log("stable series done")
rows = []
for lim in LIMITS:
    for tag, err in (("baseline known exactly", 0.0), ("baseline with estimation error", base_se)):
        e = np.random.default_rng(1).normal(0, err, N_SER) if err else np.zeros(N_SER)
        fa = [first_alarm(r, true_auc + e[k], lim) is not None for k, r in enumerate(stable)]
        rows.append({"limit": lim, "baseline": tag, "false_alarm_share_of_series_within_22_runs": round(float(np.mean(fa)), 3)})
FA = pd.DataFrame(rows); FA.to_csv(f"{OUT}/monitoring_threshold_false_alarm.csv", index=False); print(FA.pivot(index="limit", columns="baseline", values="false_alarm_share_of_series_within_22_runs").to_string())

# 2) DETECTION: decay starts at month 6 of the series; baseline = truth (best case for detection)
ONSET = 8; LAG = 5
rows = []
for d in DELTAS:
    runs = [series(ONSET, d) for _ in range(max(300, N_SER // 3))]
    for lim in LIMITS:
        idx = [first_alarm(r, true_auc, lim) for r in runs]
        # rolling index i ends at cohort month i+2 (0-based); decay affects cohorts >= ONSET; alarm after cohort c, run month = c + LAG
        det = [(i + 2 - ONSET) for i in idx if i is not None and i + 2 >= ONSET]
        early = [i for i in idx if i is not None and i + 2 < ONSET]
        rows.append({"true_drop": d, "limit": lim, "detected_within_series": round(float(len(det) / len(runs)), 3),
                     "median_cohorts_after_onset": float(np.median(det)) if det else None,
                     "median_months_from_onset_incl_5m_lag": (float(np.median(det)) + LAG) if det else None})
    log(f"detection done for drop {d}")
DT = pd.DataFrame(rows); DT.to_csv(f"{OUT}/monitoring_threshold_detection.csv", index=False)
for d in DELTAS:
    print(f"\ntrue AUC drop {d}:"); print(DT[DT.true_drop == d][["limit", "detected_within_series", "median_cohorts_after_onset", "median_months_from_onset_incl_5m_lag"]].to_string(index=False))

json.dump({"cohorts": len(cohorts), "loans": int(len(y)), "median_cohort_size": n_c, "pooled_champion_auc": float(true_auc),
           "shipped_baseline_auc": float(base_hat), "baseline_se": float(base_se),
           "monthly_cohort_auc_mean": float(auc_c.mean()), "monthly_cohort_auc_sd": float(auc_c.std(ddof=1)),
           "rolling3_auc_actual_sd": float(np.std(roll, ddof=1)), "rolling3_auc_bootstrap_se_mean": float(np.mean(bs)),
           "blend_weight_for_auc_drop": {str(d): float(l) for d, l in lam.items()}, "simulated_series": N_SER, "months_per_series": H,
           "decay_onset_month": ONSET, "label_lag_months": LAG}, open(f"{OUT}/monitoring_threshold_summary.json", "w"), indent=1)
