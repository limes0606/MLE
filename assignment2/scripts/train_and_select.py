"""
Train candidate models as of a date, select one by expanding-window walk-forward validation,
refit it on all mature loans, and store it in the model bank.
Usage: python3 train_and_select.py --snapshotdate 2024-05-01

As of date D, only loans originated up to D-5 months are used (their MOB-5 label is known on D).

SELECTION (walk-forward, expanding window)
  The last WF_FOLDS x WF_FOLD_MONTHS mature origination months (9 = 3 folds x 3 months) are each
  predicted by a model trained only on EARLIER origination months. Every candidate sees the same
  folds; the pooled out-of-time predictions rank the candidates.
  Training months immediately before a validation fold have labels that were not yet known when
  the fold began (label lag = 5 months). Purging them leaves only 2 folds and 530 training loans at
  the first training date, so the folds are used unpurged. This is acceptable for RANKING because
  every candidate gets identical folds; the performance LEVEL is taken from the sealed backtest
  and the live cohorts, not from these folds.
  Selection rule (first model, no champion yet): the candidate with the highest pooled walk-forward
  AUC becomes champion. The paired-bootstrap 95% CI of each candidate's AUC gap to the best is
  recorded to show how clear the lead is; it does not change the choice.
  Scheduled retrain (a champion already exists): the challenger is the champion's algorithm refitted
  on the newer, larger window, so the champion/challenger comparison isolates the effect of
  retraining. All candidates are still evaluated and recorded; if another algorithm leads, the run
  flags it for a separate model-change review instead of switching algorithm automatically.
BASELINE stored with the model = pooled out-of-time AUC (with CI) and a monthly-AUC band
  (mean +/- 2 SD of the monthly AUCs) because single-cohort AUC is noisy.
"""
import argparse
import json
import os
import pickle
import platform

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils.config import MODEL_BANK_DIR
from utils.model_utils import (PlattCalibrator, CANDIDATES, CATEGORICAL, CUTOFF, FEATURES, NUMERIC, SEED, WF_FOLDS,
                               WF_FOLD_MONTHS, build_pipeline, evaluate, load_training_frame)
from utils.psi import categorical_reference, numeric_reference

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
ds = p.parse_args().snapshotdate
ds_ = ds.replace("-", "_")

df = load_training_frame(ds)
months = [pd.Timestamp(m) for m in sorted(df["orig_month"].unique())]
n_val = WF_FOLDS * WF_FOLD_MONTHS
assert len(months) >= n_val + 3, "not enough mature months for walk-forward selection"
val_months = months[-n_val:]
folds = [val_months[i:i + WF_FOLD_MONTHS] for i in range(0, n_val, WF_FOLD_MONTHS)]
print(f"[train] as of {ds}: {len(df)} labelled loans, originated {months[0]:%Y-%m} to {months[-1]:%Y-%m}; "
      f"latest label date used {df.label_date.max():%Y-%m-%d}")
print(f"[train] walk-forward: {WF_FOLDS} folds x {WF_FOLD_MONTHS} months, expanding training window")

# 1) walk-forward: each fold is predicted by a model fitted on earlier months only
oof_frames, fold_info = [], []
for i, fm in enumerate(folds, 1):
    tr, va = df[df.orig_month < fm[0]], df[df.orig_month.isin(fm)]
    fold_info.append({"fold": i, "train_to": f"{tr.orig_month.max():%Y-%m}", "n_train": int(len(tr)),
                      "val_from": f"{fm[0]:%Y-%m}", "val_to": f"{fm[-1]:%Y-%m}", "n_val": int(len(va))})
    part = va[["orig_month", "label"]].copy()
    part["fold"] = i
    for name in CANDIDATES:
        part[name] = build_pipeline(name).fit(tr[FEATURES], tr["label"]).predict_proba(va[FEATURES])[:, 1]
    oof_frames.append(part)
    print(f"[train] fold {i}: train {len(tr)} loans (to {tr.orig_month.max():%Y-%m}), "
          f"validate {len(va)} loans ({fm[0]:%Y-%m}..{fm[-1]:%Y-%m})")
oof = pd.concat(oof_frames, ignore_index=True)

results = {}
for name in CANDIDATES:
    m = evaluate(oof["label"], oof[name])
    m["fold_auc"] = [float(roc_auc_score(g["label"], g[name])) for _, g in oof.groupby("fold")]
    m["monthly_auc"] = [float(roc_auc_score(g["label"], g[name])) for _, g in oof.groupby("orig_month")]
    results[name] = m
    print(f"[train] {name:14s} pooled AUC {m['auc']:.3f} (95% CI {m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f}) "
          f"folds {[round(a, 3) for a in m['fold_auc']]} Gini {m['gini']:.3f} KS {m['ks']:.3f} Brier {m['brier']:.3f}")

# 2) selection: highest pooled AUC; paired bootstrap of each gap to the best is recorded, not used to choose
best = max(results, key=lambda k: results[k]["auc"])
rng = np.random.default_rng(SEED)
y = oof["label"].to_numpy()
N_BOOT = 5000
idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
gap = {}
for name in CANDIDATES:
    if name == best:
        gap[name] = (0.0, 0.0, 0.0)
        continue
    d = []
    for ix in idx:
        yb = y[ix]
        if yb.min() == yb.max():
            continue
        d.append(roc_auc_score(yb, oof[best].to_numpy()[ix]) - roc_auc_score(yb, oof[name].to_numpy()[ix]))
    gap[name] = (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    print(f"[train] ({N_BOOT} paired draws) best ({best}) minus {name}: AUC gap {gap[name][0]:+.4f} (95% CI {gap[name][1]:+.4f} to {gap[name][2]:+.4f})")
within_ci = [k for k in CANDIDATES if k == best or gap[k][1] <= 0]
reg_path = MODEL_BANK_DIR / "registry.json"
registry = json.loads(reg_path.read_text()) if reg_path.exists() else {"models": []}
model_id = f"credit_model_{ds_}"
current = next((m for m in registry["models"] if m["status"] == "champion" and m["model_id"] != model_id), None)
if current is None:
    champion, role = best, "champion"
    reason = "highest pooled walk-forward AUC (no champion yet)"
else:
    champion, role = current["algorithm"], "challenger"
    reason = f"scheduled retrain: same algorithm as champion {current['model_id']}, refitted on newer data"
review_flag = role == "challenger" and best != champion
print(f"[train] best by AUC: {best}; gap CI includes 0 for: {within_ci}; selected: {champion} ({reason})")
if review_flag:
    print(f"[train] NOTE: {best} leads the retrain folds; algorithm change needs a separate model-change review")

# 3) baseline = pooled out-of-time metrics + monthly band for the selected model
mon = np.array(results[champion]["monthly_auc"])
baseline = {k: results[champion][k] for k in ("auc", "auc_ci_low", "auc_ci_high", "gini", "ks", "brier")}
baseline.update({"monthly_auc_mean": float(mon.mean()), "monthly_auc_sd": float(mon.std(ddof=1)),
                 "band_low": float(mon.mean() - 2 * mon.std(ddof=1)),
                 "band_high": float(mon.mean() + 2 * mon.std(ddof=1)), "n_months": int(len(mon))})
print(f"[train] baseline AUC {baseline['auc']:.3f}; monthly band {baseline['band_low']:.3f}-{baseline['band_high']:.3f}")

# 4) refit the winner on every mature loan; out-of-fold scores give an honest PSI reference
final = build_pipeline(champion).fit(df[FEATURES], df["label"])
oof5 = cross_val_predict(clone(build_pipeline(champion)), df[FEATURES], df["label"],
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
psi_reference = {"score": numeric_reference(oof5), "features": {c: numeric_reference(df[c]) for c in NUMERIC}}
psi_reference["features"].update({c: categorical_reference(df[c]) for c in CATEGORICAL})

# Calibration: map the raw model score to a default probability (Platt sigmoid) fitted on the same
# out-of-fold scores. The raw score stays the basis for ranking, drift and model comparison.
calibrator = PlattCalibrator().fit(oof5, df["label"])
print(f"[train] calibrator (Platt on out-of-fold scores): slope {calibrator.slope_:.3f}, intercept {calibrator.intercept_:+.3f}")

bundle = {
    "model_id": model_id, "algorithm": champion, "pipeline": final,
    "features": {"numeric": NUMERIC, "categorical": CATEGORICAL}, "cutoff": CUTOFF,
    "trained_asof": ds,
    "training_window": {"origination_from": f"{months[0]:%Y-%m-%d}", "origination_to": f"{months[-1]:%Y-%m-%d}",
                        "n_loans": int(len(df)), "bad_rate": float(df['label'].mean())},
    "selection": {"method": "expanding-window walk-forward (unpurged)", "folds": fold_info,
                  "validation_months": [f"{val_months[0]:%Y-%m}", f"{val_months[-1]:%Y-%m}"]},
    "candidate_metrics": results,
    "selected_because": {"rule": reason, "role": role, "best_by_auc": best, "gap_ci_includes_zero": within_ci,
                         "algorithm_change_review_flag": review_flag,
                         "auc_gap_to_best": {k: {"mean": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in gap.items()}},
    "baseline": baseline, "psi_reference": psi_reference,
    "calibrator": calibrator,
    "calibration": {"method": "Platt sigmoid on 5-fold out-of-fold scores", "n_loans": int(len(df)),
                    "slope": calibrator.slope_, "intercept": calibrator.intercept_},
    "excluded_features": "clickstream fe_1..fe_20 (available only for loans originated up to 2024-06)",
    "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
}
MODEL_BANK_DIR.mkdir(parents=True, exist_ok=True)
model_file = f"{model_id}.pkl"
with open(MODEL_BANK_DIR / model_file, "wb") as f:
    pickle.dump(bundle, f)

# 5) registry + champion pointer (first model is champion; later ones enter as challengers, never auto-promoted)
registry["models"] = [m for m in registry["models"] if m["model_id"] != model_id]
registry["models"].append({
    "model_id": model_id, "file": model_file, "algorithm": champion, "trained_asof": ds,
    "status": role, "selection_rule": reason, "best_by_auc": best,
    "training_window": bundle["training_window"], "baseline": baseline,
    "candidate_auc": {k: round(v["auc"], 4) for k, v in results.items()},
    "candidate_fold_auc": {k: [round(a, 4) for a in v["fold_auc"]] for k, v in results.items()},
})
reg_path.write_text(json.dumps(registry, indent=2))
champ = next(m for m in registry["models"] if m["status"] == "champion")
new_ptr = json.dumps({"model_id": champ["model_id"], "file": champ["file"]}, indent=2)
ptr = MODEL_BANK_DIR / "champion.json"
if not ptr.exists() or ptr.read_text() != new_ptr:          # atomic swap; untouched when the champion is unchanged
    (MODEL_BANK_DIR / "champion.json.tmp").write_text(new_ptr)
    os.replace(MODEL_BANK_DIR / "champion.json.tmp", ptr)
print(f"[train] saved {MODEL_BANK_DIR / model_file}; status={registry['models'][-1]['status']}; champion={champ['model_id']}")
