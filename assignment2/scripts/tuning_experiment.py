"""
Hyperparameter tuning experiment (standalone, NOT an Airflow task; it does not change the champion).

Question: would tuning the fixed model settings improve the model enough to matter?
Method (run from the project root, after the DAG has built the gold stores):
  1. Data: 2023 originations only (the champion's training window at 2024-05-01).
  2. Search: for each candidate model, the default settings plus N random settings (logistic regression:
     a grid of C). Each setting is scored by pooled AUC on the same expanding walk-forward folds used for model
     selection. The search is repeated for several seeds to show how stable the winner is.
  3. Check on unseen loans: the winner of each search and the default are trained on all 2023 loans and scored
     on the sealed 2024-01..04 originations (never used to tune). AUC gap with a 2,000-draw bootstrap 95% range.
Usage:  python3 scripts/tuning_experiment.py            (full run: seeds 7,11,23; 25 settings each; about 10 minutes)
        python3 scripts/tuning_experiment.py --seeds 7 --n-configs 3     (quick check)
Outputs in datamart/model_tuning/ : tuning_results.csv, tuning_vs_default.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import DATAMART_DIR
from utils.model_utils import FEATURES, WF_FOLD_MONTHS, WF_FOLDS, build_pipeline, load_training_frame

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="7,11,23")
ap.add_argument("--n-configs", type=int, default=25)
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
N_BOOT = 2000
OUT = DATAMART_DIR / "model_tuning"
OUT.mkdir(parents=True, exist_ok=True)

GRID = {
    "logreg": {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "random_forest": {"model__min_samples_leaf": [3, 5, 10, 20, 40, 80], "model__max_features": ["sqrt", 0.2, 0.35, 0.5],
                      "model__max_depth": [None, 6, 10, 16], "model__n_estimators": [300, 500]},
    "xgboost": {"model__max_depth": [2, 3, 4, 5, 6], "model__learning_rate": [0.02, 0.03, 0.05, 0.1],
                "model__n_estimators": [150, 300, 500, 800], "model__min_child_weight": [1, 5, 10, 20],
                "model__subsample": [0.6, 0.8, 1.0], "model__colsample_bytree": [0.5, 0.8, 1.0],
                "model__reg_lambda": [1, 5, 20]},
}

train = load_training_frame("2024-05-01")                       # originations 2023-01..12, labels known by 2024-05
later = load_training_frame("2024-09-01")                       # adds 2024-01..04, labels known by 2024-09
sealed = later[(later.orig_month >= "2024-01-01") & (later.orig_month <= "2024-04-01")]
months = sorted(train.orig_month.unique())
val = months[-WF_FOLDS * WF_FOLD_MONTHS:]
folds = [val[i:i + WF_FOLD_MONTHS] for i in range(0, len(val), WF_FOLD_MONTHS)]
y = sealed.label.values
print(f"[tuning] train {len(train)} loans (2023); sealed test {len(sealed)} loans (2024-01..04), {y.mean():.1%} bad")


def fit_predict(kind, params, fit_df, pred_df):
    pl = build_pipeline(kind)
    if params:
        pl.set_params(**params)
    return pl.fit(fit_df[FEATURES], fit_df.label).predict_proba(pred_df[FEATURES])[:, 1]


def walk_forward_auc(kind, params):
    ps, ys = [], []
    for fm in folds:
        a, b = train[train.orig_month < fm[0]], train[train.orig_month.isin(fm)]
        ps.append(fit_predict(kind, params, a, b))
        ys.append(b.label.values)
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


def settings(kind, seed):
    if kind == "logreg":
        return [{"model__C": c} for c in GRID[kind]["model__C"]]
    rng, seen, out = random.Random(seed), set(), []
    while len(out) < args.n_configs:
        c = {k: rng.choice(v) for k, v in GRID[kind].items()}
        key = json.dumps(c, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


boot = np.random.default_rng(42)
idx = [boot.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
idx = [i for i in idx if 0 < y[i].sum() < len(i)]


def gap_ci(p_new, p_old):
    g = [roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]) for i in idx]
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


rows, preds = [], {}
for kind in ["logreg", "random_forest", "xgboost"]:
    default_p = fit_predict(kind, {}, train, sealed)
    default_wf = walk_forward_auc(kind, {})
    preds[(kind, "default")] = default_p
    for seed in (SEEDS if kind != "logreg" else [0]):
        scored = [(walk_forward_auc(kind, c), c) for c in settings(kind, seed)]
        best_wf, best = max(scored, key=lambda t: t[0])
        tuned_p = fit_predict(kind, best, train, sealed)
        preds[(kind, f"seed{seed}")] = tuned_p
        lo, hi = gap_ci(tuned_p, default_p)
        rows.append({"model": kind, "search_seed": seed if kind != "logreg" else "grid", "settings_tried": len(scored),
                     "walkforward_auc_default": default_wf, "walkforward_auc_best": best_wf,
                     "sealed_auc_default": roc_auc_score(y, default_p), "sealed_auc_tuned": roc_auc_score(y, tuned_p),
                     "sealed_gap": roc_auc_score(y, tuned_p) - roc_auc_score(y, default_p), "gap_ci_low": lo, "gap_ci_high": hi,
                     "best_settings": json.dumps({k.replace("model__", ""): v for k, v in best.items()}, default=str)})
        print(f"[tuning] {kind} seed {seed}: walk-forward {default_wf:.4f} -> {best_wf:.4f}; sealed {rows[-1]['sealed_auc_default']:.4f} -> "
              f"{rows[-1]['sealed_auc_tuned']:.4f} (gap {rows[-1]['sealed_gap']:+.4f}, 95% {lo:+.4f} to {hi:+.4f})", flush=True)

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT / "tuning_results.csv", index=False)

# which model ranks higher on the sealed loans, default and tuned (first search seed)
cmp_rows = []
for tag, a, b in (("default", "default", "default"), ("tuned", f"seed{SEEDS[0]}", f"seed{SEEDS[0]}")):
    pa, pb = preds[("random_forest", a)], preds[("xgboost", b)]
    lo, hi = gap_ci(pb, pa)
    cmp_rows.append({"comparison": f"{tag}: XGBoost minus random forest", "sealed_auc_gap": roc_auc_score(y, pb) - roc_auc_score(y, pa),
                     "ci_low": lo, "ci_high": hi})
    print(f"[tuning] {tag}: XGBoost minus random forest {cmp_rows[-1]['sealed_auc_gap']:+.4f} (95% {lo:+.4f} to {hi:+.4f})")
pd.DataFrame(cmp_rows).round(4).to_csv(OUT / "xgboost_vs_forest_sealed.csv", index=False)

# chart: sealed AUC, default vs tuned (one dot per search seed)
BLUE, ORANGE, INK, MUTED, GRID_C = "#2a74d6", "#ea6a35", "#222222", "#6b6b6b", "#e4e4e4"
f, ax = plt.subplots(figsize=(8, 4.6))
names = {"logreg": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}
for i, kind in enumerate(names):
    r = res[res.model == kind]
    ax.scatter([i - 0.12], [r.sealed_auc_default.iloc[0]], s=110, facecolor="white", edgecolor=INK, lw=2, zorder=3,
               label="Default settings" if i == 0 else None)
    ax.scatter([i + 0.12] * len(r), r.sealed_auc_tuned, s=90, color=ORANGE, zorder=3, edgecolor="white",
               label="Tuned (one dot per search seed)" if i == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels(names.values())
ax.set_ylabel("AUC on sealed 2024-01..04 loans")
ax.set_title("Tuning gain on unseen loans: only the random forest improves", fontsize=11)
ax.grid(axis="y", color=GRID_C); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
f.text(0.01, 0.01, "Settings chosen on 2023 walk-forward folds only; sealed loans were never used to tune.", fontsize=8, color=MUTED)
f.tight_layout(rect=(0, 0.03, 1, 1)); f.savefig(OUT / "tuning_vs_default.png", dpi=200, facecolor="white"); plt.close(f)
print(f"[tuning] wrote {OUT}")
