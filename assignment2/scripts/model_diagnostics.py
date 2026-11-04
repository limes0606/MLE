"""
Model diagnostics (standalone, NOT an Airflow task; does not change the champion).

1. What drives the predictions: permutation importance of the champion on loans it has never seen
   (originations 2024-01..2025-01). Each of the 22 input features is shuffled in turn; the drop in AUC is its importance.
2. Retrain stability: the champion's settings (and the challenger's) are retrained on 2023 loans with 10 different
   random seeds and scored on the same unseen loans, to show how much AUC moves from the training randomness alone.
Usage (project root, after the DAG has built the gold stores and the champion):  python3 scripts/model_diagnostics.py
Outputs in datamart/model_diagnostics/ : permutation_importance.csv, seed_stability.csv, permutation_importance.png
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import DATAMART_DIR, MODEL_BANK_DIR
from utils.model_utils import FEATURES, SEED, build_pipeline, load_training_frame

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--seeds", type=int, default=10)
args = ap.parse_args()
OUT = DATAMART_DIR / "model_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

df = load_training_frame("2025-06-01")                      # every origination 2023-01..2025-01 with a known label
train = df[df.orig_month <= "2023-12-01"]
unseen = df[df.orig_month > "2023-12-01"]
y = unseen.label.values
print(f"[diagnostics] train {len(train)} loans (2023); unseen {len(unseen)} loans (2024-01..2025-01)")

# 1) permutation importance of the shipped champion on unseen loans
champ = json.loads((MODEL_BANK_DIR / "champion.json").read_text())
bundle = pickle.load(open(MODEL_BANK_DIR / champ["file"], "rb"))
print(f"[diagnostics] champion {bundle['model_id']} ({bundle['algorithm']})")
pl = bundle["pipeline"]
base = roc_auc_score(y, pl.predict_proba(unseen[FEATURES])[:, 1])
pi = permutation_importance(pl, unseen[FEATURES], y, scoring="roc_auc", n_repeats=args.repeats, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"feature": FEATURES, "auc_drop_mean": pi.importances_mean, "auc_drop_sd": pi.importances_std}) \
        .sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
imp["share_of_total_drop"] = imp.auc_drop_mean.clip(lower=0) / imp.auc_drop_mean.clip(lower=0).sum()
imp.round(4).to_csv(OUT / "permutation_importance.csv", index=False)
print(f"[diagnostics] champion AUC on unseen loans {base:.4f}; top features by AUC drop when shuffled:")
print(imp.head(10).round(4).to_string(index=False))

BLUE, INK, MUTED, GRID = "#2a74d6", "#222222", "#6b6b6b", "#e4e4e4"
top = imp.head(10).iloc[::-1]
f, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(top.feature, top.auc_drop_mean, xerr=top.auc_drop_sd, color=BLUE, ecolor=INK, capsize=2)
ax.set_xlabel("AUC drop when the feature is shuffled (higher = model relies on it more)")
ax.set_title("What the model relies on (unseen loans)", fontsize=11, fontweight="bold", loc="left")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
f.text(0.01, 0.01, f"Champion on {len(unseen):,} loans from 2024-01 to 2025-01; {args.repeats} shuffles per feature; bars = mean, lines = 1 SD.", fontsize=8, color=MUTED)
f.tight_layout(rect=(0, 0.03, 1, 1)); f.savefig(OUT / "permutation_importance.png", dpi=200, facecolor="white"); plt.close(f)

# 2) retrain stability: same settings, different random seeds, same unseen loans
rows = []
for kind in ["random_forest", "xgboost"]:
    aucs = []
    for s in range(args.seeds):
        m = build_pipeline(kind)
        m.set_params(model__random_state=SEED + s)
        m.fit(train[FEATURES], train.label)
        aucs.append(roc_auc_score(y, m.predict_proba(unseen[FEATURES])[:, 1]))
        rows.append({"model": kind, "seed": SEED + s, "auc_unseen": aucs[-1]})
    print(f"[diagnostics] {kind}: AUC over {args.seeds} seeds mean {np.mean(aucs):.4f}, SD {np.std(aucs, ddof=1):.4f}, "
          f"range {min(aucs):.4f} to {max(aucs):.4f}")
pd.DataFrame(rows).round(4).to_csv(OUT / "seed_stability.csv", index=False)
print(f"[diagnostics] wrote {OUT}")
