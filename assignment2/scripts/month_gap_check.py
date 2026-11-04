"""
Month-level sensitivity check of the random forest vs XGBoost comparison (standalone, NOT an Airflow task).
POST HOC and EXPLORATORY: these loans have already informed model comparisons, so this is not a confirmation test.

It repeats the walk-forward folds used at selection (same loans, same unpurged folds) for two training dates,
2024-05-01 (champion selection) and 2025-01-01 (challenger selection), trains random forest and XGBoost in memory,
and reports for each validation month: loans, bad rate, AUC of each model and the paired AUC gap (XGBoost minus RF).
It also reports the fold gaps, the pooled loan-level bootstrap gap (as used at selection) and a month-level bootstrap of
the mean monthly gap (months resampled), which keeps month-to-month variation in the interval.
Nothing is written to the model bank or the registry.
Usage (project root, after the DAG has built the gold stores):  python3 scripts/month_gap_check.py
Outputs in datamart/model_diagnostics/ : month_gap_by_month.csv, month_gap_summary.csv, month_gap_by_month.png
"""
import argparse
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
from utils.model_utils import FEATURES, SEED, WF_FOLDS, WF_FOLD_MONTHS, build_pipeline, load_training_frame

ap = argparse.ArgumentParser()
ap.add_argument("--dates", nargs="+", default=["2024-05-01", "2025-01-01"])
ap.add_argument("--boot", type=int, default=5000)
args = ap.parse_args()
OUT = DATAMART_DIR / "model_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
A, B = "random_forest", "xgboost"          # gap = B minus A

by_month, summary = [], []
for ds in args.dates:
    df = load_training_frame(ds)
    months = [pd.Timestamp(m) for m in sorted(df["orig_month"].unique())]
    n_val = WF_FOLDS * WF_FOLD_MONTHS
    val_months = months[-n_val:]
    folds = [val_months[i:i + WF_FOLD_MONTHS] for i in range(0, n_val, WF_FOLD_MONTHS)]
    frames = []
    for i, fm in enumerate(folds, 1):
        tr, va = df[df.orig_month < fm[0]], df[df.orig_month.isin(fm)]
        part = va[["orig_month", "label"]].copy()
        part["fold"] = i
        for name in (A, B):
            part[name] = build_pipeline(name).fit(tr[FEATURES], tr["label"]).predict_proba(va[FEATURES])[:, 1]
        frames.append(part)
        print(f"[month-gap] {ds}: fold {i} trained on {len(tr)} loans, validated on {len(va)} ({fm[0]:%Y-%m}..{fm[-1]:%Y-%m})")
    oof = pd.concat(frames, ignore_index=True)

    for m, g in oof.groupby("orig_month"):
        a, b = roc_auc_score(g["label"], g[A]), roc_auc_score(g["label"], g[B])
        by_month.append({"selection_date": ds, "fold": int(g["fold"].iloc[0]), "orig_month": f"{m:%Y-%m}",
                         "n_loans": len(g), "bad_rate": round(float(g["label"].mean()), 4),
                         "auc_random_forest": round(a, 4), "auc_xgboost": round(b, 4), "gap_xgb_minus_rf": round(b - a, 4)})
    mdf = pd.DataFrame([r for r in by_month if r["selection_date"] == ds])
    gaps = mdf["gap_xgb_minus_rf"].to_numpy()

    fold_gaps = [roc_auc_score(g["label"], g[B]) - roc_auc_score(g["label"], g[A]) for _, g in oof.groupby("fold")]
    pooled = roc_auc_score(oof["label"], oof[B]) - roc_auc_score(oof["label"], oof[A])

    rng = np.random.default_rng(SEED)
    y, pa, pb = oof["label"].to_numpy(), oof[A].to_numpy(), oof[B].to_numpy()
    loan_boot = []
    for _ in range(args.boot):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].min() != y[ix].max():
            loan_boot.append(roc_auc_score(y[ix], pb[ix]) - roc_auc_score(y[ix], pa[ix]))
    month_boot = [gaps[rng.integers(0, len(gaps), len(gaps))].mean() for _ in range(args.boot)]
    summary.append({
        "selection_date": ds, "n_months": len(gaps),
        "months_xgb_ahead": int((gaps > 0).sum()), "mean_monthly_gap": round(float(gaps.mean()), 4),
        "sd_monthly_gap": round(float(gaps.std(ddof=1)), 4), "min_monthly_gap": round(float(gaps.min()), 4),
        "max_monthly_gap": round(float(gaps.max()), 4),
        "fold_gaps": " / ".join(f"{x:+.4f}" for x in fold_gaps), "pooled_gap": round(float(pooled), 4),
        "pooled_loan_ci_low": round(float(np.percentile(loan_boot, 2.5)), 4),
        "pooled_loan_ci_high": round(float(np.percentile(loan_boot, 97.5)), 4),
        "month_boot_ci_low": round(float(np.percentile(month_boot, 2.5)), 4),
        "month_boot_ci_high": round(float(np.percentile(month_boot, 97.5)), 4)})
    s = summary[-1]
    print(f"[month-gap] {ds}: XGBoost ahead in {s['months_xgb_ahead']}/{s['n_months']} months; mean monthly gap {s['mean_monthly_gap']:+.4f} "
          f"(SD {s['sd_monthly_gap']:.4f}, range {s['min_monthly_gap']:+.4f} to {s['max_monthly_gap']:+.4f})")
    print(f"[month-gap] {ds}: pooled gap {s['pooled_gap']:+.4f}; loan-level 95% CI {s['pooled_loan_ci_low']:+.4f} to {s['pooled_loan_ci_high']:+.4f}; "
          f"month-level 95% CI of the mean gap {s['month_boot_ci_low']:+.4f} to {s['month_boot_ci_high']:+.4f}")

pd.DataFrame(by_month).to_csv(OUT / "month_gap_by_month.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "month_gap_summary.csv", index=False)

bm = pd.DataFrame(by_month)
fig, axes = plt.subplots(1, len(args.dates), figsize=(5.5 * len(args.dates), 3.6), sharey=True)
for ax, ds in zip(np.atleast_1d(axes), args.dates):
    d = bm[bm.selection_date == ds]
    ax.bar(d["orig_month"], d["gap_xgb_minus_rf"], color=["#2a6f97" if v > 0 else "#c0392b" for v in d["gap_xgb_minus_rf"]])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Selection as of {ds}", fontsize=10)
    ax.tick_params(axis="x", rotation=60, labelsize=8)
np.atleast_1d(axes)[0].set_ylabel("AUC gap: XGBoost minus RF")
fig.suptitle("Monthly paired AUC gap (post hoc, exploratory)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "month_gap_by_month.png", dpi=150)
print(f"[month-gap] wrote {OUT}")
