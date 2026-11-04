"""
Monthly monitoring: performance AND stability, written to the gold monitoring table.
Usage: python3 monitoring.py --snapshotdate 2024-10-01

On run date D:
  STABILITY   (cohort D, the loans scored this month; no labels needed)
      score PSI, feature PSI (22), feature missing rate (22), clickstream coverage,
      mean score, share flagged at the cutoff
  PERFORMANCE (cohort D-5 months; its MOB-5 labels become known on D)
      AUC / Gini / KS with 95% bootstrap CI, bad rate, mean score, calibration gap, Brier,
      and rolling 3-cohort AUC (pooled over cohorts D-5, D-6, D-7)

Output (long format): datamart/gold/model_monitoring/gold_model_monitoring_YYYY_MM_01.parquet
Status logic: a metric is 'green' below its amber threshold. A first breach is 'watch'; the
same level (or worse) in two consecutive runs becomes 'amber' (level 1) or 'red' (level 2).
Clickstream coverage is a pipeline fault, so it goes red immediately.
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from utils.config import CLICK_COLS, GOLD_DIR, MODEL_BANK_DIR
from utils.model_utils import add_months, auc_bootstrap, ks_stat, load_features_month, month_str
from utils.psi import missing_rate, psi

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
ds = pd.Timestamp(p.parse_args().snapshotdate)

PRED_DIR = GOLD_DIR / "model_predictions"
MON_DIR = GOLD_DIR / "model_monitoring"
MON_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = f"monitoring_{ds:%Y_%m_01}"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

# thresholds: (amber, red)
PSI_T = (0.10, 0.25)
AUC_DROP_T = (0.03, 0.05)       # drop of rolling AUC below baseline
CALIB_T = (0.03, 0.05)          # |mean score - bad rate|
COVER_T = (0.99, 0.95)          # clickstream coverage floor (amber below 99%, red below 95%)

_bundles = {}


def bundle_for(model_id):
    if model_id not in _bundles:
        with open(MODEL_BANK_DIR / f"{model_id}.pkl", "rb") as f:
            _bundles[model_id] = pickle.load(f)
    return _bundles[model_id]


def read_pred(month):
    path = PRED_DIR / f"gold_model_predictions_{month_str(month)}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def read_labels(label_month):
    path = GOLD_DIR / "label_store" / f"gold_label_store_{month_str(label_month)}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def level_high_is_bad(v, t):
    return 2 if v > t[1] else 1 if v > t[0] else 0


rows = []


def add(metric, value, cohort, segment="all", ci=(None, None), n=None, n_bad=None, ref="", level=0,
        thr=(None, None), immediate=False, mature=True, model_id=None):
    rows.append(dict(model_id=model_id, scoring_month=ds.date(), cohort_month=pd.Timestamp(cohort).date(),
                     metric_name=metric, segment=segment, metric_value=None if value is None else float(value),
                     ci_low=ci[0], ci_high=ci[1], n_loans=n, n_bad=n_bad, reference_window=ref,
                     threshold_amber=thr[0], threshold_red=thr[1], raw_level=int(level),
                     immediate=immediate, label_mature_flag=mature, computed_at=NOW, pipeline_run_id=RUN_ID))


# ---------------- stability: cohort scored this month ----------------
pred = read_pred(ds)
feat = load_features_month(ds)
if pred is not None and feat is not None:
    b = bundle_for(pred.model_id.iloc[0])
    mid, ref = b["model_id"], f"training {b['training_window']['origination_from'][:7]}..{b['training_window']['origination_to'][:7]} (out-of-fold)"
    n = len(pred)
    v = psi(b["psi_reference"]["score"], pred["score"])
    add("score_psi", v, ds, ref=ref, n=n, level=level_high_is_bad(v, PSI_T), thr=PSI_T, mature=False, model_id=mid)
    add("mean_score", pred["score"].mean(), ds, n=n, mature=False, model_id=mid)
    add("flagged_rate", pred["prediction"].mean(), ds, n=n, mature=False, model_id=mid)
    for c, r in b["psi_reference"]["features"].items():
        v = psi(r, feat[c])
        add("feature_psi", v, ds, segment=c, ref=ref, n=n, level=level_high_is_bad(v, PSI_T), thr=PSI_T,
            mature=False, model_id=mid)
        add("missing_rate", missing_rate(feat[c]), ds, segment=c, n=n, mature=False, model_id=mid)
    # clickstream: not a model input, but missing clickstream rows for newly originated customers is a data-quality incident
    cover = 1.0 - float(feat[CLICK_COLS].isna().all(axis=1).mean())
    lvl = 2 if cover < COVER_T[1] else 1 if cover < COVER_T[0] else 0
    add("clickstream_coverage", cover, ds, n=n, level=lvl, thr=COVER_T, immediate=True, mature=False, model_id=mid)

# ---------------- performance: cohort D-5 (labels known today) ----------------
cohort = add_months(ds, -5)
cp, lab = read_pred(cohort), read_labels(ds)
if cp is not None and lab is not None:
    b = bundle_for(cp.model_id.iloc[0])
    mid = b["model_id"]
    d = cp.merge(lab[["loan_id", "label"]], on="loan_id", how="inner")
    cov = len(d) / len(cp)
    add("label_coverage", cov, cohort, n=len(cp), ref="labelled loans / scored loans",
        level=2 if cov < 0.95 else 1 if cov < 0.99 else 0, thr=(0.99, 0.95), model_id=mid, immediate=True)
    if len(d) == 0:
        print(f"[monitoring] WARNING: no labels match the {cohort:%Y-%m} predictions; performance skipped, red label_coverage written")
    assert (add_months(cohort, 5) <= ds), "label from the future"
    if len(d) > 0:
        auc, lo, hi, _ = auc_bootstrap(d["label"], d["score"])
        base = b["baseline"]["auc"]
        ref = f"walk-forward baseline AUC {base:.3f}"
        nb = int(d["label"].sum())
        add("auc", auc, cohort, ci=(lo, hi), n=len(d), n_bad=nb, ref=ref, model_id=mid)
        add("gini", 2 * auc - 1, cohort, n=len(d), n_bad=nb, model_id=mid)
        add("ks", ks_stat(d["label"], d["score"]), cohort, n=len(d), n_bad=nb, model_id=mid)
        add("bad_rate", d["label"].mean(), cohort, n=len(d), n_bad=nb, model_id=mid)
        add("brier", brier_score_loss(d["label"], d["default_probability"]), cohort, n=len(d), n_bad=nb,
            ref="calibrated default_probability", model_id=mid)
        diff = d["default_probability"].to_numpy() - d["label"].to_numpy()
        gap = float(diff.mean())
        gap_se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        # alert only when the gap is both material (> threshold) and statistically distinguishable from 0
        lvl = level_high_is_bad(abs(gap), CALIB_T) if abs(gap) > 1.96 * gap_se else 0
        add("calibration_gap", gap, cohort, ci=(gap - 1.96 * gap_se, gap + 1.96 * gap_se), n=len(d), n_bad=nb,
            ref="mean default_probability - bad rate (95% CI)", level=lvl, thr=CALIB_T, model_id=mid)
        # rolling 3-cohort AUC (needs all three cohorts scored and labelled)
        pool = []
        for k in range(3):
            c = add_months(ds, -5 - k)
            pk, lk = read_pred(c), read_labels(add_months(c, 5))
            if pk is None or lk is None:
                break
            pool.append(pk.merge(lk[["loan_id", "label"]], on="loan_id", how="inner"))
        if len(pool) == 3:
            pool = pd.concat(pool)
            if pool["model_id"].nunique() != 1:      # a champion swap inside the window: scores are not comparable
                pool = None
            if pool is not None:
                r_auc, r_lo, r_hi, _ = auc_bootstrap(pool["label"], pool["score"])
                drop = base - r_auc
                add("rolling_3m_auc", r_auc, cohort, ci=(r_lo, r_hi), n=len(pool), n_bad=int(pool["label"].sum()),
                    ref=ref, level=level_high_is_bad(drop, AUC_DROP_T),
                    thr=(base - AUC_DROP_T[0], base - AUC_DROP_T[1]), model_id=mid)
else:
    print(f"[monitoring] {ds:%Y-%m}: no live performance yet (needs predictions for cohort {cohort:%Y-%m} and its labels)")

# ---------------- sealed backtest: originations after the training window, before the first live month ----------------
# These loans were not used to fit the champion or in the pipeline's selection step. Later tuning and comparison analyses
# reused them and the challenger's training window includes them, so they are a backtest of the champion only, not an
# untouched test set. Their labels mature at runs 2024-06..09, so the
# champion is scored on them directly. Segment 'backtest_sealed' keeps them out of the live rolling-AUC alerts.
champ_path = MODEL_BANK_DIR / "champion.json"
if champ_path.exists():
    cb = bundle_for(json.loads(champ_path.read_text())["model_id"])
    t_end = pd.Timestamp(cb["training_window"]["origination_to"])
    if t_end < cohort < pd.Timestamp(cb["trained_asof"]) and read_pred(cohort) is None:
        bf, bl = load_features_month(cohort), read_labels(ds)
        if bf is not None and bl is not None:
            cols = cb["features"]["numeric"] + cb["features"]["categorical"]
            bs = pd.DataFrame({"loan_id": (bf["Customer_ID"] + "_" + pd.to_datetime(bf["snapshot_date"]).dt.strftime("%Y_%m_%d")).values,
                               "score": cb["pipeline"].predict_proba(bf[cols])[:, 1]})
            bs = bs.merge(bl[["loan_id", "label"]], on="loan_id", how="inner")
            assert (add_months(cohort, 5) <= ds), "label from the future"
            a, lo, hi, _ = auc_bootstrap(bs["label"], bs["score"])
            nb = int(bs["label"].sum())
            rf = f"sealed backtest: originations after training window; baseline {cb['baseline']['auc']:.3f}"
            add("auc", a, cohort, segment="backtest_sealed", ci=(lo, hi), n=len(bs), n_bad=nb, ref=rf, model_id=cb["model_id"])
            add("bad_rate", bs["label"].mean(), cohort, segment="backtest_sealed", n=len(bs), n_bad=nb, model_id=cb["model_id"])
            add("brier", brier_score_loss(bs["label"], bs["score"]), cohort, segment="backtest_sealed", n=len(bs), n_bad=nb,
                ref="raw score (champion scored before calibration)", model_id=cb["model_id"])
            print(f"[monitoring] sealed backtest cohort {cohort:%Y-%m}: AUC {a:.3f} [{lo:.3f}, {hi:.3f}] n={len(bs)}")

# ---------------- challenger shadow comparison (out-of-time, never promoted automatically) ----------------
# A challenger is scored only on cohorts originated AFTER its training window, so it is compared with the champion
# on loans neither model has seen. Results are cumulative over those cohorts; promotion needs a human decision.
reg_path = MODEL_BANK_DIR / "registry.json"
if reg_path.exists() and champ_path.exists():
    champ_id = json.loads(champ_path.read_text())["model_id"]
    for m in json.loads(reg_path.read_text())["models"]:
        if m["status"] != "challenger" or pd.Timestamp(m["trained_asof"]) > ds:
            continue
        xb = bundle_for(m["model_id"])
        first = add_months(pd.Timestamp(xb["training_window"]["origination_to"]), 1)
        last = add_months(ds, -5)
        pools = []
        for c in pd.date_range(first, last, freq="MS"):
            cpred, fc, lc = read_pred(c), load_features_month(c), read_labels(add_months(c, 5))
            if cpred is None or fc is None or lc is None or (cpred.model_id != champ_id).any():
                continue
            cols = xb["features"]["numeric"] + xb["features"]["categorical"]
            sh = pd.DataFrame({"loan_id": (fc["Customer_ID"] + "_" + pd.to_datetime(fc["snapshot_date"]).dt.strftime("%Y_%m_%d")).values,
                               "challenger": xb["pipeline"].predict_proba(fc[cols])[:, 1]})
            pools.append(cpred[["loan_id", "score"]].merge(sh, on="loan_id").merge(lc[["loan_id", "label"]], on="loan_id"))
        if not pools:
            continue
        pool = pd.concat(pools, ignore_index=True)
        y, a_c, a_x = pool["label"].to_numpy(), pool["score"].to_numpy(), pool["challenger"].to_numpy()
        rng = np.random.default_rng(42)
        lift = []
        for _ in range(500):
            ix = rng.integers(0, len(y), len(y))
            if y[ix].min() != y[ix].max():
                lift.append(roc_auc_score(y[ix], a_x[ix]) - roc_auc_score(y[ix], a_c[ix]))
        pt = roc_auc_score(y, a_x) - roc_auc_score(y, a_c)
        ci = (float(np.percentile(lift, 2.5)), float(np.percentile(lift, 97.5)))
        nb = int(y.sum())
        rf = f"cumulative out-of-time cohorts {first:%Y-%m}..{last:%Y-%m}; promotion rule: lift >= +0.010, CI above 0, sign-off"
        add("challenger_auc", roc_auc_score(y, a_x), last, segment=m["model_id"], n=len(pool), n_bad=nb, ref=rf, model_id=m["model_id"])
        add("champion_auc_same_loans", roc_auc_score(y, a_c), last, segment=m["model_id"], n=len(pool), n_bad=nb, ref=rf, model_id=champ_id)
        add("challenger_auc_lift", pt, last, segment=m["model_id"], ci=ci, n=len(pool), n_bad=nb, ref=rf, model_id=m["model_id"])
        gate = bool(pt >= 0.010 and ci[0] > 0)
        add("challenger_promotion_gate", pt, last, segment=m["model_id"], ci=ci, n=len(pool), n_bad=nb, thr=(0.010, None),
            level=1 if gate else 0, immediate=True, model_id=m["model_id"],
            ref=("numeric gate crossed under current calculation (cumulative, overlapping cohorts, loan-level CI); human review required"
                 if gate else "numeric gate (lift >= +0.010 and CI above 0) not crossed"))
        print(f"[monitoring] challenger {m['model_id']} vs champion on {len(pool)} unseen loans: lift {pt:+.4f} (95% CI {ci[0]:+.4f} to {ci[1]:+.4f}); promotion needs lift >= +0.010 and CI > 0, plus sign-off")

if not rows:
    sys.exit(f"[monitoring] FAILED: nothing to monitor for {ds:%Y-%m-%d} (no predictions and no matured labels).")

out = pd.DataFrame(rows)

# ---------------- status: consecutive-breach rule using last month's raw levels ----------------
prev_path = MON_DIR / f"gold_model_monitoring_{month_str(add_months(ds, -1))}.parquet"
prev = pd.read_parquet(prev_path)[["metric_name", "segment", "raw_level"]] if prev_path.exists() else None
prev_lvl = {} if prev is None else {(r.metric_name, r.segment): r.raw_level for r in prev.itertuples()}
LABEL = {0: "green", 1: "amber", 2: "red"}


def status(r):
    if r.raw_level == 0:
        return "green"
    prev = prev_lvl.get((r.metric_name, r.segment), 0)
    if r.immediate or prev >= r.raw_level:
        return LABEL[r.raw_level]
    if prev >= 1:                      # already breaching last month and now worse: at least amber
        return "amber"
    return "watch"


out["status"] = [status(r) for r in out.itertuples()]
out.loc[out.threshold_amber.isna(), "status"] = "info"      # metrics with no threshold are context only
out["ci_low"] = pd.to_numeric(out["ci_low"], errors="coerce")
out["ci_high"] = pd.to_numeric(out["ci_high"], errors="coerce")
out["n_bad"] = pd.to_numeric(out["n_bad"], errors="coerce").astype("Int64")
out["scoring_month"] = pd.to_datetime(out["scoring_month"])
out["cohort_month"] = pd.to_datetime(out["cohort_month"])
path = MON_DIR / f"gold_model_monitoring_{month_str(ds)}.parquet"
out.to_parquet(path, index=False)

perf = out[out.metric_name.isin(["auc", "rolling_3m_auc"])]
stab = out[out.metric_name.isin(["score_psi", "clickstream_coverage"])]
print(f"[monitoring] {ds:%Y-%m}: {len(out)} metrics -> {path}")
for r in pd.concat([perf, stab]).itertuples():
    ci = f" [{r.ci_low:.3f}, {r.ci_high:.3f}]" if pd.notna(r.ci_low) else ""
    print(f"   {r.metric_name:22s} cohort {r.cohort_month:%Y-%m}  {r.metric_value:.3f}{ci}  {r.status}")
alerts = out[out.status.isin(["amber", "red"])]
if len(alerts):
    print(f"[monitoring] ALERTS: " + "; ".join(f"{a.metric_name}[{a.segment}]={a.status}" for a in alerts.itertuples()))
