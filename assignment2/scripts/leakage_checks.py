"""
Leakage evidence check (standalone, NOT an Airflow task). Reads the gold stores and the raw ledger and reports counts,
match rates and exceptions. Each line is tagged by what kind of evidence it is:
  VERIFIED         tested on the actual rows; FAIL if any exception is found
  SCREEN           a statistical screen that can raise a warning but cannot prove absence of leakage
  REVIEW           a design or code fact that rows cannot prove (documented, not tested)
  KNOWN LIMITATION a weakness that is measured and disclosed, not passed
A cohort is treated as mature when its start month + 5 months is on or before the latest label partition in the store
(the audit date). The script is meant to run after the backfill has finished.
Usage (project root, after the DAG has built the gold stores):  python3 scripts/leakage_checks.py
Output: console table and datamart/model_diagnostics/leakage_checks.csv
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import CLICK_COLS, DATA_DIR, DATAMART_DIR, GOLD_DIR, MODEL_BANK_DIR, MOB_THRESHOLD
from utils.model_utils import (FEATURES, NUMERIC, WF_FOLDS, WF_FOLD_MONTHS, add_months, build_pipeline,
                               load_training_frame)

OUT = DATAMART_DIR / "model_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
rows = []


def rec(kind, check, ok, detail):
    result = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    rows.append({"evidence": kind, "check": check, "result": result, "detail": detail})
    print(f"[{kind:16s}] {result:4s} {check}: {detail}")


# ---------- label store, raw ledger, audit date ----------
part = {Path(f).stem.replace("gold_label_store_", ""): f for f in glob.glob(str(GOLD_DIR / "label_store" / "*.parquet"))}
last_label_month = max(pd.Timestamp(k.replace("_", "-")) for k in part)
audit = last_label_month
print(f"[leakage] audit date {audit:%Y-%m-%d} (latest label partition); {len(part)} label partitions; "
      f"a cohort is mature when its start month + {MOB_THRESHOLD} months <= the audit date")

lab = pd.concat([pd.read_parquet(f) for f in sorted(part.values())], ignore_index=True)
lab["loan_start_date"] = pd.to_datetime(lab["loan_start_date"])
if "loan_id" not in lab:
    lab["loan_id"] = lab["Customer_ID"].astype(str) + "_" + lab["loan_start_date"].dt.strftime("%Y_%m_%d")
raw = pd.read_csv(DATA_DIR / "lms_loan_daily.csv")

nk = int(lab[["loan_id", "Customer_ID", "loan_start_date", "label"]].isna().any(axis=1).sum())
rec("VERIFIED", "label store: no null key or label values", nk == 0, f"{nk} rows with a null loan_id, Customer_ID, loan_start_date or label")
rec("VERIFIED", "label store: loan_id unique", not lab["loan_id"].duplicated().any(),
    f"{len(lab)} label rows, {lab['loan_id'].nunique()} unique loan_id")
d = int(lab.duplicated(["Customer_ID", "loan_start_date"]).sum())
rec("VERIFIED", "label store: (Customer_ID, loan_start_date) key unique", d == 0, f"{d} duplicate keys")

r5 = raw[raw["installment_num"] == MOB_THRESHOLD]
n_loans = raw["loan_id"].nunique()
rec("VERIFIED", f"ledger: at most one instalment-{MOB_THRESHOLD} row per loan", not r5["loan_id"].duplicated().any(),
    f"{r5['loan_id'].nunique()} of {n_loans} loans reach instalment {MOB_THRESHOLD}; {int(r5['loan_id'].duplicated().sum())} duplicates")
nk = int(r5[["loan_id", "snapshot_date"]].isna().any(axis=1).sum())
rec("VERIFIED", f"ledger instalment-{MOB_THRESHOLD} rows: no null loan_id or date", nk == 0, f"{nk} rows with a null key")
null_amt = r5[r5["overdue_amt"].isna()]
rec("VERIFIED", f"ledger: overdue_amt present at instalment {MOB_THRESHOLD}", len(null_amt) == 0,
    f"{len(null_amt)} loans with a missing amount" + (f" (first ids {list(null_amt['loan_id'].head(5))})" if len(null_amt) else ""))
valid = r5[r5["overdue_amt"].notna()].set_index("loan_id")
recomputed = (valid["overdue_amt"] > 0).astype(int).rename("raw")
m = lab.set_index("loan_id")["label"].astype(int).rename("stored").to_frame().join(recomputed, how="left")
no_ref = int(m["raw"].isna().sum())
mism = int((m["raw"].notna() & (m["stored"] != m["raw"])).sum())
rec("VERIFIED", "stored label = (overdue_amt > 0 at instalment 5) recomputed from the raw ledger, valid amounts only",
    mism == 0 and no_ref == 0, f"{len(m)} stored labels; {mism} mismatches; {no_ref} labels without a valid ledger instalment-5 amount")

bad_timing = 0
for k, f in part.items():
    dd = pd.read_parquet(f, columns=["loan_start_date"])
    exp = pd.Timestamp(k.replace("_", "-")) - pd.DateOffset(months=MOB_THRESHOLD)
    bad_timing += int((pd.to_datetime(dd["loan_start_date"]).dt.to_period("M") != exp.to_period("M")).sum())
rec("VERIFIED", f"label timing: each label partition holds only loans started {MOB_THRESHOLD} months earlier", bad_timing == 0,
    f"{len(part)} partitions; {bad_timing} loans in the wrong partition")

# ---------- feature store and the feature-to-label join ----------
fs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(GOLD_DIR / "feature_store" / "*.parquet")))], ignore_index=True)
fs["snapshot_date"] = pd.to_datetime(fs["snapshot_date"])
nk = int(fs[["Customer_ID", "snapshot_date"]].isna().any(axis=1).sum())
rec("VERIFIED", "feature store: no null key values", nk == 0, f"{nk} rows with a null Customer_ID or snapshot_date")
d = int(fs.duplicated(["Customer_ID", "snapshot_date"]).sum())
rec("VERIFIED", "feature store: (Customer_ID, snapshot_date) key unique", d == 0, f"{len(fs)} rows; {d} duplicate keys")

lab_k = lab[["Customer_ID", "loan_start_date", "label"]].rename(columns={"loan_start_date": "_start"})
lj = fs.merge(lab_k, left_on=["Customer_ID", "snapshot_date"], right_on=["Customer_ID", "_start"], how="outer", indicator=True)
both = lj[lj["_merge"] == "both"]
labels_wo_feat = int((lj["_merge"] == "right_only").sum())
rec("VERIFIED", "every label has exactly one feature row", labels_wo_feat == 0 and len(both) == len(lab),
    f"{len(both)} of {len(lab)} labels matched a feature row; {labels_wo_feat} labels without features")
feat_only = lj[lj["_merge"] == "left_only"]
mature = (feat_only["snapshot_date"] + pd.DateOffset(months=MOB_THRESHOLD)) <= audit        # empty-safe boolean mask
rec("VERIFIED", "every feature row that should be mature has a label", int(mature.sum()) == 0,
    f"{int(mature.sum())} mature feature rows without a label; {int((~mature).sum())} feature rows not yet mature at the audit date (expected)")
rec("VERIFIED", "joined snapshot_date = loan_start_date on every matched row", bool((both["snapshot_date"] == both["_start"]).all()),
    f"{len(both)} rows checked. This confirms the recorded date only; that each source value really existed at application "
    f"time needs source lineage (REVIEW item below)")

# ---------- predictions and model dates ----------
reg = json.loads((MODEL_BANK_DIR / "registry.json").read_text())["models"]
asof_of = {mm["model_id"]: pd.Timestamp(mm["trained_asof"]) for mm in reg}
pred = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(GOLD_DIR / "model_predictions" / "*.parquet")))], ignore_index=True)
pred["snapshot_date"] = pd.to_datetime(pred["snapshot_date"])
nk = int(pred[["loan_id", "Customer_ID", "snapshot_date", "model_id"]].isna().any(axis=1).sum())
rec("VERIFIED", "predictions: no null key values", nk == 0, f"{nk} rows with a null loan_id, Customer_ID, snapshot_date or model_id")
rec("VERIFIED", "predictions: loan_id unique", not pred["loan_id"].duplicated().any(), f"{len(pred)} predictions, {pred['loan_id'].nunique()} unique")

unknown = sorted(set(pred["model_id"]) - set(asof_of))
rec("VERIFIED", "every prediction's model_id exists in the registry", len(unknown) == 0,
    f"models used: {sorted(set(pred['model_id']))}" + (f"; NOT IN REGISTRY: {unknown}" if unknown else ""))
known = pred[pred["model_id"].isin(asof_of)]
early = int((known["snapshot_date"] < known["model_id"].map(asof_of)).sum())
rec("VERIFIED", "no prediction was made by a model trained after the scoring date", early == 0 and len(unknown) == 0,
    f"{early} of {len(known)} predictions earlier than their model's training date")

pm = pred.merge(lab[["loan_id", "label"]], on="loan_id", how="left")
bad = []
for c, g in pm.groupby("snapshot_date"):
    if add_months(c, MOB_THRESHOLD) <= audit:
        miss = int(g["label"].isna().sum())
        print(f"[leakage] cohort {c:%Y-%m}: {len(g)} predictions, {len(g) - miss} matched to a label, {miss} unmatched")
        if miss:
            bad.append((f"{c:%Y-%m}", miss))
    else:
        print(f"[leakage] cohort {c:%Y-%m}: {len(g)} predictions, label not mature at the audit date (excluded)")
rec("VERIFIED", "every mature prediction cohort matches a label", len(bad) == 0,
    "all mature cohorts fully matched" if not bad else "unmatched: " + ", ".join(f"{a}={b}" for a, b in bad))

# ---------- training windows ----------
for mdl in reg:
    tw, asof = pd.Timestamp(mdl["training_window"]["origination_to"]), pd.Timestamp(mdl["trained_asof"])
    rec("VERIFIED", f"{mdl['model_id']}: last training origination had a known label on the training date",
        add_months(tw, MOB_THRESHOLD) <= asof,
        f"last origination {tw:%Y-%m}, label known {add_months(tw, MOB_THRESHOLD):%Y-%m-%d}, trained as of {asof:%Y-%m-%d}")

# ---------- inputs ----------
repay = {"loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance", "installment_num", "label", "tenure"}
rec("VERIFIED", "no repayment-ledger column or label among the model inputs", not (set(FEATURES) & repay), f"{len(FEATURES)} inputs")
rec("VERIFIED", "clickstream columns are not model inputs", not (set(FEATURES) & set(CLICK_COLS)), "clickstream is monitored for coverage only")

single = {}
for c in NUMERIC:
    s = both[[c, "label"]].dropna()
    if s["label"].nunique() == 2 and s[c].nunique() > 1:
        a = roc_auc_score(s["label"], s[c])
        single[c] = max(a, 1 - a)
top = sorted(single.items(), key=lambda kv: -kv[1])[:3]
rec("SCREEN", "highest single-input AUC against the label (numeric inputs one at a time)", None,
    ", ".join(f"{k} {v:.3f}" for k, v in top) + ". A very high value would be a warning sign; a low one does not rule out leakage "
    "through combinations, categorical inputs or other paths")

# ---------- known limitation: unpurged folds ----------
for asof in ("2024-05-01", "2025-01-01"):
    df = load_training_frame(asof)
    months = sorted(df["orig_month"].unique())
    n_val = WF_FOLDS * WF_FOLD_MONTHS
    vm = [pd.Timestamp(x) for x in months[-n_val:]]
    out = []
    for i in range(0, n_val, WF_FOLD_MONTHS):
        fm = vm[i:i + WF_FOLD_MONTHS]
        out.append(int((df[df["orig_month"] < fm[0]]["label_date"] > fm[0]).sum()))
    rec("KNOWN LIMITATION", f"unpurged walk-forward folds, selection as of {asof}", None,
        f"training loans whose label was not yet known when each fold began: {out[0]} / {out[1]} / {out[2]} (folds 1 / 2 / 3). "
        f"Not removed because purging leaves too few training loans; applied equally to every candidate")

# ---------- review items (documented, not tested on rows) ----------
pipe = build_pipeline(next(m["algorithm"] for m in reg if m["status"] == "champion"))
rec("REVIEW", "imputers, scalers and encoders are fitted inside the model pipeline (training split only)", None,
    "pipeline steps: " + " -> ".join(n for n, _ in pipe.steps) + "; code review, not a data test")
rec("REVIEW", "calibrator and score-PSI reference use shuffled five-fold out-of-fold scores on the training loans", None,
    "design fact from train_and_select.py; shuffled folds are not a time-forward calibration test, so calibration on a later period is checked only through the monitoring reliability results")
rec("REVIEW", "feature-PSI references are the training feature values themselves", None, "design fact from train_and_select.py")
rec("REVIEW", "each source value was available at application time", None,
    "needs source lineage from the data owner; the snapshot-date check confirms recorded dates only")
rec("REVIEW", "the sealed backtest block and later cohorts were reused for model comparisons", None,
    "no untouched holdout exists; later analyses are post hoc")

pd.DataFrame(rows).to_csv(OUT / "leakage_checks.csv", index=False)
fails = [r for r in rows if r["result"] == "FAIL"]
print(f"[leakage] {len(rows)} lines written to {OUT / 'leakage_checks.csv'}; FAIL count {len(fails)}")
for r in fails:
    print(f"[leakage] FAIL: {r['check']} -> {r['detail']}")
if fails:
    sys.exit(1)
