"""
Gold layer, built one snapshot_date (= Airflow run date D) at a time.

Label store   (partition D): loans that reach installment_num == MOB_THRESHOLD in
              month D, i.e. loans originated MOB_THRESHOLD months before D.
              label = 1 if overdue_amt > 0 at that checkpoint. The label only
              becomes known on D, so training at date T may only use partitions <= T.
Feature store (partition D): one row per customer who originated a loan in D --
              attributes + financials (+ clickstream when available) captured at
              origination. Name and SSN (PII) are dropped here, at gold.

DATA LEAKAGE GUARD
  - Features never touch loan_daily, so repayment columns cannot leak in.
  - Features come only from the snapshot dated at origination (D): no future data.
  - No population statistic (mean/median/mode) is computed here; imputers and
    scalers are fitted on the training split only, inside the training step.
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from utils.config import SILVER_DIR, GOLD_DIR, MOB_THRESHOLD, CLICK_COLS


def _silver_path(table, snapshot_date):
    return str(SILVER_DIR / table / f"silver_{table}_{snapshot_date.replace('-', '_')}.parquet")


def _exists(path):
    return os.path.exists(path)


def _write_gold(df, store, snapshot_date):
    out_dir = GOLD_DIR / store
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f"gold_{store}_{snapshot_date.replace('-', '_')}.parquet")
    df.coalesce(1).write.mode("overwrite").parquet(out_path)
    return out_path


def process_gold_label_store(spark, snapshot_date):
    src = _silver_path("loan_daily", snapshot_date)
    if not _exists(src):
        print(f"[gold] label_store   {snapshot_date}: no loan_daily silver partition -> skipped")
        return None

    loans = spark.read.parquet(src)
    short = loans.filter(F.col("tenure") < MOB_THRESHOLD).select("loan_id").distinct().count()
    if short > 0:
        print(f"[gold] WARNING: {short} loan(s) have tenure < {MOB_THRESHOLD}; the 'no censoring' assumption no longer holds.")

    label = (
        loans.filter(F.col("installment_num") == MOB_THRESHOLD)
             .filter(F.col("overdue_amt").isNotNull())
             .withColumn("label", F.when(F.col("overdue_amt") > 0, 1).otherwise(0))
             .select("loan_id", "Customer_ID", "loan_start_date", "label")
    )
    n = label.count()
    if n == 0:
        print(f"[gold] label_store   {snapshot_date}: no loans at installment {MOB_THRESHOLD} -> skipped")
        return None
    bad = label.filter(F.col("label") == 1).count()
    out = _write_gold(label, "label_store", snapshot_date)
    print(f"[gold] label_store   {snapshot_date}: {n:>5d} rows, bad rate={bad / n:.3f} -> {out}")
    return out


def process_gold_feature_store(spark, snapshot_date):
    a_path, f_path = _silver_path("attributes", snapshot_date), _silver_path("financials", snapshot_date)
    if not (_exists(a_path) and _exists(f_path)):
        print(f"[gold] feature_store {snapshot_date}: no attributes/financials silver partition -> skipped")
        return None

    attrs = spark.read.parquet(a_path).drop("snapshot_date")
    fin = spark.read.parquet(f_path).drop("snapshot_date")
    feats = attrs.join(fin, on="Customer_ID", how="inner").drop("Name", "SSN")

    c_path = _silver_path("clickstream", snapshot_date)
    if _exists(c_path):
        click = spark.read.parquet(c_path).select("Customer_ID", *CLICK_COLS)
        feats = feats.join(click, on="Customer_ID", how="left")
    else:  # no clickstream rows exist for customers originated after 2024-06 (cause not established): keep the schema, all-null columns
        for c in CLICK_COLS:
            feats = feats.withColumn(c, F.lit(None).cast(DoubleType()))

    for c in CLICK_COLS:
        feats = feats.withColumn(c, F.col(c).cast(DoubleType()))
    feats = feats.withColumn("snapshot_date", F.lit(snapshot_date).cast("date"))
    ordered = ["Customer_ID", "snapshot_date"] + [c for c in feats.columns if c not in ("Customer_ID", "snapshot_date")]
    feats = feats.select(*ordered)

    n = feats.count()
    out = _write_gold(feats, "feature_store", snapshot_date)
    print(f"[gold] feature_store {snapshot_date}: {n:>5d} rows, {len(feats.columns)} cols -> {out}")
    return out
