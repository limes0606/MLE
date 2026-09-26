"""
Gold layer processing.

Gold = business-ready tables: the label store (one row per loan, labeled at
a fixed mob checkpoint) and the feature store (one row per customer, built
from attributes + financials + origination-date clickstream).
"""
import os
import pyspark.sql.functions as F

MOB_THRESHOLD = 6


def process_gold_label_store(silver_loan_daily_all, gold_dir):
    """
    Label definition: bad=1 if overdue_amt > 0 at installment_num == 6
    (fixed mob=6 checkpoint), else bad=0. All loans have tenure=10 months
    and are fully observed, so every loan reaches mob=6 with no censoring.
    This is a point-in-time definition, not a cumulative one: a loan
    overdue at mob=6 but cured by mob=7 still counts as bad, and vice
    versa -- a deliberate simplification, stated explicitly here as an
    assumption.

    silver_loan_daily_all: the UNION of all loan_daily silver snapshots
    (read with a wildcard path), not a single monthly partition -- we need
    every loan's row at installment_num == MOB_THRESHOLD, and different
    loans hit that mob at different calendar snapshot_dates.
    """
    label_store = (
        silver_loan_daily_all
        .filter(F.col("installment_num") == MOB_THRESHOLD)
        .filter(F.col("overdue_amt").isNotNull())
        .withColumn("label", F.when(F.col("overdue_amt") > 0, 1).otherwise(0))
        .select("loan_id", "Customer_ID", "loan_start_date", "label")
    )

    out_dir = os.path.join(gold_dir, "label_store")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gold_label_store.parquet")
    label_store.write.mode("overwrite").parquet(out_path)
    n = label_store.count()
    bad_rate = label_store.filter(F.col("label") == 1).count() / n
    print(f"[gold] label_store: {n:>6d} rows, bad rate={bad_rate:.3f} -> {out_path}")
    return label_store


def process_gold_feature_store(silver_attributes_all, silver_financials_all,
                                silver_clickstream_all, gold_dir):
    """
    Feature store: built from attributes + financials, joined on
    (Customer_ID, snapshot_date) -- both are captured exactly at loan
    origination for every customer, so this join is always aligned with
    no leakage risk. Clickstream is left-joined after filtering to rows
    where its own snapshot_date matches the customer's loan_start_date;
    only ~72% of customers (8,974 / 12,500) have clickstream coverage, so
    the remaining customers keep null clickstream features rather than
    being dropped from the feature store.
    """
    feature_store = (
        silver_attributes_all.alias("a")
        .join(silver_financials_all.alias("f"), on=["Customer_ID", "snapshot_date"], how="inner")
    )

    feature_store = feature_store.drop("Name", "SSN")

    click = silver_clickstream_all.withColumnRenamed("snapshot_date", "click_snapshot_date")
    feature_store = feature_store.join(
        click,
        on=(feature_store["Customer_ID"] == click["Customer_ID"]) &
           (feature_store["snapshot_date"] == click["click_snapshot_date"]),
        how="left",
    ).drop(click["Customer_ID"]).drop("click_snapshot_date")

    out_dir = os.path.join(gold_dir, "feature_store")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gold_feature_store.parquet")
    feature_store.write.mode("overwrite").parquet(out_path)
    print(f"[gold] feature_store: {feature_store.count():>6d} rows -> {out_path}")
    return feature_store