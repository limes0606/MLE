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
    label_store = (
        silver_loan_daily_all
        .filter(F.col("installment_num") == MOB_THRESHOLD)
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
    feature_store = (
        silver_attributes_all.alias("a")
        .join(silver_financials_all.alias("f"), on=["Customer_ID", "snapshot_date"], how="inner")
        .drop(F.col("f.snapshot_date"))
    )

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