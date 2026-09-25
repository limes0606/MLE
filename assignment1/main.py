"""
main.py -- orchestrates the full bronze -> silver -> gold pipeline for
CS611 Assignment 1 (Data Processing Pipelines).

Usage:
    python main.py

Builds a `datamart/` folder containing:
    datamart/bronze/{clickstream,loan_daily,financials,attributes}/...
    datamart/silver/{clickstream,loan_daily,financials,attributes}/...
    datamart/gold/{label_store,feature_store}/...
"""
import os

from pyspark.sql import SparkSession

from utils.data_processing_bronze import process_bronze_table, RAW_FILE_MAP
from utils.data_processing_silver import SILVER_PROCESSORS
from utils.data_processing_gold import (
    process_gold_label_store,
    process_gold_feature_store,
    MOB_THRESHOLD,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATAMART_DIR = os.path.join(BASE_DIR, "datamart")
BRONZE_DIR = os.path.join(DATAMART_DIR, "bronze")
SILVER_DIR = os.path.join(DATAMART_DIR, "silver")
GOLD_DIR = os.path.join(DATAMART_DIR, "gold")


def get_snapshot_dates(spark, table_name):
    raw_path = os.path.join(DATA_DIR, RAW_FILE_MAP[table_name])
    df = spark.read.csv(raw_path, header=True, inferSchema=True)
    dates = [r["snapshot_date"] for r in df.select("snapshot_date").distinct().collect()]
    return sorted(str(d) for d in dates)


def run_bronze_silver(spark, table_name):
    dates = get_snapshot_dates(spark, table_name)
    print(f"\n=== {table_name}: {len(dates)} monthly snapshots ({dates[0]} -> {dates[-1]}) ===")
    silver_fn = SILVER_PROCESSORS[table_name]
    for d in dates:
        bronze_path = process_bronze_table(table_name, d, DATA_DIR, BRONZE_DIR, spark)
        bronze_df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        silver_fn(bronze_df, d, SILVER_DIR)


def main():
    spark = SparkSession.builder.appName("cs611_assignment1_pipeline").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs(DATAMART_DIR, exist_ok=True)

    print(f"Label definition in effect: bad=1 if overdue_amt > 0 at "
          f"installment_num == {MOB_THRESHOLD} (mob=6 fixed checkpoint).")

    for table_name in ["clickstream", "loan_daily", "financials", "attributes"]:
        run_bronze_silver(spark, table_name)

    silver_loan_daily_all = spark.read.parquet(os.path.join(SILVER_DIR, "loan_daily", "*.parquet"))
    silver_attributes_all = spark.read.parquet(os.path.join(SILVER_DIR, "attributes", "*.parquet"))
    silver_financials_all = spark.read.parquet(os.path.join(SILVER_DIR, "financials", "*.parquet"))
    silver_clickstream_all = spark.read.parquet(os.path.join(SILVER_DIR, "clickstream", "*.parquet"))

    process_gold_label_store(silver_loan_daily_all, GOLD_DIR)
    process_gold_feature_store(silver_attributes_all, silver_financials_all, silver_clickstream_all, GOLD_DIR)

    print("\nPipeline complete. See ./datamart for bronze/silver/gold tables.")
    spark.stop()


if __name__ == "__main__":
    main()