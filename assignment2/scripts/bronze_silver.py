"""Bronze + silver for all four source tables, for one snapshot date.
Usage: python3 bronze_silver.py --snapshotdate 2023-01-01"""
import argparse

from utils.bronze import process_bronze_table
from utils.config import SILVER_DIR
from utils.silver import SILVER_PROCESSORS
from utils.spark_utils import get_spark

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
ds = p.parse_args().snapshotdate

spark = get_spark(f"bronze_silver_{ds}")
for table in ["clickstream", "loan_daily", "financials", "attributes"]:
    bronze_path = process_bronze_table(table, ds, spark)
    if bronze_path is None:
        continue
    bronze_df = spark.read.csv(bronze_path, header=True, inferSchema=True)
    SILVER_PROCESSORS[table](bronze_df, ds, SILVER_DIR)
spark.stop()
