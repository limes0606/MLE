"""Gold feature store partition for one snapshot date (loans originated that month).
Usage: python3 gold_feature_store.py --snapshotdate 2023-01-01"""
import argparse

from utils.gold import process_gold_feature_store
from utils.spark_utils import get_spark

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
ds = p.parse_args().snapshotdate

spark = get_spark(f"gold_feature_store_{ds}")
process_gold_feature_store(spark, ds)
spark.stop()
