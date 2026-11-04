"""Gold label store partition for one snapshot date (loans reaching MOB=5 that month).
Usage: python3 gold_label_store.py --snapshotdate 2023-06-01"""
import argparse

from utils.gold import process_gold_label_store
from utils.spark_utils import get_spark

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
ds = p.parse_args().snapshotdate

spark = get_spark(f"gold_label_store_{ds}")
process_gold_label_store(spark, ds)
spark.stop()
