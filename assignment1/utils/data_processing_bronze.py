"""
Bronze layer processing.

Bronze = raw landing zone. Preserves source records at snapshot level --
no business cleaning, no type coercion, no feature engineering -- and
persists them as partitioned CSV snapshots, one partition per snapshot_date.
"""
import os
from pyspark.sql import SparkSession

RAW_FILE_MAP = {
    "clickstream": "feature_clickstream.csv",
    "loan_daily": "lms_loan_daily.csv",
    "financials": "features_financials.csv",
    "attributes": "features_attributes.csv",
}


def process_bronze_table(table_name: str, snapshot_date: str, data_dir: str,
                          bronze_dir: str, spark: SparkSession) -> str:
    raw_filename = RAW_FILE_MAP[table_name]
    raw_path = os.path.join(data_dir, raw_filename)

    df = spark.read.csv(raw_path, header=True, inferSchema=True)
    df = df.filter(df.snapshot_date == snapshot_date)

    out_dir = os.path.join(bronze_dir, table_name)
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"bronze_{table_name}_{snapshot_date.replace('-', '_')}.csv"
    out_path = os.path.join(out_dir, out_name)

    n = df.count()
    df.toPandas().to_csv(out_path, index=False)
    print(f"[bronze] {table_name:12s} {snapshot_date}: {n:>6d} rows -> {out_path}")
    return out_path