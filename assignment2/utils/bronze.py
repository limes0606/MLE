"""
Bronze = raw landing zone. One CSV per table per snapshot_date, rows copied
exactly as they arrive: no cleaning, no type coercion, no feature engineering.
"""
import os
from typing import Optional

from utils.config import DATA_DIR, BRONZE_DIR, RAW_FILE_MAP


def process_bronze_table(table_name: str, snapshot_date: str, spark) -> Optional[str]:
    """Returns the bronze CSV path, or None if the source has no rows for this date."""
    raw_path = str(DATA_DIR / RAW_FILE_MAP[table_name])
    df = spark.read.csv(raw_path, header=True, inferSchema=True)
    df = df.filter(df.snapshot_date == snapshot_date)

    n = df.count()
    if n == 0:
        print(f"[bronze] {table_name:12s} {snapshot_date}: no source rows -> skipped")
        return None

    out_dir = BRONZE_DIR / table_name
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f"bronze_{table_name}_{snapshot_date.replace('-', '_')}.csv")
    df.toPandas().to_csv(out_path, index=False)
    print(f"[bronze] {table_name:12s} {snapshot_date}: {n:>6d} rows -> {out_path}")
    return out_path
