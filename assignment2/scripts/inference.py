"""
Score the loans originated in the run month with the champion model and write the gold predictions table.
Usage: python3 inference.py --snapshotdate 2024-06-01

 - Model: champion from model_bank/champion.json (fails loudly if the bank is empty).
 - Scored population: gold feature store partition for the run month (new loans at origination).
 - Output: datamart/gold/model_predictions/gold_model_predictions_YYYY_MM_01.parquet
   columns: loan_id, Customer_ID, snapshot_date, model_id, score (raw model output), default_probability
            (calibrated score), prediction (default_probability >= cutoff, reference flag), cutoff, scored_at
 - No labels are read here, so inference cannot leak outcome information.
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone

import pandas as pd

from utils.config import GOLD_DIR, MODEL_BANK_DIR
from utils.model_utils import load_features_month, month_str

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
ds = p.parse_args().snapshotdate

champ_path = MODEL_BANK_DIR / "champion.json"
if not champ_path.exists():
    sys.exit(f"[inference] FAILED: no champion in {MODEL_BANK_DIR}. Run train_and_select first.")
champ = json.loads(champ_path.read_text())
with open(MODEL_BANK_DIR / champ["file"], "rb") as f:
    bundle = pickle.load(f)

# a model cannot score a month before it was trained
if pd.Timestamp(ds) < pd.Timestamp(bundle["trained_asof"]):
    sys.exit(f"[inference] FAILED: champion {bundle['model_id']} was trained as of "
             f"{bundle['trained_asof']}, after {ds}.")

df = load_features_month(ds)
if df is None or len(df) == 0:
    sys.exit(f"[inference] FAILED: no feature-store partition for {ds}.")

cols = bundle["features"]["numeric"] + bundle["features"]["categorical"]
if "calibrator" not in bundle:
    sys.exit(f"[inference] FAILED: champion {bundle['model_id']} has no calibrator; retrain with the current train_and_select.py.")
score = bundle["pipeline"].predict_proba(df[cols])[:, 1]
prob = bundle["calibrator"].predict(score)
out = pd.DataFrame({
    # same key as the label store: Customer_ID + "_" + loan start date (YYYY_MM_DD)
    "loan_id": (df["Customer_ID"] + "_" + pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y_%m_%d")).values,
    "Customer_ID": df["Customer_ID"].values,
    "snapshot_date": pd.to_datetime(df["snapshot_date"]).values,
    "model_id": bundle["model_id"],
    "score": score,
    "default_probability": prob,
    "prediction": (prob >= bundle["cutoff"]).astype(int),
    "cutoff": bundle["cutoff"],
    "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
})
out_dir = GOLD_DIR / "model_predictions"
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / f"gold_model_predictions_{month_str(ds)}.parquet"
out.to_parquet(path, index=False)
print(f"[inference] {ds}: scored {len(out)} loans with {bundle['model_id']} | mean score {score.mean():.3f}, mean default probability {prob.mean():.3f} | "
      f"flagged {out.prediction.mean():.1%} at cutoff {bundle['cutoff']} -> {path}")
