"""Shared paths and constants. PIPELINE_BASE overrides the root (used for local tests)."""
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("PIPELINE_BASE", Path(__file__).resolve().parents[1]))
DATA_DIR = BASE_DIR / "data"
DATAMART_DIR = BASE_DIR / "datamart"
BRONZE_DIR = DATAMART_DIR / "bronze"
SILVER_DIR = DATAMART_DIR / "silver"
GOLD_DIR = DATAMART_DIR / "gold"
MODEL_BANK_DIR = BASE_DIR / "model_bank"

RAW_FILE_MAP = {
    "clickstream": "feature_clickstream.csv",
    "loan_daily": "lms_loan_daily.csv",
    "financials": "features_financials.csv",
    "attributes": "features_attributes.csv",
}

MOB_THRESHOLD = 5            # label = overdue_amt > 0 at installment_num == 5
CLICK_COLS = [f"fe_{i}" for i in range(1, 21)]

# Decision-economics ASSUMPTIONS (not in the data). Used only for the cutoff trade-off chart.
# Loss on a bad loan is an illustrative exposure-based proxy: the outstanding balance at instalment 5 (observed exposure), less an assumed recovery rate.
# The ledger carries no interest, so the margin earned on a good loan is a scenario input.
MARGIN_SCENARIOS = {                    # net margin earned per good loan, in loan currency units
    "650 (about 14% interest)": 650,
    "1,500": 1500,
    "3,000": 3000,
}
RECOVERY_RATES = (0.0, 0.30)            # scenarios: share of the outstanding balance recovered on a bad loan
RECOVERY_RATE = 0.0                     # share of the outstanding balance recovered on a bad loan
REFERENCE_CUTOFF = 0.30                 # placeholder cutoff stored with the model; the decision is the business's
