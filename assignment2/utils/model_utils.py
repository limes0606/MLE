"""
Shared modelling code: loading training data AS OF a date, preprocessing, candidate
models and evaluation metrics. Imported by training, inference and monitoring so the
same feature definitions and preprocessing are used everywhere.

DATA LEAKAGE GUARDS
  - Only loans whose MOB-5 label is already known on the training date are used
    (origination month <= date - 5 months). Partitions are read from an explicit
    list of months, never a wildcard, so future partitions on disk cannot leak in.
  - Imputers, clippers, scalers and encoders live inside the sklearn Pipeline and
    are fitted on the training split only.
  - Clickstream (fe_1..fe_20) is NOT used by the model: it exists only for loans
    originated up to 2024-06 and is missing for every later loan, which would
    create a false drift signal. Its coverage is monitored instead.
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from utils.config import GOLD_DIR, MOB_THRESHOLD, CLICK_COLS

FIRST_ORIGINATION = "2023-01-01"
WF_FOLDS = 3               # walk-forward selection: 3 folds ...
WF_FOLD_MONTHS = 3         # ... of 3 origination months each (last 9 mature months)
CUTOFF = 0.30               # business decision threshold stored with the model
SEED = 42

NUMERIC = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance", "Credit_History_Age_Months", "Num_Type_of_Loan",
]
CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
FEATURES = NUMERIC + CATEGORICAL


def month_str(ts):
    return pd.Timestamp(ts).strftime("%Y_%m_01")


def add_months(ts, n):
    return pd.Timestamp(ts) + pd.DateOffset(months=n)


def origination_months(asof):
    """Origination months whose MOB-5 label is known on `asof`."""
    last = add_months(asof, -MOB_THRESHOLD)
    return list(pd.date_range(FIRST_ORIGINATION, last, freq="MS"))


def _read_partition(store, month):
    path = GOLD_DIR / store / f"gold_{store}_{month_str(month)}.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_features_month(month):
    """Gold feature-store partition for loans originated in `month` (or None)."""
    df = _read_partition("feature_store", month)
    if df is not None:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def load_training_frame(asof):
    """Training table as of `asof`: features at origination + MOB-5 label."""
    frames = []
    for m in origination_months(asof):
        feat = load_features_month(m)
        lab = _read_partition("label_store", add_months(m, MOB_THRESHOLD))
        if feat is None or lab is None:
            raise FileNotFoundError(f"missing gold partition for origination month {pd.Timestamp(m):%Y-%m} "
                                    f"(feature store or label store); refusing to train on a shorter window")
        label_date = add_months(m, MOB_THRESHOLD)
        assert label_date <= pd.Timestamp(asof), "label from the future"
        lab["loan_start_date"] = pd.to_datetime(lab["loan_start_date"])
        df = feat.merge(lab[["Customer_ID", "loan_start_date", "label"]], on="Customer_ID", how="inner")
        assert (df["loan_start_date"] == df["snapshot_date"]).all(), "feature snapshot != origination date"
        df["orig_month"] = pd.Timestamp(m)
        df["label_date"] = label_date
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert out["label_date"].max() <= pd.Timestamp(asof)
    return out


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips each column to its training 1st/99th percentile (tames outliers for linear models)."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_pipeline(kind):
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", _onehot())])
    if kind == "logreg":
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("clip", QuantileClipper()), ("scale", StandardScaler())])
        model = LogisticRegression(C=1.0, max_iter=2000)
    elif kind == "random_forest":
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=2, random_state=SEED)
    elif kind == "xgboost":
        num = "passthrough"   # XGBoost handles missing values natively
        model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", n_jobs=2, random_state=SEED)
    else:
        raise ValueError(kind)
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    return Pipeline([("pre", pre), ("model", model)])


# Candidate algorithms evaluated by walk-forward selection (champion = highest pooled AUC).
CANDIDATES = ["logreg", "random_forest", "xgboost"]


class PlattCalibrator:
    """Sigmoid (Platt) calibration of a model score s: p = 1 / (1 + exp(-(slope * logit(s) + intercept))).
    Monotone, two parameters, ranking-preserving. Defined here (not in a script) so pickled bundles load anywhere."""

    @staticmethod
    def _logit(s):
        s = np.clip(np.asarray(s, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(s / (1 - s))

    def fit(self, scores, y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(scores).reshape(-1, 1), np.asarray(y))
        self.slope_, self.intercept_ = float(lr.coef_[0][0]), float(lr.intercept_[0])
        return self

    def predict(self, scores):
        return 1.0 / (1.0 + np.exp(-(self.slope_ * self._logit(scores) + self.intercept_)))


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def auc_bootstrap(y, p, n_boot=500, seed=SEED):
    """Returns (auc, ci_low, ci_high, standard_error)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return (float(roc_auc_score(y, p)), float(np.quantile(aucs, 0.025)),
            float(np.quantile(aucs, 0.975)), float(np.std(aucs)))


def evaluate(y, p):
    auc, lo, hi, se = auc_bootstrap(y, p)
    return {"auc": auc, "auc_ci_low": lo, "auc_ci_high": hi, "auc_se": se,
            "gini": 2 * auc - 1, "ks": ks_stat(y, p), "brier": float(brier_score_loss(y, p)),
            "n": int(len(y)), "n_bad": int(np.sum(y))}
