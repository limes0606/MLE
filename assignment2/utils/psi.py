"""
Population Stability Index (PSI) helpers.

A reference is built ONCE at training time from the training population; later
months are compared against it. Bin edges come from the reference only; the
first/last bin are open-ended, NaN gets its own bin (so a missing-data shift
shows up), and empty bins are floored at 1e-4.
"""
import numpy as np
import pandas as pd

EPS = 1e-4
MISSING = "__MISSING__"
OTHER = "__OTHER__"


def numeric_reference(values, n_bins=10):
    s = pd.Series(values, dtype="float64")
    x = s.dropna().to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    counts = np.histogram(x, bins=np.concatenate([[-np.inf], edges, [np.inf]]))[0]
    return {"type": "numeric", "edges": edges.tolist(),
            "props": (counts / len(s)).tolist(), "nan": float(s.isna().mean())}


def categorical_reference(values):
    s = pd.Series(values, dtype="object")
    s = s.where(s.notna(), MISSING)
    return {"type": "categorical", "props": s.value_counts(normalize=True).to_dict()}


def _psi(expected, actual):
    e = np.maximum(np.asarray(expected, dtype=float), EPS)
    a = np.maximum(np.asarray(actual, dtype=float), EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(ref, values):
    """PSI of `values` (a new month) against a reference dict. NaN if no values."""
    s = pd.Series(values)
    if len(s) == 0:
        return float("nan")
    if ref["type"] == "numeric":
        x = pd.to_numeric(s, errors="coerce")
        counts = np.histogram(x.dropna().to_numpy(),
                              bins=np.concatenate([[-np.inf], ref["edges"], [np.inf]]))[0]
        actual = list(counts / len(x)) + [float(x.isna().mean())]
        expected = list(ref["props"]) + [ref["nan"]]
        return _psi(expected, actual)
    s = s.astype("object").where(s.notna(), MISSING)
    cats = list(ref["props"].keys())
    shares = s.value_counts(normalize=True)
    actual = [float(shares.get(c, 0.0)) for c in cats] + [float(shares[~shares.index.isin(cats)].sum())]
    expected = [ref["props"][c] for c in cats] + [0.0]
    return _psi(expected, actual)


def missing_rate(values):
    return float(pd.Series(values).isna().mean()) if len(values) else float("nan")
