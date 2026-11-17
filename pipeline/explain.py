"""
pipeline/explain.py
SHAP artifacts.

- TreeSHAP on the boosted and random-forest direction models (class-1 output, any shap version).
- Exact linear SHAP from the logistic model (LinearDir.shap).
- SkewSHAP on the quantile models. Upper-tail width is q90 - q50 and lower-tail width is q50 - q10,
  and SHAP is additive, so the attribution of (upper - lower) = q90 + q10 - 2 q50 is
      SkewSHAP = SHAP(q90) + SHAP(q10) - 2 SHAP(q50).
  (The README's SHAP(upper) - SHAP(lower) read as SHAP(q90) - SHAP(q10) would attribute the spread,
  q90 - q10, not the asymmetry.)
- Interaction values on the boosted model for the top pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def _class1(sv) -> np.ndarray:
    """Normalize shap outputs: list [neg, pos], (n, f, 2) array, or (n, f)."""
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def tree_shap(estimator, X: pd.DataFrame) -> Optional[pd.DataFrame]:
    if shap is None:
        return None
    ex = shap.TreeExplainer(estimator)
    sv = _class1(ex.shap_values(X))
    return pd.DataFrame(sv, index=X.index, columns=X.columns)


def expected_value(estimator) -> float:
    """The model's average raw output (SHAP base value): class-1 log-odds for classifiers."""
    ev = np.atleast_1d(shap.TreeExplainer(estimator).expected_value)
    return float(ev[-1])


def skew_base(qset) -> float:
    """Base value of the asymmetry q90 + q10 - 2 q50 (the start of a SkewSHAP waterfall)."""
    return (expected_value(qset.models[0.9]) + expected_value(qset.models[0.1])
            - 2 * expected_value(qset.models[0.5]))


def beeswarm_sample(X: pd.DataFrame, sv: pd.DataFrame, features: List[str], n: int = 250,
                    seed: int = 42) -> pd.DataFrame:
    """Long table for a SHAP beeswarm: per feature, SHAP value and the feature value scaled 0-1
    within the sample (so ranks, flags and interaction terms share one colour scale)."""
    idx = X.sample(min(n, len(X)), random_state=seed).index
    rows = []
    for f in features:
        x = X.loc[idx, f].astype(float)
        lo, hi = x.min(), x.max()
        xs = (x - lo) / (hi - lo) if hi > lo else x * 0 + 0.5
        rows.append(pd.DataFrame({"feature": f, "x": xs.values, "shap": sv.loc[idx, f].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def skew_shap(qset, X: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
    """qset: QuantileSet trained on forward returns with 0.1/0.5/0.9."""
    if shap is None:
        return None
    s = {q: tree_shap(qset.models[q], X[qset.cols]) for q in (0.1, 0.5, 0.9)}
    return {"upper": s[0.9] - s[0.5], "lower": s[0.5] - s[0.1],
            "skew": s[0.9] + s[0.1] - 2 * s[0.5]}


def interaction_strength(estimator, X: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    if shap is None:
        return pd.DataFrame()
    iv = shap.TreeExplainer(estimator).shap_interaction_values(X)
    if isinstance(iv, list):
        iv = iv[-1]
    iv = np.asarray(iv)
    if iv.ndim == 4:          # (n, f, f, classes)
        iv = iv[..., -1]
    m = np.abs(iv).mean(axis=0)
    cols = list(X.columns)
    rows = [{"a": cols[i], "b": cols[j], "mean_abs": float(m[i, j] * 2)}
            for i in range(len(cols)) for j in range(i + 1, len(cols))]
    return pd.DataFrame(rows).sort_values("mean_abs", ascending=False).head(top)


def global_importance(sv: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"mean_abs": sv.abs().mean(), "mean_signed": sv.mean()}).sort_values("mean_abs", ascending=False)


def top_contributors(sv: pd.DataFrame, n: int = 5) -> List[List[tuple]]:
    vals = sv.to_numpy()
    cols = np.array(sv.columns)
    idx = np.argsort(-np.abs(vals), axis=1)[:, :n]
    return [[(cols[j], float(vals[i, j])) for j in row] for i, row in enumerate(idx)]


def dependence_sample(X: pd.DataFrame, sv: pd.DataFrame, features: List[str], color: pd.DataFrame,
                      n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Long table for dependence plots: feature value vs SHAP, with regime / extreme-volume colour keys."""
    idx = X.sample(min(n, len(X)), random_state=seed).index
    rows = []
    for f in features:
        if f not in sv:
            continue
        d = pd.DataFrame({"feature": f, "x": X.loc[idx, f].values, "shap": sv.loc[idx, f].values})
        for c in color.columns:
            d[c] = color.loc[idx, c].values
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
