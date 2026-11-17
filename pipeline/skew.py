"""
pipeline/skew.py
SkewScore = upper-tail width / lower-tail width of the predicted FORWARD-RETURN distribution:

    SkewScore = (q90 - q50) / max(q50 - q10, floor)
    floor     = max(skew_floor_abs, skew_floor_rel * (q90 - q10))

Quantiles arrive already rearranged (q10 <= q50 <= q90), so both widths are >= 0 and the score
cannot flip sign. The floor stops near-zero lower tails from producing scores like 50.

Alternate (path) score: E-style ratio from the MFE and MAE medians, q50(MFE) / max(|q50(MAE)|, floor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def skew_score(q10, q50, q90, floor_abs: float = 0.002, floor_rel: float = 0.10) -> np.ndarray:
    q10, q50, q90 = (np.asarray(x, dtype=float) for x in (q10, q50, q90))
    width = np.maximum(q90 - q10, 0.0)
    floor = np.maximum(floor_abs, floor_rel * width)
    up = np.maximum(q90 - q50, 0.0)
    dn = np.maximum(q50 - q10, 0.0)
    return up / np.maximum(dn, floor)


def path_skew_score(mfe_q50, mae_q50, floor_abs: float = 0.002) -> np.ndarray:
    mfe = np.maximum(np.asarray(mfe_q50, float), 0.0)
    mae = np.abs(np.minimum(np.asarray(mae_q50, float), 0.0))
    return mfe / np.maximum(mae, floor_abs)


def bowley(q10, q50, q90) -> np.ndarray:
    """Quantile skewness in [-1, 1]: (q90 + q10 - 2 q50) / (q90 - q10)."""
    q10, q50, q90 = (np.asarray(x, dtype=float) for x in (q10, q50, q90))
    w = q90 - q10
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, (q90 + q10 - 2 * q50) / w, 0.0)


# (z90 - z10) / (z75 - z25) for a normal distribution = 2.5631 / 1.3490
NORMAL_TAIL_RATIO = 1.9001


def tail_weight(q10, q25, q75, q90, eps: float = 1e-6) -> np.ndarray:
    """Quantile kurtosis, scaled so a normal distribution scores 1.0.

        tail_weight = [(q90 - q10) / (q75 - q25)] / 1.90

    The outer (80%) range relative to the inner (50%) range. Above 1 = fatter tails than normal
    (outcomes bunch near the middle with occasional big moves); below 1 = thinner tails. Unlike
    moment kurtosis it needs no fourth moments, so it is robust to single outliers, and it pairs
    with the quantile (Bowley) skew used elsewhere.
    """
    q10, q25, q75, q90 = (np.asarray(x, dtype=float) for x in (q10, q25, q75, q90))
    inner = np.maximum(q75 - q25, eps)
    return np.maximum(q90 - q10, 0.0) / inner / NORMAL_TAIL_RATIO


def positive_skew_flag(score, p_up, confidence, threshold: float = 1.2, p_up_min: float = 0.5) -> np.ndarray:
    return ((np.asarray(score) > threshold) & (np.asarray(p_up) >= p_up_min)
            & (np.asarray(confidence) != "low"))
