"""
pipeline/overlay.py
War Room regime overlay for live scores.

The models rank names against their sector, so a top score says "best of today's universe", not
"this will make money". Whether to act, and how big, stays with the regime layer: this reads the
posture snapshot written by narrative_regime_model.py (data/narrative_regime.parquet) and labels
each positive-skew name accordingly. It is applied to live scores only; the posture history is too
short to backtest with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

VIEW = {"BUY_RIP": "ACTIONABLE", "BUY_DIP": "ACTIONABLE", "NEUTRAL": "SELECTIVE", "AVOID": "BEST OF WEAK TAPE"}


def read_regime(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        row = pd.read_parquet(p).iloc[0].to_dict()
    except Exception:
        return {}
    keep = ("generated_at", "posture", "posture_confidence", "regime_score", "current_state_name",
            "entry_mode", "real_rate_label", "carry_signal")
    return {k: (str(row[k]) if k in ("generated_at",) else row[k]) for k in keep if k in row and row[k] is not None}


def apply_overlay(live: pd.DataFrame, regime: Dict) -> pd.DataFrame:
    live = live.copy()
    posture = regime.get("posture")
    live["regime_posture"] = posture or "UNKNOWN"
    if not posture:
        view = "NO REGIME DATA"
    else:
        view = VIEW.get(str(posture), "SELECTIVE")
        if str(regime.get("entry_mode", "")).upper() == "CASH":
            view = "BEST OF WEAK TAPE"
    live["trade_view"] = np.where(live["positive_skew"], view, "")
    return live
