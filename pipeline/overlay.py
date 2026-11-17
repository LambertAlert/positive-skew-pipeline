"""
pipeline/overlay.py
War Room regime overlay for live scores.

The models rank names against their sector, so a top score says "best of today's universe", not
"this will make money". Whether to act, and how big, stays with the regime layer: this reads the
posture snapshot written by narrative_regime_model.py (data/narrative_regime.parquet in
LambertAlert/pivot-scanner, downloaded each run from GitHub by default) and labels
each positive-skew name accordingly. It is applied to live scores only; the posture history is too
short to backtest with.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

VIEW = {"BUY_RIP": "ACTIONABLE", "BUY_DIP": "ACTIONABLE", "NEUTRAL": "SELECTIVE", "AVOID": "BEST OF WEAK TAPE"}
# entry_mode overrides: CASH -> BEST OF WEAK TAPE · ANTICIPATION_ONLY -> WATCH ONLY · TIGHT_MA -> ACTIONABLE · TIGHT MA


CACHE = Path("data/war_room/narrative_regime.parquet")


def fetch_regime(url: str, cache: Optional[Path] = None, log=print) -> Optional[Path]:
    """Download the snapshot (atomic write). On failure fall back to the last downloaded copy."""
    cache = Path(cache) if cache else CACHE
    try:
        import requests
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        os.replace(tmp, cache)
        return cache
    except Exception as e:
        if cache.exists():
            log(f"regime snapshot download failed ({type(e).__name__}); using the copy from {cache}")
            return cache
        log(f"regime snapshot download failed ({type(e).__name__}) and no cached copy; running without the overlay")
        return None


def read_regime(path: str, max_age_days: int = 4, log=print) -> Dict:
    """Posture snapshot as a dict, plus `stale` when it is older than max_age_days."""
    if not path:
        return {}
    p = fetch_regime(path, log=log) if str(path).startswith(("http://", "https://")) else Path(path)
    if p is None or not Path(p).exists():
        return {}
    try:
        row = pd.read_parquet(p).iloc[0].to_dict()
    except Exception:
        return {}
    keep = ("generated_at", "posture", "posture_confidence", "regime_score", "current_state_name",
            "entry_mode", "entry_mode_reason", "real_rate_label", "carry_signal", "curve_regime", "fed_stance")
    out = {k: (str(row[k]) if k == "generated_at" else row[k]) for k in keep if k in row and row[k] is not None}
    try:
        age = (pd.Timestamp.now() - pd.Timestamp(out["generated_at"]).tz_localize(None)).days
        out["age_days"] = int(age)
        out["stale"] = bool(age > max_age_days)
    except Exception:
        pass
    out["source"] = str(path)
    return out


def apply_overlay(live: pd.DataFrame, regime: Dict) -> pd.DataFrame:
    live = live.copy()
    posture = regime.get("posture") if not regime.get("stale") else None
    live["regime_posture"] = posture or "UNKNOWN"
    if not posture:
        view = "STALE REGIME DATA" if regime.get("stale") else "NO REGIME DATA"
    else:
        view = VIEW.get(str(posture), "SELECTIVE")
        # the War Room entry mode (posture x acceleration x real rate x carry) can only tighten the view
        em = str(regime.get("entry_mode", "") or "").upper()
        if em == "CASH":
            view = "BEST OF WEAK TAPE"
        elif em == "ANTICIPATION_ONLY" and view != "BEST OF WEAK TAPE":
            view = "WATCH ONLY"
        elif em == "TIGHT_MA" and view == "ACTIONABLE":
            view = "ACTIONABLE · TIGHT MA"
    live["trade_view"] = np.where(live["positive_skew"], view, "")
    return live
