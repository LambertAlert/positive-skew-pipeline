"""
pipeline/config.py
Every tunable in one place. Override by editing here or passing a dict to Config(**overrides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LukFilters:
    """Martin Luk's three morning screeners, applied point-in-time on daily bars.

    Same thresholds as the lab's LK_DEFAULTS (claude/luk-scanner-handoff.md), except the
    price floor, which is $9 here (the lab's whole-market panel used $1).
    All fields are measured on completed sessions through t-1, so membership on day t is
    known before day t opens (delay-1).
    """
    min_price: float = 9.0              # close > $9 (strict)
    min_mcap: float = 100e6             # market cap > $100M
    min_avg_vol10: float = 500_000      # 10-session average volume > 500K shares
    min_pxvol: float = 50e6             # last close x last volume > $50M (Premarket, Potent)
    min_pxvol_1w: float = 300e6         # last close x 5-session volume > $300M (Leader)
    min_vol_1w: float = 3.5             # mean(H/L - 1) over 5 sessions x 100 > 3.5 (Premarket, Potent)
    potent_open_band: float = 0.5       # Potent: |close/open - 1| x 100 outside +/-0.5
    # universe = union of the screeners listed here
    screens: List[str] = field(default_factory=lambda: ["premarket", "leader", "potent"])


@dataclass
class Config:
    # data
    api_key_env: List[str] = field(default_factory=lambda: ["MASSIVE_API_KEY"])
    base_url: str = "https://api.massive.com"
    cache_dir: str = "data/psp_cache"
    out_dir: str = "data/psp_out"
    start: str = "auto"                 # "auto" = plan's earliest Massive date + the feature warm-up
    end: str = ""                       # "" = last completed session
    benchmark: str = "SPY"
    stock_types: List[str] = field(default_factory=lambda: ["CS", "ADRC"])
    include_delisted: bool = True       # reduces survivorship bias in the reference set
    warmup_sessions: int = 260          # history needed before the first scored day
    # Plans with short history (e.g. 2 years): bars before Massive's earliest date can be backfilled
    # from yfinance. "auto" backfills only when Massive can't cover the warm-up.
    backfill: str = "none"              # none (Massive only) | auto | yfinance
    backfill_use: str = "warmup"        # warmup: backfill only feeds features (no training/testing on it)
                                        # full: also train on it (survivors only before the Massive date)
    calls_per_minute: int = 0           # 0 = no limit; set 5 for the free tier
    probe_history: bool = True          # find the plan's earliest grouped-bars date once and cache it

    luk: LukFilters = field(default_factory=LukFilters)

    # targets
    # "relative": models learn the return over the sector ETF (rel_h) and whether a name beat the day's
    #             median universe member on that basis (dir_xs_h). "absolute": raw return and its sign.
    target_mode: str = "relative"
    # War Room posture snapshot (written by narrative_regime_model.py in LambertAlert/pivot-scanner),
    # used as a live overlay. A URL is downloaded each run (cached in data/war_room/); a path is read as is.
    regime_path: str = ("https://raw.githubusercontent.com/LambertAlert/pivot-scanner/main/"
                        "data/narrative_regime.parquet")
    regime_max_age_days: int = 4        # older snapshots are shown as stale
    horizons: List[int] = field(default_factory=lambda: [3, 5, 7, 10, 21])
    entry: str = "open"                 # "open" of day t (signal from t-1 close) or "close" of day t
    hit_thresholds: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.03])
    climax_rvol: float = 3.0            # extreme-volume day: RVOL >= 3 or highest volume in 63 sessions
    post_climax_days: int = 5

    # IC screen
    ic_min_abs: float = 0.015           # |mean cross-sectional IC|
    ic_min_abs_ir: float = 0.15         # |mean / std| of the daily IC series
    ic_min_sign_stability: float = 0.55 # share of calendar quarters with the same sign as the mean
    ic_min_names_per_day: int = 20
    max_survivors: int = 30
    max_interactions: int = 8

    # harness
    purge_min: int = 21                 # purge = max(purge_min, h + 1) sessions between the last train row and test
    train_min_sessions: int = 252
    test_sessions: int = 63
    window: str = "expanding"           # or "rolling"
    rolling_sessions: int = 756
    calib_frac: float = 0.2             # tail of each training fold held out (purged) for isotonic calibration
    max_train_rows: int = 400_000       # sampled by row inside each fold, for speed
    seed: int = 42

    # skew
    skew_threshold: float = 1.2
    skew_floor_abs: float = 0.002       # denominator floor (return units)
    skew_floor_rel: float = 0.10        # ... or 10% of the q10-q90 width, whichever is larger
    p_up_min: float = 0.5
    iso_contamination: float = 0.05

    def apply_profile(self, n_dates: int, profile: str = "auto") -> str:
        """Shorter walk-forward settings when there are few dates to work with."""
        if profile == "standard" or (profile == "auto" and n_dates >= 700):
            return "standard"
        self.train_min_sessions = min(self.train_min_sessions, max(126, int(n_dates * 0.4)))
        self.test_sessions = min(self.test_sessions, 42)
        self.rolling_sessions = min(self.rolling_sessions, n_dates)
        return "short"

    def dir_label(self, h: int) -> str:
        return f"dir_xs_{h}" if self.target_mode == "relative" else f"dir_ret_{h}"

    def mag_label(self, h: int) -> str:
        return f"rel_{h}" if self.target_mode == "relative" else f"ret_{h}"

    def purge(self, h: int) -> int:
        return max(self.purge_min, h + 1)
