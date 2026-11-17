"""
pipeline/universe.py
Point-in-time stock universe from Martin Luk's three screeners (Premarket, Leader, Potent),
the same filters the War Room lab uses, with a $9 price floor.

Every screen field is computed on the completed session and then shifted one session, so
`in_universe` on day t uses only bars through t-1 — the same information set as the features.

Fields follow the lab's lk_features:
  avg_vol10  = 10-session mean volume
  pxvol      = close x volume (last session)
  pxvol_1w   = close x 5-session volume
  vol_1w     = mean(H/L - 1) over 5 sessions x 100
  chg_open   = (close / open - 1) x 100
  ema21/50   = ewm(span, min_periods=span)
  mcap       = shares outstanding x close  (see note below)

Market-cap note: Massive gives current (or as-of-date) shares, not a daily history, so mcap here is
today's share count x that day's close. With a $9 floor and $50M+ dollar volume the $100M cap rarely
binds; names with unknown shares are kept and marked mcap_known=False, as the lab does.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import LukFilters

SIC_DIVISIONS = [  # (lo, hi, name) by SIC code range
    (100, 999, "Agriculture"), (1000, 1499, "Mining"), (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"), (4000, 4999, "Transport & Utilities"), (5000, 5199, "Wholesale"),
    (5200, 5999, "Retail"), (6000, 6799, "Finance"), (7000, 8999, "Services"), (9100, 9999, "Public"),
]


def sic_division(code) -> str:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    for lo, hi, name in SIC_DIVISIONS:
        if lo <= c <= hi:
            return name
    return "Unknown"


def sic_group(code) -> str:
    """2-digit SIC major group, e.g. '28' chemicals & pharma, '36' electronics, '73' software/services."""
    try:
        return f"{int(code) // 100:02d}"
    except (TypeError, ValueError):
        return "Unknown"


SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC"]

# (lo, hi, etf) checked in order, first match wins; finer ranges come before the broad ones
_SIC_ETF = [
    (2830, 2836, "XLV"), (3840, 3851, "XLV"), (8000, 8099, "XLV"), (5120, 5122, "XLV"),
    (3570, 3579, "XLK"), (3660, 3679, "XLK"), (3820, 3829, "XLK"), (7370, 7379, "XLK"),
    (4800, 4899, "XLC"), (2700, 2799, "XLC"), (7800, 7849, "XLC"),
    (4900, 4991, "XLU"),
    (1300, 1399, "XLE"), (2900, 2999, "XLE"), (4600, 4699, "XLE"), (1200, 1299, "XLE"),
    (6500, 6553, "XLRE"), (6798, 6798, "XLRE"),
    (6000, 6799, "XLF"),
    (2000, 2199, "XLP"), (2840, 2844, "XLP"), (5400, 5499, "XLP"),
    (3710, 3716, "XLY"), (2300, 2399, "XLY"), (2500, 2599, "XLY"), (3940, 3949, "XLY"),
    (5200, 5999, "XLY"), (7000, 7099, "XLY"), (7900, 7999, "XLY"), (1520, 1531, "XLY"),
    (1000, 1099, "XLB"), (1400, 1499, "XLB"), (2400, 2499, "XLB"), (2600, 2699, "XLB"),
    (2800, 2899, "XLB"), (3200, 3399, "XLB"),
    (1500, 1799, "XLI"), (3400, 3599, "XLI"), (3700, 3799, "XLI"), (3800, 3899, "XLI"),
    (4000, 4799, "XLI"), (5000, 5199, "XLI"), (7300, 7399, "XLI"), (8700, 8799, "XLI"),
]


def sic_to_etf(code, fallback: str = "SPY") -> str:
    """SPDR sector ETF for a SIC code (SPY when unknown or unmapped)."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return fallback
    for lo, hi, etf in _SIC_ETF:
        if lo <= c <= hi:
            return etf
    return fallback


def luk_fields(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker screen fields on each completed session (NOT shifted)."""
    df = bars.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)
    df["avg_vol10"] = g["volume"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    df["pxvol"] = df["close"] * df["volume"]
    df["pxvol_1w"] = df["close"] * g["volume"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    rng = df["high"] / df["low"].replace(0, np.nan) - 1.0
    df["vol_1w"] = rng.groupby(df["ticker"]).transform(lambda s: s.rolling(5, min_periods=5).mean()) * 100
    df["chg_open"] = (df["close"] / df["open"].replace(0, np.nan) - 1.0) * 100
    df["ema21"] = g["close"].transform(lambda s: s.ewm(span=21, min_periods=21).mean())
    df["ema50"] = g["close"].transform(lambda s: s.ewm(span=50, min_periods=50).mean())
    return df


def screen_flags(df: pd.DataFrame, f: LukFilters, shares: Optional[pd.Series] = None) -> pd.DataFrame:
    """Boolean screens on completed-session fields; caller shifts them."""
    out = pd.DataFrame(index=df.index)
    if shares is not None:
        sh = df["ticker"].map(shares)
        mcap = sh * df["close"]
    else:
        mcap = pd.Series(np.nan, index=df.index)
    known = mcap.notna()
    cap_ok = (~known) | (mcap > f.min_mcap)
    price_ok = df["close"] > f.min_price

    out["scr_premarket"] = (price_ok & cap_ok & (df["avg_vol10"] > f.min_avg_vol10)
                            & (df["pxvol"] > f.min_pxvol) & (df["vol_1w"] > f.min_vol_1w))
    out["scr_leader"] = (price_ok & cap_ok & (df["avg_vol10"] > f.min_avg_vol10)
                         & (df["pxvol_1w"] > f.min_pxvol_1w))
    out["scr_potent"] = (price_ok & cap_ok & (df["close"] > df["ema50"]) & (df["close"] >= df["ema21"])
                         & (df["vol_1w"] > f.min_vol_1w) & (df["pxvol"] > f.min_pxvol)
                         & (df["chg_open"].abs() > f.potent_open_band))
    out["mcap"] = mcap
    out["mcap_known"] = known
    return out


def build_universe(bars: pd.DataFrame, f: LukFilters, shares: Optional[pd.Series] = None,
                   sic: Optional[pd.Series] = None) -> pd.DataFrame:
    """Returns bars + point-in-time screen flags (as of t-1), in_universe, sector, peer_group."""
    df = luk_fields(bars)
    flags = screen_flags(df, f, shares)
    for c in flags.columns:
        df[c] = flags[c]
    scr_cols = ["scr_premarket", "scr_leader", "scr_potent", "mcap", "mcap_known"]
    # delay-1: today's membership is yesterday's screen result
    df[scr_cols] = df.groupby("ticker", sort=False)[scr_cols].shift(1)
    for c in ["scr_premarket", "scr_leader", "scr_potent"]:
        df[c] = df[c].fillna(False).astype(bool)
    want = [f"scr_{s}" for s in f.screens]
    df["in_universe"] = df[want].any(axis=1)

    if sic is not None:
        df["sic_code"] = df["ticker"].map(sic)
        df["sector"] = df["sic_code"].map(sic_division)
        df["peer_group"] = df["sic_code"].map(sic_group)
        df["bench_etf"] = df["sic_code"].map(sic_to_etf)
    else:
        df["sector"] = "All"
        df["peer_group"] = "All"
        df["bench_etf"] = "SPY"
    drop = ["avg_vol10", "pxvol", "pxvol_1w", "vol_1w", "chg_open", "ema21", "ema50"]
    return df.drop(columns=drop)


def ever_passing_without_cap(bars: pd.DataFrame, f: LukFilters) -> np.ndarray:
    """Tickers that pass some screen at least once ignoring the cap test — the only ones
    whose share counts are worth fetching."""
    df = luk_fields(bars)
    fl = screen_flags(df, f, shares=None)
    hit = fl[[f"scr_{s}" for s in f.screens]].any(axis=1)
    return df.loc[hit, "ticker"].unique()


def append_next_session(bars: pd.DataFrame) -> pd.DataFrame:
    """Add one empty row per ticker that traded on the last date, dated the next weekday.
    After the delay-1 shift that row carries everything known at the last close, so it is the row
    the live model scores. (A market holiday just means the 'next session' label is a day early.)"""
    last = bars["date"].max()
    nxt = last + pd.tseries.offsets.BDay(1)
    live = bars.loc[bars["date"] == last, ["ticker"]].copy()
    live["date"] = nxt
    for c in ("sector", "peer_group", "sic_code", "bench_etf"):
        if c in bars:
            live[c] = live["ticker"].map(bars.drop_duplicates("ticker", keep="last").set_index("ticker")[c])
    return pd.concat([bars, live], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_market(cfg, client, next_session: bool = True, log=print) -> Dict[str, pd.DataFrame]:
    """Pull everything from Massive and return {'bars': universe-tagged stock bars, 'spy': benchmark bars,
    'etfs': SPDR sector ETF bars, 'details': reference rows}."""
    end = cfg.end or (pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    if cfg.start in ("", "auto"):
        # first scored day = the plan's earliest date + the warm-up the 252-day features need
        lo = (pd.Timestamp(end) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
        earliest = pd.Timestamp(client.earliest_available(lo, end))
        cfg.start = (earliest + pd.tseries.offsets.BDay(cfg.warmup_sessions + 5)).strftime("%Y-%m-%d")
        log(f"Massive plan history starts {earliest.date()}; scoring from {cfg.start} "
            f"(the first {cfg.warmup_sessions} sessions warm up the features)")
    start = (pd.Timestamp(cfg.start) - pd.tseries.offsets.BDay(cfg.warmup_sessions + 30)).strftime("%Y-%m-%d")
    stocks = client.stock_set()
    benches = {cfg.benchmark} | set(SECTOR_ETFS)
    names = set(stocks["ticker"]) | benches
    raw = client.grouped_range(start, end, tickers=names)
    spy = raw[raw["ticker"] == cfg.benchmark].reset_index(drop=True)
    etfs = raw[raw["ticker"].isin(SECTOR_ETFS)].reset_index(drop=True)
    bars = raw[~raw["ticker"].isin(benches) & (raw["close"] >= 1.0)].reset_index(drop=True)

    cand = ever_passing_without_cap(bars, cfg.luk)
    bars = bars[bars["ticker"].isin(cand)].reset_index(drop=True)

    # short plan: Massive can't cover the feature warm-up -> backfill older bars from yfinance
    first_massive = raw["date"].min()
    short = first_massive > pd.Timestamp(start) + pd.tseries.offsets.BDay(20)
    backfilled = False
    if cfg.backfill == "yfinance" or (cfg.backfill == "auto" and short):
        from .backfill import backfill
        bars, spy, etfs = backfill(bars, spy, etfs, cand, start, first_massive, log=log)
        backfilled = True
    last_seen = bars.groupby("ticker")["date"].max().dt.strftime("%Y-%m-%d").to_dict()
    delisted = set(stocks.loc[stocks.get("active", True) == False, "ticker"]) if "active" in stocks else set()
    asof = {t: last_seen[t] for t in cand if t in delisted}
    det = client.ticker_details(cand, asof=asof)
    shares = det.set_index("ticker")["shares"].astype(float) if len(det) else None
    sic = det.set_index("ticker")["sic_code"] if len(det) else None
    if next_session:
        bars = append_next_session(bars)
    tagged = build_universe(bars, cfg.luk, shares=shares, sic=sic)
    return {"bars": tagged, "spy": spy, "etfs": etfs, "details": det,
            "first_massive": first_massive, "backfilled": backfilled}
