"""
pipeline/backfill.py
yfinance backfill for the dates before a short Massive plan's history.

Why: the features need ~260 sessions of history (252-day momentum, 52-week high, 252-day AVWAP,
HV20 percentile). On a 2-year plan that would eat half the data before the first scored day.

How:
- Only the tickers Massive already put in the candidate set, plus SPY and the sector ETFs.
- Batch downloads only (War Room rule), auto_adjust=True, tz_localize(None).
- Splice: yfinance is dividend-adjusted and Massive is split-only, so each ticker's yfinance prices
  are scaled by the median Massive/yfinance close over the first overlapping sessions. Without this
  there would be a fake jump at the join date. Volume is left as is (both are split-adjusted).
- Rows are tagged source="yfinance". Tickers with no overlap (or no yfinance data) are skipped.

Survivorship: the backfill only contains names alive when the Massive window starts, so the default
backfill_use="warmup" uses it for feature history only; nothing is trained or scored on it.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

OVERLAP_SESSIONS = 10


def to_yf(t: str) -> str:
    return t.replace(".", "-")          # BRK.B -> BRK-B


def yf_download(tickers: List[str], start: str, end: str, chunk: int = 200) -> pd.DataFrame:
    """Long OHLCV frame from yfinance, batched. Separate function so tests can replace it."""
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        raw = yf.download([to_yf(t) for t in batch], start=start, end=end, auto_adjust=True,
                          group_by="ticker", threads=True, progress=False)
        if raw is None or raw.empty:
            continue
        for t in batch:
            key = to_yf(t)
            if isinstance(raw.columns, pd.MultiIndex):
                if key not in raw.columns.get_level_values(0):
                    continue
                d = raw[key]
            else:
                d = raw
            d = d.dropna(subset=["Close"])
            if d.empty:
                continue
            idx = pd.DatetimeIndex(d.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            frames.append(pd.DataFrame({"date": idx.normalize(), "ticker": t, "open": d["Open"].values,
                                        "high": d["High"].values, "low": d["Low"].values,
                                        "close": d["Close"].values, "volume": d["Volume"].values}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "volume"])


def splice(massive: pd.DataFrame, yfb: pd.DataFrame, first_date: pd.Timestamp) -> pd.DataFrame:
    """Prepend yfinance rows dated before first_date, scaled to Massive at the overlap."""
    if yfb.empty:
        return massive
    out = []
    m_first = massive[massive["date"] >= first_date].sort_values("date")
    for t, y in yfb.groupby("ticker"):
        m = m_first[m_first["ticker"] == t].head(OVERLAP_SESSIONS)
        ov = m.merge(y[["date", "close"]], on="date", suffixes=("", "_y"))
        if len(ov) < 3:
            continue
        ratio = float(np.median(ov["close"] / ov["close_y"]))
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        old = y[y["date"] < first_date].copy()
        for c in ("open", "high", "low", "close"):
            old[c] = old[c] * ratio
        old["vwap"] = np.nan
        old["trades"] = np.nan
        old["source"] = "yfinance"
        out.append(old)
    if not out:
        return massive
    m = massive.copy()
    if "source" not in m:
        m["source"] = "massive"
    return (pd.concat(out + [m], ignore_index=True)
              .sort_values(["ticker", "date"]).reset_index(drop=True))


def backfill(bars: pd.DataFrame, spy: pd.DataFrame, etfs: pd.DataFrame, tickers: Iterable[str],
             need_from: str, first_date: pd.Timestamp, log=print, downloader=None):
    """Returns (bars, spy, etfs) with yfinance history prepended before first_date."""
    downloader = downloader or yf_download
    benches = sorted(set(spy["ticker"]) | set(etfs["ticker"]))
    names = sorted(set(tickers)) + benches
    end = (first_date + pd.tseries.offsets.BDay(OVERLAP_SESSIONS + 5)).strftime("%Y-%m-%d")
    log(f"yfinance backfill: {len(names):,} tickers, {need_from} → {first_date.date()} (batched)")
    yfb = downloader(names, need_from, end)
    got = yfb["ticker"].nunique() if len(yfb) else 0
    log(f"yfinance backfill: data for {got:,} of {len(names):,}")
    bars = splice(bars, yfb[yfb["ticker"].isin(set(tickers))], first_date)
    spy = splice(spy, yfb[yfb["ticker"].isin(set(spy["ticker"]))], first_date)
    etfs = splice(etfs, yfb[yfb["ticker"].isin(set(etfs["ticker"]))], first_date)
    return bars, spy, etfs
