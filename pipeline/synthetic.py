"""
pipeline/synthetic.py
Offline stand-in for Massive: bars shaped like the grouped-daily endpoint (open/high/low/close/
volume/vwap), SPY, share counts, SIC codes, and SPDR sector ETFs. Stock returns share a market factor
and a sector factor, so sector-relative labels have something real to strip out.

signal=0.0 is a pure random walk: an honest pipeline must find ~no OOS skill on it (the leakage
test). signal>0 plants a small, known effect — next-day drift rises with the 21-session momentum
and upside jumps follow extreme-volume days — so the tests can check the pipeline recovers it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SICS = [2834, 3674, 7372, 6022, 1311, 5812, 3711, 4911, 2836, 7370, 3571, 6798]
SIC_ETF = {2834: "XLV", 2836: "XLV", 3674: "XLK", 7372: "XLK", 7370: "XLK", 3571: "XLK", 6022: "XLF",
           1311: "XLE", 5812: "XLY", 3711: "XLY", 4911: "XLU", 6798: "XLRE"}


def make_market(n_tickers: int = 300, n_days: int = 900, signal: float = 0.0, seed: int = 7,
                start: str = "2022-01-03"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    mkt = rng.normal(0.0003, 0.010, n_days)
    spy_c = 400 * np.cumprod(1 + mkt)
    spy = pd.DataFrame({"date": dates, "ticker": "SPY", "open": spy_c / (1 + rng.normal(0, .002, n_days)),
                        "close": spy_c, "volume": 8e7})
    spy["high"] = spy[["open", "close"]].max(axis=1) * 1.004
    spy["low"] = spy[["open", "close"]].min(axis=1) * 0.996
    spy["vwap"] = (spy["high"] + spy["low"] + spy["close"]) / 3

    etf_names = sorted(set(SIC_ETF.values()))
    sector_f = {e: rng.normal(0.0, 0.008, n_days) for e in etf_names}   # sector factors the stocks share
    etf_frames = []
    for e in etf_names:
        ec = 100 * np.cumprod(1 + mkt + sector_f[e])
        eo = ec / (1 + rng.normal(0, .002, n_days))
        etf_frames.append(pd.DataFrame({"date": dates, "ticker": e, "open": eo, "close": ec, "volume": 1e7,
                                        "high": np.maximum(eo, ec) * 1.004, "low": np.minimum(eo, ec) * 0.996}))
    etfs = pd.concat(etf_frames, ignore_index=True)
    etfs["vwap"] = (etfs["high"] + etfs["low"] + etfs["close"]) / 3

    frames, shares, sic = [], {}, {}
    for i in range(n_tickers):
        t = f"S{i:03d}"
        sec = sector_f[SIC_ETF[SICS[i % len(SICS)]]]
        beta = rng.uniform(0.6, 1.6)
        vol = rng.uniform(0.018, 0.045)
        p0 = rng.uniform(8, 150)
        base_v = rng.lognormal(np.log(2.0e6), 0.8)
        r = np.empty(n_days)
        c = np.empty(n_days)
        v = base_v * rng.lognormal(0, 0.35, n_days)
        spike = rng.random(n_days) < 0.02
        v[spike] *= rng.uniform(3, 8, spike.sum())
        price = p0
        for d in range(n_days):
            drift = 0.0
            if signal and d > 21:
                mom = c[d - 1] / c[d - 21] - 1
                drift += signal * np.tanh(mom * 5) * 0.004
                if spike[d - 1]:
                    drift += signal * 0.01 * rng.exponential(1.0)  # right-skewed follow-through
            r[d] = beta * mkt[d] + sec[d] + rng.normal(drift, vol)
            price *= (1 + r[d])
            c[d] = price
        prev = np.concatenate([[p0], c[:-1]])
        o = prev * (1 + rng.normal(0, vol / 3, n_days))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n_days)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n_days)))
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "open": o, "high": hi, "low": lo, "close": c,
                                    "volume": v, "vwap": (hi + lo + c) / 3, "trades": 1000}))
        shares[t] = rng.uniform(20e6, 400e6)
        sic[t] = SICS[i % len(SICS)]
    bars = pd.concat(frames, ignore_index=True)
    return bars, spy, pd.Series(shares), pd.Series(sic), etfs
