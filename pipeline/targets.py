"""
pipeline/targets.py
Forward direction and path labels, strictly after the feature information set.

Timing (entry="open", the default): features on row t use bars through t-1; the trade enters at
the OPEN of day t and the h-session window is days t .. t+h-1, exiting at the close of t+h-1.
entry="close" enters at the close of t and uses days t+1 .. t+h.

Labels are NaN unless the full window exists (no partial MFE/MAE, no 0-filled directions).
Path labels use intraday highs and lows.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _fwd_max(s: pd.Series, h: int, start: int) -> pd.Series:
    """max of s over [t+start, t+start+h-1], NaN if any of it is missing."""
    return s.rolling(h, min_periods=h).max().shift(-(start + h - 1))


def _fwd_min(s: pd.Series, h: int, start: int) -> pd.Series:
    return s.rolling(h, min_periods=h).min().shift(-(start + h - 1))


def _fwd_sum(s: pd.Series, h: int, start: int) -> pd.Series:
    return s.rolling(h, min_periods=h).sum().shift(-(start + h - 1))


def _ticker_targets(g: pd.DataFrame, horizons: Sequence[int], entry: str,
                    thresholds: Sequence[float]) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    o, h_, l, c = (g[k].astype(float) for k in ("open", "high", "low", "close"))
    out = {"date": g["date"], "ticker": g["ticker"]}
    start = 0 if entry == "open" else 1
    px_in = o if entry == "open" else c
    oc = (c - o) / o.replace(0, np.nan)
    sma10 = c.rolling(10, min_periods=10).mean()

    for h in horizons:
        exit_c = c.shift(-(start + h - 1))
        ret = exit_c / px_in - 1.0
        mfe = _fwd_max(h_, h, start) / px_in - 1.0
        mae = _fwd_min(l, h, start) / px_in - 1.0
        ok = ret.notna() & mfe.notna() & mae.notna()
        out[f"ret_{h}"] = ret.where(ok)
        out[f"dir_ret_{h}"] = (ret > 0).astype(float).where(ok)
        out[f"abs_ret_{h}"] = ret.abs().where(ok)
        out[f"mfe_{h}"] = mfe.where(ok)
        out[f"mae_{h}"] = mae.where(ok)
        out[f"range_{h}"] = (mfe - mae).where(ok)
        ocm = _fwd_sum(oc, h, start)
        out[f"oc_mag_{h}"] = ocm
        out[f"dir_oc_{h}"] = (ocm > 0).astype(float).where(ocm.notna())
        for x in thresholds:
            tag = f"{x * 1000:.0f}"  # 0.005 -> '5' (0.5%), 0.01 -> '10' (1%) ...
            out[f"hit_up_{tag}_{h}"] = (mfe >= x).astype(float).where(ok)
            out[f"hit_dn_{tag}_{h}"] = (mae <= -x).astype(float).where(ok)

        # hold-if-still-strong overlay: at the exit close, if price is above its 10-day SMA and the
        # trade is green, hold h more sessions. Decided with data through the exit day only.
        exit_sma = sma10.shift(-(start + h - 1))
        strong = (exit_c > exit_sma) & (ret > 0)
        ext_c = c.shift(-(start + 2 * h - 1))
        ext_ret = ext_c / px_in - 1.0
        out[f"still_strong_{h}"] = strong.astype(float).where(ok & exit_sma.notna())
        ov = pd.Series(np.where(strong, ext_ret, ret), index=g.index)
        ov[~ok | (strong & ext_ret.isna())] = np.nan
        out[f"overlay_ret_{h}"] = ov
    return pd.DataFrame(out)


def build_targets(bars: pd.DataFrame, spy: pd.DataFrame, horizons: List[int],
                  entry: str = "open", thresholds: Sequence[float] = (0.005, 0.01, 0.02, 0.03),
                  etfs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Absolute labels per ticker, plus benchmark-relative ones.

    excess_h = return minus SPY over the same window (same entry convention).
    rel_h    = return minus the stock's sector ETF (bars['bench_etf']) over the same window;
               SPY when the ETF is unknown or has no bar that day.
    """
    pieces = [_ticker_targets(g, horizons, entry, thresholds) for _, g in bars.groupby("ticker", sort=False)]
    out = pd.concat(pieces, ignore_index=True)

    def bench_rets(b: pd.DataFrame, name: str) -> pd.DataFrame:
        t = _ticker_targets(b.assign(ticker=name), horizons, entry, thresholds)
        return t[["date"] + [f"ret_{h}" for h in horizons]]

    spy_r = bench_rets(spy, "SPY").rename(columns={f"ret_{h}": f"spy_ret_{h}" for h in horizons})
    out = out.merge(spy_r, on="date", how="left")

    if etfs is not None and len(etfs) and "bench_etf" in bars:
        e = pd.concat([bench_rets(g, t).assign(bench_etf=t) for t, g in etfs.groupby("ticker")], ignore_index=True)
        e = e.rename(columns={f"ret_{h}": f"etf_ret_{h}" for h in horizons})
        out = out.merge(bars[["date", "ticker", "bench_etf"]], on=["date", "ticker"], how="left")
        out = out.merge(e, on=["date", "bench_etf"], how="left")
    else:
        out["bench_etf"] = "SPY"
        for h in horizons:
            out[f"etf_ret_{h}"] = np.nan

    extra = {}
    for h in horizons:
        ex = out[f"ret_{h}"] - out[f"spy_ret_{h}"]
        bench = out[f"etf_ret_{h}"].fillna(out[f"spy_ret_{h}"])
        rel = out[f"ret_{h}"] - bench
        extra[f"excess_{h}"] = ex
        extra[f"dir_excess_{h}"] = (ex > 0).astype(float).where(ex.notna())
        extra[f"rel_{h}"] = rel
    drop = [f"spy_ret_{h}" for h in horizons] + [f"etf_ret_{h}" for h in horizons] + ["bench_etf"]
    out = pd.concat([out.drop(columns=drop), pd.DataFrame(extra)], axis=1)
    out = out.astype({c: np.float32 for c in out.columns if out[c].dtype == np.float64})
    return out


def add_cross_sectional_labels(panel: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """dir_xs_h = 1 if the name beat its sector by more than the median universe member did that day.
    Needs the universe-only panel, so it runs after features and targets are merged."""
    extra = {}
    for h in horizons:
        rel = panel[f"rel_{h}"]
        med = rel.groupby(panel["date"]).transform("median")
        extra[f"dir_xs_{h}"] = (rel > med).astype(np.float32).where(rel.notna())
        extra[f"rel_rank_{h}"] = rel.groupby(panel["date"]).rank(pct=True).astype(np.float32)
    return pd.concat([panel, pd.DataFrame(extra, index=panel.index)], axis=1)


def label_cols(targets: pd.DataFrame) -> List[str]:
    return [c for c in targets.columns if c not in ("date", "ticker")]
