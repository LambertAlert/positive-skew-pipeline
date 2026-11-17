"""
pipeline/features.py
Delay-1 feature matrix from daily bars. Every raw feature on row t uses bars through t-1 only
(computed on the completed session, then shifted one session per ticker).

Peer ranking: within each date, among universe members, by SIC 2-digit group; groups with fewer
than MIN_PEERS names fall back to the SIC division, then to the whole universe that day.
Model inputs are the *_pr columns (percentiles in [0, 1]) plus a few flags and calendar fields.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ANCHORS = [21, 63, 126, 252]
MIN_PEERS = 8

# raw features that are flags / calendar: kept as-is, never peer-ranked
FLAG_COLS = ["hv_elevated", "hv_transition", "gap_filled", "highest_vol_21", "highest_vol_63",
             "highest_vol_252", "highest_ever", "post_climax", "dow", "month"]
# raw features used only for conditioning / reporting (not model inputs)
META_COLS = ["hv20_pct", "days_since_climax"]


def _anchor_index(close: np.ndarray, w: int):
    """For each t, index of the more recent of the w-session closing high and low (inclusive of t)."""
    n = len(close)
    hi = np.full(n, -1, dtype=np.int64)
    side = np.full(n, np.nan)
    if n < w:
        return hi, side
    win = sliding_window_view(close, w)               # row j covers [j, j+w-1]
    ih = np.nanargmax(np.where(np.isnan(win), -np.inf, win), axis=1)
    il = np.nanargmin(np.where(np.isnan(win), np.inf, win), axis=1)
    start = np.arange(n - w + 1)
    a_hi, a_lo = start + ih, start + il
    anchor = np.maximum(a_hi, a_lo)
    hi[w - 1:] = anchor
    side[w - 1:] = np.where(a_hi >= a_lo, 1.0, -1.0)  # +1 anchored at the high, -1 at the low
    return hi, side


def _avwap(tp: np.ndarray, vol: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    pv = np.nan_to_num(tp * vol)
    v = np.nan_to_num(vol)
    cpv = np.concatenate([[0.0], np.cumsum(pv)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    t = np.arange(len(tp))
    ok = anchor >= 0
    out = np.full(len(tp), np.nan)
    a = anchor[ok]
    num = cpv[t[ok] + 1] - cpv[a]
    den = cv[t[ok] + 1] - cv[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.where(den > 0, num / den, np.nan)
    return out


def _rolling_pct_rank(x: np.ndarray, w: int, min_obs: int) -> np.ndarray:
    """Percentile of x[t] within x[t-w+1 .. t] (past only). Point-in-time replacement for .rank(pct=True)."""
    n = len(x)
    out = np.full(n, np.nan)
    pad = np.concatenate([np.full(w - 1, np.nan), x])
    win = sliding_window_view(pad, w)
    cur = x[:, None]
    valid = ~np.isnan(win)
    cnt = valid.sum(axis=1)
    le = ((win <= cur) & valid).sum(axis=1)
    ok = (cnt >= min_obs) & ~np.isnan(x)
    out[ok] = le[ok] / cnt[ok]
    return out


def _ticker_features(g: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    o, h, l, c, v = (g[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame({"date": g["date"], "ticker": g["ticker"]})
    ret1 = c.pct_change()
    prev_c = c.shift(1)
    tp = g["vwap"].astype(float) if "vwap" in g and g["vwap"].notna().any() else (h + l + c) / 3.0
    tp = tp.fillna((h + l + c) / 3.0)

    # ---- AVWAP anchored at the more recent N-session closing extreme
    cn = c.to_numpy()
    above = np.zeros(len(g))
    for w in ANCHORS:
        anc, side = _anchor_index(cn, w)
        av = pd.Series(_avwap(tp.to_numpy(), v.to_numpy(), anc))
        f[f"avwap_dist_{w}"] = (c - av) / c
        f[f"avwap_side_{w}"] = side
        f[f"avwap_slope_{w}"] = av.pct_change(5)
        above += (c > av).fillna(False).to_numpy()
    f["avwap_stack"] = above                                   # 0-4 anchors below price
    f["session_vwap_dist"] = (c - g["vwap"]) / c if "vwap" in g else np.nan

    # ---- gaps
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, min_periods=14).mean()
    gap = (o - prev_c) / prev_c
    f["gap"] = gap
    f["gap_atr"] = (o - prev_c) / atr14.shift(1)
    f["gap_filled"] = np.where(gap > 0, (l <= prev_c), np.where(gap < 0, (h >= prev_c), False)).astype(float)

    # ---- volume
    vma21 = v.rolling(21, min_periods=10).mean()
    rvol = v / vma21.shift(1)                                  # vs the prior 21 sessions
    f["rvol"] = rvol
    f["dollar_vol"] = np.log1p(c * v)
    for n in (21, 63, 252):
        f[f"highest_vol_{n}"] = (v >= v.rolling(n, min_periods=min(n, 10)).max()).astype(float)
    f["highest_ever"] = (v >= v.cummax()).astype(float)        # within the loaded history
    f["gap_x_rvol"] = gap * rvol

    climax = (rvol >= 3.0) | (f["highest_vol_63"] == 1.0)
    idx = pd.Series(np.where(climax, np.arange(len(g)), np.nan)).ffill()
    dsc = np.arange(len(g)) - idx
    f["days_since_climax"] = dsc
    f["post_climax"] = (dsc <= 5).astype(float)
    climax_close = pd.Series(np.where(climax, c, np.nan)).ffill()
    f["post_climax_drift"] = np.where(dsc <= 21, c / climax_close - 1.0, np.nan)

    # ---- open-to-close
    oc = (c - o) / o.replace(0, np.nan)
    f["oc_ret"] = oc
    f["oc_ma5"] = oc.rolling(5, min_periods=3).mean()
    f["oc_ma21"] = oc.rolling(21, min_periods=10).mean()
    f["oc_hit10"] = (oc > 0).astype(float).rolling(10, min_periods=5).mean()
    f["oc_x_rvol"] = oc * rvol
    f["clv"] = ((c - l) / (h - l).replace(0, np.nan))         # close location in range

    # ---- momentum
    for w in (21, 63, 126, 252):
        f[f"mom_{w}"] = c.pct_change(w)
    spy = spy_close.reindex(g["date"]).to_numpy()
    spy = pd.Series(spy)
    f["rel_mom_63"] = f["mom_63"] - spy.pct_change(63)
    f["rs_ibd"] = (0.4 * c / c.shift(63) + 0.2 * c / c.shift(126)
                   + 0.2 * c / c.shift(189) + 0.2 * c / c.shift(252))
    f["prox_52w"] = c / h.rolling(252, min_periods=126).max()
    rs_line = c / spy
    f["rs_line_8w"] = rs_line / rs_line.rolling(40, min_periods=20).max()
    f["dist_8w_high"] = c / c.rolling(40, min_periods=20).max() - 1.0
    f["mom_21_roc"] = f["mom_21"] - f["mom_21"].shift(5)       # 1st derivative
    f["mom_21_roc2"] = f["mom_21_roc"] - f["mom_21_roc"].shift(5)  # 2nd derivative

    # ---- mean reversion
    f["rev_5"] = -c.pct_change(5)
    for w in (5, 10, 20):
        ma = c.rolling(w, min_periods=w).mean()
        f[f"dist_ma{w}"] = (c - ma) / ma
    sd20 = c.rolling(20, min_periods=20).std()
    f["z_px20"] = (c - c.rolling(20, min_periods=20).mean()) / sd20   # Bollinger z

    # ---- volatility regime
    f["hv5"] = ret1.rolling(5, min_periods=5).std() * np.sqrt(252)
    f["hv20"] = ret1.rolling(20, min_periods=15).std() * np.sqrt(252)
    f["hv60"] = ret1.rolling(60, min_periods=40).std() * np.sqrt(252)
    f["vol_of_vol"] = f["hv20"].rolling(60, min_periods=40).std() / f["hv20"].rolling(60, min_periods=40).mean()
    lhl = np.log(h / l.replace(0, np.nan)) ** 2
    f["park20"] = np.sqrt(lhl.rolling(20, min_periods=15).mean() / (4 * np.log(2))) * np.sqrt(252)
    f["hv20_pct"] = _rolling_pct_rank(f["hv20"].to_numpy(), 252, 63)   # past-only percentile
    f["hv_elevated"] = (f["hv20_pct"] > 0.70).astype(float)
    f["hv_transition"] = (f["hv5"] > f["hv20"]).astype(float)
    f["range_pct"] = (h - l) / c
    # realized tail fatness: excess kurtosis of daily returns over 60 sessions (0 = normal)
    f["kurt60"] = ret1.rolling(60, min_periods=40).kurt()

    # ---- calendar (of the completed session; shifted below like everything else)
    f["dow"] = g["date"].dt.dayofweek.astype(float)
    f["month"] = g["date"].dt.month.astype(float)

    # ---- delay-1: row t carries what was known at the close of t-1
    cols = [k for k in f.columns if k not in ("date", "ticker")]
    f[cols] = f[cols].shift(1)
    # calendar of the day being predicted is known in advance, so restore it
    f["dow"] = g["date"].dt.dayofweek.astype(float)
    f["month"] = g["date"].dt.month.astype(float)
    return f.copy()


def raw_feature_cols(f: pd.DataFrame) -> List[str]:
    return [c for c in f.columns if c not in ("date", "ticker") + tuple(FLAG_COLS) + tuple(META_COLS)
            and not c.endswith("_pr") and not c.startswith("scr_")
            and c not in ("in_universe", "sector", "peer_group", "mcap_known", "bench_etf")]


def peer_rank(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Cross-sectional percentile per date within peer group (with fallbacks)."""
    key = df["peer_group"].astype(str)
    n_grp = df.groupby(["date", key])["ticker"].transform("size")
    n_sec = df.groupby(["date", df["sector"].astype(str)])["ticker"].transform("size")
    eff = np.where(n_grp >= MIN_PEERS, "g:" + key,
                   np.where(n_sec >= MIN_PEERS, "s:" + df["sector"].astype(str), "ALL"))
    ranks = df[cols].groupby([df["date"], pd.Series(eff, index=df.index)]).rank(pct=True)
    ranks.columns = [f"{c}_pr" for c in cols]
    return ranks


def build_features(bars: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """bars: universe-tagged stock bars (from universe.build_universe). spy: benchmark bars.
    Returns one row per universe member-day with raw, flag, meta and *_pr columns."""
    spy_close = spy.set_index("date")["close"].astype(float)
    pieces = [_ticker_features(g, spy_close) for _, g in bars.groupby("ticker", sort=False)]
    f = pd.concat(pieces, ignore_index=True)
    meta = bars[["date", "ticker", "in_universe", "sector", "peer_group"]
                + [c for c in ("scr_premarket", "scr_leader", "scr_potent", "mcap_known", "bench_etf") if c in bars]]
    f = f.merge(meta, on=["date", "ticker"], how="left")
    f = f[f["in_universe"].fillna(False).astype(bool)].reset_index(drop=True)

    raw = raw_feature_cols(f)
    f = pd.concat([f, peer_rank(f, raw)], axis=1).copy()

    # category scores from peer ranks (README: momentum category, MRV category)
    f["mom_category_pr"] = f[["mom_21_pr", "mom_63_pr", "mom_126_pr", "mom_252_pr"]].mean(axis=1)
    # stretched below the MAs and recently down = high mean-reversion score
    f["mrv_category_pr"] = ((1.0 - f["dist_ma20_pr"]) + (1.0 - f["z_px20_pr"]) + f["rev_5_pr"]) / 3.0
    f = f.astype({c: np.float32 for c in f.columns if f[c].dtype == np.float64})
    return f


def model_feature_cols(f: pd.DataFrame) -> List[str]:
    pr = [c for c in f.columns if c.endswith("_pr")]
    flags = [c for c in FLAG_COLS if c in f.columns]
    return pr + flags
