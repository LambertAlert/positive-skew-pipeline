"""
pipeline/evaluate.py
Out-of-sample metrics from the walk-forward predictions (one row per name-day-horizon).

y_dir / y_mag are the labels the models were trained on: in the default relative mode, whether the
name beat the day's median universe member vs its sector ETF, and the return over that ETF.
Absolute return (y_ret) and excess vs SPY (y_excess) are reported alongside for context; the
hold-if-still-strong overlay is absolute, since it is a trade-management rule.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROB_COLS = ["p_up", "p_linear", "p_rf", "p_gbm", "p_bin_iso"]


def _safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def xs_auc(d: pd.DataFrame, col: str) -> float:
    """Mean per-date AUC: ranks names against each other on the same day, so market-wide up or down
    days (which no stock-level feature can call) don't swamp the score."""
    a = d.groupby("date").apply(lambda g: _safe_auc(g["y_dir"].to_numpy(int), g[col].to_numpy(float))
                                if len(g) >= 20 else np.nan)
    return float(a.mean())


def direction_metrics(o: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, dh in o.groupby("horizon"):
        for reg, d in [("all", dh)] + [(r, dh[dh["hv_regime"] == r]) for r in ("low", "mid", "high")]:
            d = d[d["y_dir"].notna()]
            if len(d) < 100:
                continue
            y = d["y_dir"].to_numpy(int)
            base = np.full(len(y), y.mean())
            for c in PROB_COLS:
                pp = np.clip(d[c].to_numpy(float), 1e-4, 1 - 1e-4)
                rows.append({"horizon": h, "regime": reg, "model": c.replace("p_", "") if c != "p_up" else "ensemble",
                             "n": len(d), "brier": brier_score_loss(y, pp), "brier_base": brier_score_loss(y, base),
                             "logloss": log_loss(y, pp, labels=[0, 1]), "auc": _safe_auc(y, pp),
                             "auc_xs": xs_auc(d, c)})
    t = pd.DataFrame(rows)
    if len(t):
        t["brier_skill"] = 1 - t["brier"] / t["brier_base"]
    return t


def calibration_table(o: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = o[o["y_dir"].notna()].copy()
    d["bin"] = pd.cut(d["p_up"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (d.groupby(["horizon", "bin"], observed=True)
              .agg(n=("p_up", "size"), mean_p=("p_up", "mean"), hit=("y_dir", "mean")).reset_index()
              .assign(bin=lambda x: x["bin"].astype(str)))


def decile_lift(o: pd.DataFrame, score: str = "p_up") -> pd.DataFrame:
    """Cross-sectional deciles per date (10 = highest score)."""
    d = o[o["y_mag"].notna()].copy()
    d["dec"] = d.groupby(["horizon", "date"])[score].rank(pct=True).mul(10).clip(upper=9.999).astype(int) + 1
    return (d.groupby(["horizon", "dec"])
              .agg(n=("y_mag", "size"), mean_mag=("y_mag", "mean"), mean_ret=("y_ret", "mean"),
                   mean_excess=("y_excess", "mean"), hit=("y_dir", "mean"),
                   mfe=("y_mfe", "mean"), mae=("y_mae", "mean")).reset_index())


def pinball(y, q, tau):
    e = y - q
    return np.mean(np.maximum(tau * e, (tau - 1) * e))


def quantile_metrics(o: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, d in o.groupby("horizon"):
        d = d[d["y_mag"].notna()]
        if len(d) < 100:
            continue
        y = d["y_mag"].to_numpy(float)
        for tau, c in [(0.1, "mag_q10"), (0.25, "mag_q25"), (0.5, "mag_q50"), (0.75, "mag_q75"), (0.9, "mag_q90")]:
            if c not in d:
                continue
            q = d[c].to_numpy(float)
            # baseline: the same quantile of realized returns over the previous dates (expanding, no look-ahead)
            base = d.groupby("date")["y_mag"].quantile(tau).expanding().mean().shift(1).reindex(d["date"]).to_numpy()
            ok = ~np.isnan(base)
            rows.append({"horizon": h, "target": "mag", "tau": tau, "pinball": pinball(y, q, tau),
                         "pinball_base": pinball(y[ok], base[ok], tau) if ok.any() else np.nan,
                         "coverage_below": float(np.mean(y < q))})
        for tau, c, ycol in [(0.5, "mfe_q50", "y_mfe"), (0.9, "mfe_q90", "y_mfe"),
                             (0.1, "mae_q10", "y_mae"), (0.5, "mae_q50", "y_mae")]:
            yy, q = d[ycol].to_numpy(float), d[c].to_numpy(float)
            rows.append({"horizon": h, "target": ycol[2:], "tau": tau, "pinball": pinball(yy, q, tau),
                         "pinball_base": np.nan, "coverage_below": float(np.mean(yy < q))})
        inside = np.mean((y >= d["mag_q10"].to_numpy()) & (y <= d["mag_q90"].to_numpy()))
        rows.append({"horizon": h, "target": "mag", "tau": "q10-q90", "pinball": np.nan,
                     "pinball_base": np.nan, "coverage_below": float(inside)})
    t = pd.DataFrame(rows)
    if len(t):
        t["pinball_skill"] = 1 - t["pinball"] / t["pinball_base"]
    return t


def realized_tail_weight(r: pd.Series) -> float:
    """Quantile kurtosis of realized outcomes, normal = 1.0 (same scale as the prediction)."""
    q10, q25, q75, q90 = r.quantile([0.1, 0.25, 0.75, 0.9])
    return float((q90 - q10) / (q75 - q25) / 1.9001) if q75 > q25 else np.nan


def realized_skew(r: pd.Series) -> float:
    q10, q50, q90 = r.quantile([0.1, 0.5, 0.9])
    return float((q90 + q10 - 2 * q50) / (q90 - q10)) if q90 > q10 else np.nan


def skew_buckets(o: pd.DataFrame) -> pd.DataFrame:
    d = o[o["y_mag"].notna()].copy()
    d["bucket"] = pd.cut(d["skew_score"], [-np.inf, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf],
                         labels=["<0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", ">2.0"])
    g = d.groupby(["horizon", "bucket"], observed=True)
    t = g.agg(n=("y_mag", "size"), mean_mag=("y_mag", "mean"), median_mag=("y_mag", "median"),
              mean_ret=("y_ret", "mean"),
              mean_mfe=("y_mfe", "mean"), mean_mae=("y_mae", "mean"), hit_up_2=("y_hit_up_20", "mean"),
              hit_dn_2=("y_hit_dn_20", "mean")).reset_index()
    t["mfe_mae_ratio"] = t["mean_mfe"] / t["mean_mae"].abs()
    t["realized_bowley"] = g["y_mag"].apply(realized_skew).values
    t["realized_tail"] = g["y_mag"].apply(realized_tail_weight).values
    qs = g["y_mag"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    for q, name in [(0.1, "p10"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (0.9, "p90")]:
        t[name] = qs[q].values
    t["bucket"] = t["bucket"].astype(str)
    return t


def flag_summary(o: pd.DataFrame) -> pd.DataFrame:
    d = o[o["y_mag"].notna()]
    rows = []
    for h, dh in d.groupby("horizon"):
        for name, m in [("positive_skew", dh["positive_skew"]), ("not_flagged", ~dh["positive_skew"]),
                        ("low_confidence", dh["confidence"] == "low"), ("all", pd.Series(True, index=dh.index))]:
            x = dh[m.astype(bool)]
            if len(x) == 0:
                continue
            rows.append({"horizon": h, "group": name, "n": len(x), "mean_mag": x["y_mag"].mean(),
                         "mean_ret": x["y_ret"].mean(), "hit": x["y_dir"].mean(),
                         "mfe_mae_ratio": x["y_mfe"].mean() / abs(x["y_mae"].mean()),
                         "realized_bowley": realized_skew(x["y_mag"]),
                         "realized_tail": realized_tail_weight(x["y_mag"])})
    return pd.DataFrame(rows)


def overlay_eval(o: pd.DataFrame) -> pd.DataFrame:
    """Hold-if-still-strong vs fixed horizon, only where confidence is normal."""
    d = o[o["y_overlay_ret"].notna() & o["still_strong_eligible"]]
    rows = []
    for h, dh in d.groupby("horizon"):
        for name, x in [("all_eligible", dh), ("positive_skew", dh[dh["positive_skew"]])]:
            if len(x) == 0:
                continue
            rows.append({"horizon": h, "group": name, "n": len(x), "share_extended": x["y_still_strong"].mean(),
                         "fixed_ret": x["y_ret"].mean(), "overlay_ret": x["y_overlay_ret"].mean()})
    return pd.DataFrame(rows)


def ic_oos(o: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank IC of the predictions themselves, OOS."""
    rows = []
    for h, dh in o[o["y_mag"].notna()].groupby("horizon"):
        for c, y in [("p_up", "y_mag"), ("mag_q50", "y_mag"), ("skew_score", "y_mag"), ("skew_score", "y_mfe"),
                     ("mfe_q50", "y_mfe"), ("mae_q50", "y_mae")]:
            ic = dh.groupby("date").apply(lambda g: g[c].corr(g[y], method="spearman") if len(g) >= 20 else np.nan)
            rows.append({"horizon": h, "score": c, "label": y, "mean_ic": ic.mean(),
                         "ir": ic.mean() / ic.std(), "n_days": int(ic.notna().sum())})
    return pd.DataFrame(rows)


def tail_buckets(o: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Does predicted tail weight show up in realized outcomes? Rows bucketed into per-date
    quintiles of predicted tail_weight; realized tail weight, spread and big-move rates per bucket.
    Per-date ranks keep a calm month and a wild month from landing in different buckets."""
    if "tail_weight" not in o:
        return pd.DataFrame()
    d = o[o["y_mag"].notna() & o["tail_weight"].notna()].copy()
    d["bucket"] = (d.groupby(["horizon", "date"])["tail_weight"].rank(pct=True).mul(n)
                     .clip(upper=n - 1e-9).astype(int) + 1)
    rows = []
    for (h, b), x in d.groupby(["horizon", "bucket"]):
        y = x["y_mag"]
        rows.append({"horizon": h, "bucket": int(b), "n": len(x), "pred_tail": x["tail_weight"].mean(),
                     "realized_tail": realized_tail_weight(y), "iqr": y.quantile(.75) - y.quantile(.25),
                     "p10": y.quantile(.1), "p90": y.quantile(.9), "mean_abs": y.abs().mean()})
    t = pd.DataFrame(rows)
    # share of outcomes beyond 2x the horizon's overall inner range from the median: "big moves"
    for h, x in d.groupby("horizon"):
        y = x["y_mag"]
        med, iqr = y.median(), y.quantile(.75) - y.quantile(.25)
        big = (y - med).abs() > 2 * iqr
        share = big.groupby(x["bucket"]).mean()
        t.loc[t["horizon"] == h, "big_move_share"] = t.loc[t["horizon"] == h, "bucket"].map(share).values
    return t


def edge_timeseries(o: pd.DataFrame, score: str = "p_up") -> pd.DataFrame:
    """Monthly mean of the daily top-decile minus bottom-decile spread in the trained label."""
    d = o[o["y_mag"].notna()].copy()
    d["dec"] = d.groupby(["horizon", "date"])[score].rank(pct=True).mul(10).clip(upper=9.999).astype(int) + 1
    daily = (d[d["dec"].isin([1, 10])].groupby(["horizon", "date", "dec"])["y_mag"].mean()
               .unstack("dec").dropna())
    if daily.empty:
        return pd.DataFrame(columns=["horizon", "month", "spread", "n_days"])
    daily["spread"] = daily[10] - daily[1]
    daily = daily.reset_index()
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    return (daily.groupby(["horizon", "month"])
                 .agg(spread=("spread", "mean"), n_days=("spread", "size")).reset_index())


def evaluate_all(o: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {"direction": direction_metrics(o), "calibration": calibration_table(o), "lift": decile_lift(o),
            "quantiles": quantile_metrics(o), "skew_buckets": skew_buckets(o), "flags": flag_summary(o),
            "overlay": overlay_eval(o), "ic_oos": ic_oos(o), "edge": edge_timeseries(o),
            "tails": tail_buckets(o)}
