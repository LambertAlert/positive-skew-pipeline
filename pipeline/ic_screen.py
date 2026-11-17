"""
pipeline/ic_screen.py
Cross-sectional Spearman IC: one IC per date (features and labels ranked across that day's
universe), then mean, std, IR, t-stat, hit rate and quarterly sign stability over a set of dates.

The daily IC series is computed once for the whole sample. That is not leakage: date d's IC uses
only rows from date d and labels realized by d+h. Each walk-forward fold then aggregates only the
dates inside its (purged) training window, so the gate never sees test-period outcomes.

Conditions: 'all', HV20-regime buckets on the stock's own past-only HV20 percentile
(low <= .33 < mid <= .66 < high), and 'post_climax' (within 5 sessions of an extreme-volume day).
Within a condition the date-wide ranks are re-centred on that subset; this is Pearson-on-ranks,
a close stand-in for re-ranking inside each subset.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CONDITIONS = ["all", "hv_low", "hv_mid", "hv_high", "post_climax"]


def condition_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    hv = df["hv20_pct"].to_numpy(dtype=float)
    m = {"all": np.ones(len(df), dtype=bool),
         "hv_low": hv <= 0.33,
         "hv_mid": (hv > 0.33) & (hv <= 0.66),
         "hv_high": hv > 0.66}
    if "post_climax" in df:
        m["post_climax"] = df["post_climax"].to_numpy(dtype=float) == 1.0
    return m


class DailyIC:
    """Holds date-wise ranks and computes daily IC matrices on demand (cached)."""

    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str], min_names: int = 20):
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
        self.df = df
        self.dates = df["date"].to_numpy()
        self.udates, self.starts = np.unique(self.dates, return_index=True)
        self.codes = np.searchsorted(self.udates, self.dates)
        self.features = list(feature_cols)
        self.min_names = min_names
        self.X = df[self.features].groupby(df["date"]).rank(pct=True).to_numpy(dtype=np.float32)
        self.masks = condition_masks(df)
        self._cache: Dict[tuple, pd.DataFrame] = {}

    def _blocksum(self, a: np.ndarray) -> np.ndarray:
        return np.add.reduceat(a, self.starts, axis=0)

    def ic(self, target: str, condition: str = "all", X: Optional[np.ndarray] = None,
           names: Optional[List[str]] = None) -> pd.DataFrame:
        """Dates x features DataFrame of daily IC for one label under one condition."""
        key = (target, condition) if X is None else None
        if key and key in self._cache:
            return self._cache[key]
        X = self.X if X is None else X
        names = self.features if names is None else names
        y = self.df[target].groupby(self.df["date"]).rank(pct=True).to_numpy(dtype=np.float64)
        use = self.masks[condition] & ~np.isnan(y)
        yz = np.where(use, y, 0.0)
        parts = []
        for j in range(0, X.shape[1], 8):                                  # column chunks bound memory
            Xc = X[:, j:j + 8].astype(np.float64)
            vx = ~np.isnan(Xc) & use[:, None]
            cnt = self._blocksum(vx.astype(np.float64))                    # D x k, jointly valid names
            with np.errstate(invalid="ignore", divide="ignore"):
                mx = self._blocksum(np.where(vx, Xc, 0.0)) / cnt
                my = self._blocksum(vx * yz[:, None]) / cnt
                dx = np.where(vx, Xc - mx[self.codes], 0.0)
                dy = np.where(vx, yz[:, None] - my[self.codes], 0.0)
                icc = self._blocksum(dx * dy) / np.sqrt(self._blocksum(dx * dx) * self._blocksum(dy * dy))
            icc[cnt < self.min_names] = np.nan
            parts.append(icc)
        ic = np.concatenate(parts, axis=1) if parts else np.empty((len(self.udates), 0))
        out = pd.DataFrame(ic, index=pd.DatetimeIndex(self.udates), columns=names)
        if key:
            self._cache[key] = out
        return out


def summarize(daily: pd.DataFrame, dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    d = daily if dates is None else daily.loc[daily.index.isin(dates)]
    mean, std, n = d.mean(), d.std(), d.notna().sum()
    q = d.groupby(d.index.to_period("Q")).mean()
    sign = np.sign(mean)
    stab = (np.sign(q) == sign).sum() / q.notna().sum().replace(0, np.nan)
    return pd.DataFrame({
        "mean_ic": mean, "std_ic": std, "ir": mean / std, "t": mean / std * np.sqrt(n),
        "pct_pos": (d > 0).sum() / n.replace(0, np.nan), "sign_stability": stab, "n_days": n,
    })


def screen(dic: DailyIC, targets: Sequence[str], train_dates: pd.DatetimeIndex, cfg,
           conditions: Sequence[str] = CONDITIONS) -> pd.DataFrame:
    """Long table: feature x target x condition stats over the training dates, with a pass flag."""
    rows = []
    for t in targets:
        for cond in conditions:
            if cond not in dic.masks:
                continue
            s = summarize(dic.ic(t, cond), train_dates)
            s["target"], s["condition"] = t, cond
            rows.append(s.rename_axis("feature").reset_index())
    tab = pd.concat(rows, ignore_index=True)
    tab["passes"] = ((tab["mean_ic"].abs() >= cfg.ic_min_abs) & (tab["ir"].abs() >= cfg.ic_min_abs_ir)
                     & (tab["sign_stability"] >= cfg.ic_min_sign_stability))
    return tab


def survivors(tab: pd.DataFrame, max_n: int) -> List[str]:
    """Features passing under any target/condition, ordered by their best |t|."""
    ok = tab[tab["passes"]]
    if ok.empty:
        return []
    best = ok.assign(abst=ok["t"].abs()).groupby("feature")["abst"].max().sort_values(ascending=False)
    return best.index[:max_n].tolist()


def interaction_matrix(df_sorted: pd.DataFrame, pairs: Sequence[tuple]) -> np.ndarray:
    """Centred-rank products (a-.5)(b-.5) of peer-ranked features."""
    cols = [((df_sorted[a].to_numpy(np.float32) - 0.5) * (df_sorted[b].to_numpy(np.float32) - 0.5)) for a, b in pairs]
    return np.column_stack(cols) if cols else np.empty((len(df_sorted), 0))


def combination_search(dic: DailyIC, surv: Sequence[str], targets: Sequence[str],
                       train_dates: pd.DatetimeIndex, cfg, top_k: int = 10) -> pd.DataFrame:
    """Two-way products of the top survivors, screened like single features (unconditional)."""
    base = list(surv)[:top_k]
    pairs = [(a, b) for i, a in enumerate(base) for b in base[i + 1:]]
    if not pairs:
        return pd.DataFrame(columns=["feature", "a", "b"])
    names = [f"{a}__x__{b}" for a, b in pairs]
    dates = dic.df["date"].values
    blocks = []
    for j in range(0, len(pairs), 8):  # chunked to bound memory on the full universe
        raw = interaction_matrix(dic.df, pairs[j:j + 8])
        blocks.append(pd.DataFrame(raw).groupby(dates).rank(pct=True).to_numpy(np.float32))
    Xr = np.concatenate(blocks, axis=1)
    rows = []
    for t in targets:
        s = summarize(dic.ic(t, "all", X=Xr, names=names), train_dates)
        s["target"] = t
        rows.append(s.rename_axis("feature").reset_index())
    tab = pd.concat(rows, ignore_index=True)
    tab["passes"] = ((tab["mean_ic"].abs() >= cfg.ic_min_abs) & (tab["ir"].abs() >= cfg.ic_min_abs_ir)
                     & (tab["sign_stability"] >= cfg.ic_min_sign_stability))
    best = (tab[tab["passes"]].assign(abst=lambda x: x["t"].abs())
            .groupby("feature")["abst"].max().sort_values(ascending=False))
    keep = best.index[:cfg.max_interactions].tolist()
    lut = dict(zip(names, pairs))
    return pd.DataFrame([{"feature": k, "a": lut[k][0], "b": lut[k][1], "best_abs_t": best[k]} for k in keep])


def core_targets(h: int, mode: str = "relative") -> List[str]:
    if mode == "relative":
        return [f"dir_xs_{h}", f"rel_{h}", f"mfe_{h}", f"mae_{h}"]
    return [f"dir_ret_{h}", f"ret_{h}", f"excess_{h}", f"mfe_{h}", f"mae_{h}"]
