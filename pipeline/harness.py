"""
pipeline/harness.py
Purged walk-forward, per horizon. Everything that learns from data — the IC gate, the combination
search, every model, the calibrators, the quantile models and the anomaly threshold — is fit inside
the fold's training window only.

Fold layout for horizon h (indices are trading sessions; purge P = max(purge_min, h + 1)):

    [ fit ............ ] P [ calib ] P [ test (test_sessions) ]
    ^ train_start              ^ train_end           ^ s

Rows dated on or before train_end have labels that end before the test block starts, and fit rows'
labels end before the calibration slice starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .features import model_feature_cols
from .ic_screen import DailyIC, combination_search, core_targets, screen, survivors
from .models import (AnomalyFlag, BinIsotonicDir, BoostDir, Calibrator, ForestDir, LinearDir,
                     QuantileSet, design)
from .targets import add_cross_sectional_labels
from .skew import bowley, path_skew_score, positive_skew_flag, skew_score, tail_weight

LABELS = ["ret", "dir_ret", "excess", "dir_excess", "rel", "rel_rank", "dir_xs", "mfe", "mae", "range",
          "oc_mag", "dir_oc", "still_strong", "overlay_ret", "hit_up_20", "hit_dn_20"]


def prepare(feats: pd.DataFrame, targs: pd.DataFrame, horizons: Optional[List[int]] = None) -> pd.DataFrame:
    p = feats.merge(targs, on=["date", "ticker"], how="left")
    p = p.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True).copy()
    hs = horizons or sorted({int(c.split("_")[1]) for c in p.columns if c.startswith("rel_") and c[4:].isdigit()})
    p = add_cross_sectional_labels(p, hs)
    hv = p["hv20_pct"]
    p["hv_regime"] = np.select([hv <= 0.33, hv <= 0.66, hv > 0.66], ["low", "mid", "high"], "unknown")
    return p


@dataclass
class Fold:
    k: int
    train_start: int
    fit_end: int
    cal_start: int
    train_end: int
    test_start: int
    test_end: int


def make_folds(n_dates: int, cfg: Config, h: int) -> List[Fold]:
    P = cfg.purge(h)
    folds, k = [], 0
    s = cfg.train_min_sessions + P
    while s < n_dates:
        train_end = s - P - 1
        train_start = 0 if cfg.window == "expanding" else max(0, train_end - cfg.rolling_sessions + 1)
        span = train_end - train_start + 1
        cal_len = max(21, int(cfg.calib_frac * span))
        cal_start = train_end - cal_len + 1
        fit_end = cal_start - P - 1
        if fit_end - train_start < 63:
            s += cfg.test_sessions
            continue
        folds.append(Fold(k, train_start, fit_end, cal_start, train_end, s, min(s + cfg.test_sessions, n_dates) - 1))
        k += 1
        s += cfg.test_sessions
    return folds


@dataclass
class Bundle:
    h: int
    features: List[str]
    pairs: List[Tuple[str, str]]
    dir_models: Dict[str, object]
    calibrators: Dict[str, Calibrator]
    q_ret: QuantileSet
    q_mfe: QuantileSet
    q_mae: QuantileSet
    anomaly: AnomalyFlag
    ic_table: pd.DataFrame
    combos: pd.DataFrame
    gate_relaxed: bool = False
    fit_rows: int = 0
    X_fit_sample: Optional[pd.DataFrame] = None


def _rows(p: pd.DataFrame, dates, label: str) -> pd.DataFrame:
    r = p[p["date"].isin(dates)]
    return r[r[label].notna()]


def fit_bundle(p: pd.DataFrame, dic: DailyIC, udates: np.ndarray, cfg: Config, h: int,
               train_start: int, fit_end: int, cal_start: int, train_end: int) -> Bundle:
    fit_dates = pd.DatetimeIndex(udates[train_start:fit_end + 1])
    cal_dates = pd.DatetimeIndex(udates[cal_start:train_end + 1])
    targets = core_targets(h, cfg.target_mode)
    lab, mag = cfg.dir_label(h), cfg.mag_label(h)

    tab = screen(dic, targets, fit_dates, cfg)
    surv = survivors(tab, cfg.max_survivors)
    relaxed = False
    if len(surv) < 3:  # gate found too little: fall back to the strongest by |t| on direction, and say so
        relaxed = True
        d = tab[(tab["target"] == lab) & (tab["condition"] == "all")]
        surv = d.assign(a=d["t"].abs()).sort_values("a", ascending=False)["feature"].head(10).tolist()
    combos = combination_search(dic, surv, [lab, mag], fit_dates, cfg)
    pairs = list(zip(combos["a"], combos["b"])) if len(combos) else []

    fit = _rows(p, fit_dates, lab)
    if len(fit) > cfg.max_train_rows:
        fit = fit.sample(cfg.max_train_rows, random_state=cfg.seed).sort_index()
    cal = _rows(p, cal_dates, lab)
    Xf, Xc = design(fit, surv, pairs), design(cal, surv, pairs)
    yf, yc = fit[lab], cal[lab]

    dir_models = {"linear": LinearDir().fit(Xf, yf), "rf": ForestDir(cfg.seed).fit(Xf, yf),
                  "gbm": BoostDir(cfg.seed).fit(Xf, yf), "bin_iso": BinIsotonicDir(surv[0]).fit(Xf, yf)}
    calibrators = {k: Calibrator().fit(m.score(Xc), yc.to_numpy(float)) for k, m in dir_models.items()}

    fit_m = fit[fit[mag].notna()]
    q_ret = QuantileSet([0.1, 0.25, 0.5, 0.75, 0.9], cfg.seed).fit(Xf.loc[fit_m.index], fit_m[mag])
    q_mfe = QuantileSet([0.5, 0.9], cfg.seed).fit(Xf, fit[f"mfe_{h}"])
    q_mae = QuantileSet([0.1, 0.5], cfg.seed).fit(Xf, fit[f"mae_{h}"])
    anomaly = AnomalyFlag(cfg.iso_contamination, cfg.seed).fit(Xf)
    return Bundle(h, surv, pairs, dir_models, calibrators, q_ret, q_mfe, q_mae, anomaly, tab, combos,
                  relaxed, len(fit), Xf.sample(min(2000, len(Xf)), random_state=cfg.seed))


def predict_bundle(b: Bundle, rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    X = design(rows, b.features, b.pairs)
    h = b.h
    out = rows[["date", "ticker", "sector", "hv_regime", "post_climax"]
               + [c for c in ("bench_etf", "scr_premarket", "scr_leader", "scr_potent") if c in rows]].copy()
    out["horizon"] = h
    probs = {}
    for k, m in b.dir_models.items():
        probs[k] = b.calibrators[k](m.score(X))
        out[f"p_{k}"] = probs[k]
    P = np.column_stack(list(probs.values()))
    out["p_up"] = P[:, :3].mean(axis=1)                  # ensemble of linear, rf, gbm (calibrated)
    out["disagreement"] = P.std(axis=1)
    qr, qm, qa = b.q_ret.predict(X), b.q_mfe.predict(X), b.q_mae.predict(X)
    for c in ("q10", "q25", "q50", "q75", "q90"):
        out[f"mag_{c}"] = qr[c].values
    out["mfe_q50"], out["mfe_q90"] = qm["q50"].values, qm["q90"].values
    out["mae_q10"], out["mae_q50"] = qa["q10"].values, qa["q50"].values
    out["skew_score"] = skew_score(out["mag_q10"], out["mag_q50"], out["mag_q90"], cfg.skew_floor_abs, cfg.skew_floor_rel)
    out["path_skew"] = path_skew_score(out["mfe_q50"], out["mae_q50"], cfg.skew_floor_abs)
    out["bowley"] = bowley(out["mag_q10"], out["mag_q50"], out["mag_q90"])
    out["tail_weight"] = tail_weight(out["mag_q10"], out["mag_q25"], out["mag_q75"], out["mag_q90"])
    out["anomaly"] = b.anomaly.score(X)
    out["confidence"] = np.where(out["anomaly"] > b.anomaly.threshold, "low", "normal")
    out["positive_skew"] = positive_skew_flag(out["skew_score"], out["p_up"], out["confidence"],
                                              cfg.skew_threshold, cfg.p_up_min)
    out["still_strong_eligible"] = out["confidence"] != "low"
    for l in LABELS:
        col = f"{l}_{h}"
        if col in rows:
            out[f"y_{l}"] = rows[col].values
    # the labels this run's models were trained on (relative by default)
    out["y_dir"] = rows[cfg.dir_label(h)].values if cfg.dir_label(h) in rows else np.nan
    out["y_mag"] = rows[cfg.mag_label(h)].values if cfg.mag_label(h) in rows else np.nan
    return out


def walk_forward(p: pd.DataFrame, cfg: Config, horizons: Optional[List[int]] = None,
                 log=print) -> Dict[str, object]:
    horizons = horizons or cfg.horizons
    feats = model_feature_cols(p)
    dic = DailyIC(p, feats, cfg.ic_min_names_per_day)
    p = dic.df  # date-sorted copy aligned with the IC ranks
    udates = dic.udates
    preds, gates, last = [], [], {}
    for h in horizons:
        folds = make_folds(len(udates), cfg, h)
        log(f"h={h}: {len(folds)} folds")
        for f in folds:
            b = fit_bundle(p, dic, udates, cfg, h, f.train_start, f.fit_end, f.cal_start, f.train_end)
            test = p[p["date"].isin(udates[f.test_start:f.test_end + 1])]
            pr = predict_bundle(b, test, cfg)
            pr["fold"] = f.k
            preds.append(pr)
            gates.append({"horizon": h, "fold": f.k, "train_end": udates[f.train_end],
                          "test_start": udates[f.test_start], "test_end": udates[f.test_end],
                          "n_fit": b.fit_rows, "gate_relaxed": b.gate_relaxed,
                          "survivors": "|".join(b.features),
                          "interactions": "|".join(f"{a}__x__{c}" for a, c in b.pairs)})
            last[h] = b
            log(f"  fold {f.k}: test {str(udates[f.test_start])[:10]}..{str(udates[f.test_end])[:10]}"
                f"  fit={b.fit_rows:,}  survivors={len(b.features)}  pairs={len(b.pairs)}"
                f"{'  (gate relaxed)' if b.gate_relaxed else ''}")
    oos = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return {"oos": oos, "gates": pd.DataFrame(gates), "last_bundles": last, "dic": dic, "panel": p}


def fit_live(p: pd.DataFrame, dic: DailyIC, cfg: Config, h: int) -> Bundle:
    """Train on every date whose h-label is complete, with the same purged calibration tail."""
    udates = dic.udates
    have = p.loc[p[cfg.dir_label(h)].notna(), "date"]
    last_lab = np.searchsorted(udates, have.max())
    P = cfg.purge(h)
    train_start = 0 if cfg.window == "expanding" else max(0, last_lab - cfg.rolling_sessions + 1)
    span = last_lab - train_start + 1
    cal_len = max(21, int(cfg.calib_frac * span))
    cal_start = last_lab - cal_len + 1
    fit_end = cal_start - P - 1
    return fit_bundle(p, dic, udates, cfg, h, train_start, fit_end, cal_start, last_lab)
