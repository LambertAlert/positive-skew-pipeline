"""
pipeline/run.py
End to end: Massive -> LUK universe ($9+) -> features/targets -> purged walk-forward -> OOS metrics
-> live scores for the next session -> SHAP artifacts -> dashboard.

    export MASSIVE_API_KEY=...
    python -m pipeline.run --start 2021-01-04                   # full run
    python -m pipeline.run --start 2021-01-04 --horizons 5 10   # fewer horizons
    python -m pipeline.run --live-only                          # skip walk-forward, just score tomorrow
    python -m pipeline.run --source synthetic                   # offline smoke test

Outputs (data/psp_out/):
    oos_predictions.parquet   walk-forward predictions + realized labels
    live_scores.parquet/.csv  next-session scores with top SkewSHAP drivers
    metrics_*.csv             direction, calibration, lift, quantiles, skew buckets, flags, overlay, OOS IC
    gates.csv                 survivors and interactions per fold
    ic_latest_h*.csv          the live model's IC screen
    shap_*.csv                global importance, SkewSHAP, interactions, dependence samples
    dashboard.html            built from the files above
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import evaluate_all
from .explain import (beeswarm_sample, dependence_sample, expected_value, global_importance,
                      interaction_strength, skew_base, skew_shap, top_contributors, tree_shap)
from .features import build_features
from .harness import fit_live, predict_bundle, prepare, walk_forward
from .ic_screen import DailyIC
from .features import model_feature_cols
from .models import design
from .overlay import apply_overlay, read_regime
from .targets import build_targets
from .universe import append_next_session, build_universe


def load(cfg: Config, source: str, log):
    if source == "synthetic":
        from .synthetic import make_market
        bars, spy, shares, sic, etfs = make_market(signal=float(os.environ.get("PSP_SYNTH_SIGNAL", "0.6")))
        bars = append_next_session(bars)
        tagged = build_universe(bars, cfg.luk, shares=shares, sic=sic)
        return tagged, spy, etfs, {"first_massive": tagged["date"].min(), "backfilled": False}
    from .massive_client import MassiveClient
    from .universe import load_market
    t0 = time.time()
    m = load_market(cfg, MassiveClient(cfg), next_session=True, log=log)
    log(f"Massive load {time.time() - t0:.0f}s: {m['bars']['ticker'].nunique():,} candidate tickers, "
        f"{int(m['bars']['in_universe'].sum()):,} universe name-days")
    if m["backfilled"]:
        log(f"Massive history starts {m['first_massive'].date()}; earlier bars come from yfinance "
            f"({'feature warm-up only' if cfg.backfill_use == 'warmup' else 'warm-up and training'})")
    return m["bars"], m["spy"], m["etfs"], {"first_massive": m["first_massive"], "backfilled": m["backfilled"]}


def shap_artifacts(b, X_live: pd.DataFrame, live_meta: pd.DataFrame, out: Path, h: int, panel: pd.DataFrame):
    Xs = b.X_fit_sample
    res = {}
    gb = tree_shap(b.dir_models["gbm"].estimator, Xs)
    rf = tree_shap(b.dir_models["rf"].estimator, Xs.head(800))
    lin = b.dir_models["linear"].shap(Xs)
    sk = skew_shap(b.q_ret, Xs)
    if gb is not None:
        global_importance(gb).to_csv(out / f"shap_gbm_h{h}.csv")
        res["gbm"] = global_importance(gb).head(15)
    if rf is not None:
        global_importance(rf).to_csv(out / f"shap_rf_h{h}.csv")
    global_importance(lin).to_csv(out / f"shap_linear_h{h}.csv")
    if sk is not None:
        g = pd.DataFrame({"skew_mean_abs": sk["skew"].abs().mean(), "skew_mean": sk["skew"].mean(),
                          "upper_mean": sk["upper"].mean(), "lower_mean": sk["lower"].mean()}
                         ).sort_values("skew_mean_abs", ascending=False)
        g.to_csv(out / f"shap_skew_h{h}.csv")
        res["skew"] = g.head(15)
        top = g.index[:4].tolist()
        # colour keys come from the panel, not the design matrix (the flags may not have passed the gate)
        color = panel.loc[Xs.index, ["hv_elevated", "post_climax"]].fillna(0)
        dependence_sample(Xs, sk["skew"], top, color).to_csv(out / f"shap_dependence_h{h}.csv", index=False)
        bee = beeswarm_sample(Xs, sk["skew"], g.index[:10].tolist())
        bee.to_csv(out / f"shap_beeswarm_h{h}.csv", index=False)
        res["beeswarm"] = bee
    try:
        inter = interaction_strength(b.dir_models["gbm"].estimator, Xs.head(150))
        inter.to_csv(out / f"shap_interactions_h{h}.csv", index=False)
        res["interactions"] = inter
    except Exception:  # interaction values need lightgbm + shap; non-fatal
        res["interactions"] = pd.DataFrame()
    # local drivers for every live row
    if sk is not None and len(X_live):
        live_sk = skew_shap(b.q_ret, X_live)["skew"]
        live_meta = live_meta.copy()
        live_meta["skew_drivers"] = [json.dumps([[k, round(v, 6)] for k, v in row])
                                     for row in top_contributors(live_sk, 6)]
        # waterfall ends: base value and this name's total (base + every feature's SHAP)
        sb = skew_base(b.q_ret)
        live_meta["skew_base"] = sb
        live_meta["skew_total"] = sb + live_sk.sum(axis=1).values
        if gb is not None:
            live_gb = tree_shap(b.dir_models["gbm"].estimator, X_live)
            live_meta["dir_drivers"] = [json.dumps([[k, round(v, 5)] for k, v in row])
                                        for row in top_contributors(live_gb, 6)]
            db = expected_value(b.dir_models["gbm"].estimator)
            live_meta["dir_base"] = db
            live_meta["dir_total"] = db + live_gb.sum(axis=1).values
    return res, live_meta


METRIC_KEYS = ("direction", "calibration", "lift", "quantiles", "skew_buckets", "flags", "overlay", "ic_oos", "edge", "tails")


def rebuild_dashboard(out: Path, cfg: Config):
    """Rebuild dashboard.html from a finished run's files (no data pull, no model fits)."""
    from .dashboard import write_dashboard
    meta = json.loads((out / "meta.json").read_text())
    cfg.horizons = meta["horizons"]
    metrics = {k: pd.read_csv(out / f"metrics_{k}.csv") for k in METRIC_KEYS if (out / f"metrics_{k}.csv").exists()}
    live = pd.read_parquet(out / "live_scores.parquet")
    shap_res = {}
    for h in cfg.horizons:
        r = {}
        for k, fn in (("gbm", f"shap_gbm_h{h}.csv"), ("skew", f"shap_skew_h{h}.csv")):
            if (out / fn).exists():
                r[k] = pd.read_csv(out / fn, index_col=0).head(15)
        if (out / f"shap_beeswarm_h{h}.csv").exists():
            r["beeswarm"] = pd.read_csv(out / f"shap_beeswarm_h{h}.csv")
        if (out / f"shap_interactions_h{h}.csv").exists():
            r["interactions"] = pd.read_csv(out / f"shap_interactions_h{h}.csv")
        shap_res[h] = r
    write_dashboard(out, meta, metrics, live, shap_res, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="massive", choices=["massive", "synthetic"])
    ap.add_argument("--start", help='first scored day, or "auto" (default): plan start + warm-up')
    ap.add_argument("--end")
    ap.add_argument("--horizons", nargs="*", type=int)
    ap.add_argument("--max-train-rows", type=int)
    ap.add_argument("--test-sessions", type=int)
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--dashboard-only", action="store_true", help="rebuild dashboard.html from saved outputs")
    ap.add_argument("--out")
    ap.add_argument("--target-mode", choices=["relative", "absolute"])
    ap.add_argument("--regime-path")
    ap.add_argument("--backfill", choices=["auto", "yfinance", "none"])
    ap.add_argument("--backfill-use", choices=["warmup", "full"])
    ap.add_argument("--profile", default="auto", choices=["auto", "standard", "short"])
    ap.add_argument("--calls-per-minute", type=int)
    a = ap.parse_args(argv)

    cfg = Config()
    if a.start: cfg.start = a.start
    if a.end: cfg.end = a.end
    if a.horizons: cfg.horizons = a.horizons
    if a.max_train_rows: cfg.max_train_rows = a.max_train_rows
    if a.test_sessions: cfg.test_sessions = a.test_sessions
    if a.out: cfg.out_dir = a.out
    if a.target_mode: cfg.target_mode = a.target_mode
    if a.regime_path: cfg.regime_path = a.regime_path
    if a.backfill: cfg.backfill = a.backfill
    if a.backfill_use: cfg.backfill_use = a.backfill_use
    if a.calls_per_minute: cfg.calls_per_minute = a.calls_per_minute
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    if a.dashboard_only:
        rebuild_dashboard(out, cfg)
        log(f"rebuilt -> {out / 'dashboard.html'}")
        return

    tagged, spy, etfs, hist = load(cfg, a.source, log)
    feats = build_features(tagged, spy)
    first_real = pd.Timestamp(hist["first_massive"])
    if a.source == "massive":
        floor = pd.Timestamp(cfg.start)
        if hist["backfilled"] and cfg.backfill_use == "warmup":
            # yfinance rows only feed the rolling features; train and score on Massive dates
            floor = max(floor, first_real)
        else:
            # with no backfill, rows inside the warm-up have incomplete features
            floor = max(floor, first_real + pd.tseries.offsets.BDay(cfg.warmup_sessions))
        feats = feats[feats["date"] >= floor]
    targs = build_targets(tagged, spy, cfg.horizons, cfg.entry, cfg.hit_thresholds, etfs=etfs)
    panel = prepare(feats, targs, cfg.horizons)
    live_date = panel["date"].max()
    prof = cfg.apply_profile(panel["date"].nunique(), a.profile)
    if prof == "short":
        log(f"short history ({panel['date'].nunique()} sessions): walk-forward uses "
            f"{cfg.train_min_sessions}-session minimum training and {cfg.test_sessions}-session test blocks")
    log(f"panel {len(panel):,} rows, {panel['ticker'].nunique():,} names, "
        f"{panel['date'].nunique():,} dates; scoring session {live_date.date()}")

    meta = {"source": a.source, "start": str(panel["date"].min().date()), "end": str(live_date.date()),
            "horizons": cfg.horizons, "price_floor": cfg.luk.min_price, "screens": cfg.luk.screens,
            "entry": cfg.entry, "target_mode": cfg.target_mode, "profile": prof,
            "backfilled": bool(hist["backfilled"]), "backfill_use": cfg.backfill_use,
            "first_massive": str(first_real.date()), "n_names": int(panel["ticker"].nunique()),
            "avg_universe": float(panel.groupby("date").size().mean()), "generated": time.strftime("%Y-%m-%d %H:%M")}

    metrics = {}
    if not a.live_only:
        wf = walk_forward(panel, cfg, log=log)
        oos = wf["oos"]
        oos.to_parquet(out / "oos_predictions.parquet", index=False)
        wf["gates"].to_csv(out / "gates.csv", index=False)
        # with backfill_use="full", folds may test on survivor-only yfinance dates: keep them out of the scores
        ev = oos[oos["date"] >= first_real] if hist["backfilled"] else oos
        metrics = evaluate_all(ev)
        for k, v in metrics.items():
            v.to_csv(out / f"metrics_{k}.csv", index=False)
        dic, panel_sorted = wf["dic"], wf["panel"]
    else:
        # keep showing the last walk-forward's OOS metrics, if a previous run left them
        for k in METRIC_KEYS:
            fp = out / f"metrics_{k}.csv"
            if fp.exists():
                metrics[k] = pd.read_csv(fp)
        dic = DailyIC(panel, model_feature_cols(panel), cfg.ic_min_names_per_day)
        panel_sorted = dic.df

    live_rows = panel_sorted[panel_sorted["date"] == live_date]
    lives, shap_res = [], {}
    for h in cfg.horizons:
        b = fit_live(panel_sorted, dic, cfg, h)
        b.ic_table.to_csv(out / f"ic_latest_h{h}.csv", index=False)
        pr = predict_bundle(b, live_rows, cfg)
        X_live = design(live_rows, b.features, b.pairs)
        shap_res[h], pr = shap_artifacts(b, X_live, pr, out, h, panel_sorted)
        pr["survivors"] = len(b.features)
        lives.append(pr)
        log(f"live h={h}: {len(pr)} names, {int(pr['positive_skew'].sum())} positive-skew")
    live = pd.concat(lives, ignore_index=True)
    regime = read_regime(cfg.regime_path)
    live = apply_overlay(live, regime)
    meta["regime"] = regime
    log(f"regime overlay: {regime.get('posture', 'no snapshot at ' + cfg.regime_path)}")
    live.to_parquet(out / "live_scores.parquet", index=False)
    live.to_csv(out / "live_scores.csv", index=False)

    (out / "meta.json").write_text(json.dumps(meta, default=str))
    from .dashboard import write_dashboard
    write_dashboard(out, meta, metrics, live, shap_res, cfg)
    log(f"done -> {out / 'dashboard.html'}")


if __name__ == "__main__":
    main()
