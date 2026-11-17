"""
Run: python -m pytest -q tests/        (from the repo root)

The leakage tests perturb the bar of day t and check nothing dated <= t moves.
"""

import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pipeline.config import Config, LukFilters
from pipeline.synthetic import make_market
from pipeline.universe import append_next_session, build_universe, luk_fields
from pipeline.features import build_features, model_feature_cols, _rolling_pct_rank, _anchor_index, _avwap
from pipeline.targets import build_targets
from pipeline.harness import make_folds, prepare, fit_bundle, predict_bundle, walk_forward
from pipeline.ic_screen import DailyIC
from pipeline.models import QuantileSet, ForestDir, design
from pipeline.skew import skew_score, positive_skew_flag
from pipeline.explain import tree_shap, skew_shap
from pipeline.massive_client import MassiveClient
from pipeline.universe import sic_to_etf
from pipeline.targets import add_cross_sectional_labels
from pipeline.overlay import apply_overlay


@pytest.fixture(scope="module")
def mkt():
    return make_market(n_tickers=60, n_days=420, signal=0.6, seed=3)


@pytest.fixture(scope="module")
def built(mkt):
    bars, spy, sh, sic, etfs = mkt
    cfg = Config()
    tg = build_universe(append_next_session(bars), cfg.luk, sh, sic)
    F = build_features(tg, spy)
    T = build_targets(tg, spy, [5, 21], etfs=etfs)
    return cfg, tg, F, T


# ------------------------------------------------------------------ universe
def test_price_floor_is_strictly_above_9():
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=30).repeat(1), "ticker": "X",
                       "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.0, "volume": 2e7})
    df.loc[20:, "close"] = 9.01
    tg = build_universe(df, LukFilters(), shares=pd.Series({"X": 1e9}))
    # day 21 is the first whose prior close (day 20) is 9.01 > 9
    assert not tg.loc[20, "scr_leader"] and tg.loc[21, "scr_leader"]


def test_universe_is_delay1(mkt):
    bars, spy, sh, sic, etfs = mkt
    f = LukFilters()
    a = build_universe(bars, f, sh, sic)
    t = bars["date"].unique()[300]
    b2 = bars.copy()
    m = b2["date"] == t
    b2.loc[m, ["close", "high"]] *= 3.0
    b2.loc[m, "volume"] *= 50
    b = build_universe(b2, f, sh, sic)
    keep = a["date"] <= t
    for c in ["scr_premarket", "scr_leader", "scr_potent", "in_universe"]:
        assert (a.loc[keep, c].values == b.loc[keep, c].values).all(), c


def test_mcap_unknown_kept_and_small_dropped():
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=15), "ticker": "X", "open": 20.0,
                       "high": 21.0, "low": 19.5, "close": 20.0, "volume": 2e7})
    small = build_universe(df, LukFilters(), shares=pd.Series({"X": 1e6}))   # $20M cap
    unknown = build_universe(df, LukFilters(), shares=pd.Series({"Y": 1e9}))
    assert not small["scr_leader"].any()
    assert unknown["scr_leader"].iloc[-1] and not unknown["mcap_known"].iloc[-1]


# ------------------------------------------------------------------ features
def test_features_are_delay1(mkt):
    bars, spy, sh, sic, etfs = mkt
    cfg = Config()
    tg = build_universe(bars, cfg.luk, sh, sic)
    t = bars["date"].unique()[330]
    tg2 = tg.copy()
    m = tg2["date"] == t
    tg2.loc[m, ["open", "high", "low", "close", "vwap"]] *= 1.5
    tg2.loc[m, "volume"] *= 20
    A, B = build_features(tg, spy), build_features(tg2, spy)
    cols = [c for c in A.columns if c not in ("date", "ticker", "sector", "peer_group", "bench_etf")]
    a, b = A[A["date"] <= t].set_index(["date", "ticker"]), B[B["date"] <= t].set_index(["date", "ticker"])
    a, b = a.align(b, join="inner")
    diff = (a[cols].astype(float) - b[cols].astype(float)).abs().max()
    assert diff.fillna(0).max() < 1e-6, diff[diff > 1e-6]
    # and the next day DOES move (the perturbation is visible one session later)
    nxt = A["date"][A["date"] > t].min()
    an = A[A["date"] == nxt].set_index("ticker")["mom_21"]
    bn = B[B["date"] == nxt].set_index("ticker")["mom_21"]
    assert (an - bn.reindex(an.index)).abs().max() > 1e-3


def test_hv20_pct_is_past_only():
    x = np.random.default_rng(0).normal(size=400)
    full = _rolling_pct_rank(x, 252, 63)
    trunc = _rolling_pct_rank(x[:300], 252, 63)
    assert np.allclose(full[:300], trunc, equal_nan=True)


def test_avwap_anchor_more_recent_extreme():
    c = np.array([10, 12, 11, 9, 10, 10.5], float)
    anc, side = _anchor_index(c, 4)
    # window ending at idx 5 = [11, 9, 10, 10.5]: high at 2, low at 3 -> anchor 3 (low)
    assert anc[5] == 3 and side[5] == -1
    v = np.ones(6)
    assert np.isclose(_avwap(c, v, anc)[5], np.mean(c[3:6]))


def test_peer_ranks_in_unit_interval(built):
    _, _, F, _ = built
    pr = F[[c for c in F.columns if c.endswith("_pr")]]
    assert (pr.min().min() >= 0) and (pr.max().max() <= 1)


# ------------------------------------------------------------------ targets
def test_targets_by_hand():
    d = pd.bdate_range("2024-01-01", periods=8)
    bars = pd.DataFrame({"date": d, "ticker": "X", "open": [10, 11, 12, 13, 14, 15, 16, 17],
                         "high": [10.5, 11.5, 14, 13.5, 14.5, 15.5, 16.5, 17.5],
                         "low": [9.5, 10.5, 11.5, 12.5, 8, 14.5, 15.5, 16.5],
                         "close": [10, 11, 12, 13, 14, 15, 16, 17], "volume": 1e6})
    spy = bars.assign(ticker="SPY")
    T = build_targets(bars, spy, [3])
    r = T.iloc[1]  # entry open day1 = 11, window days 1..3, exit close day3 = 13
    assert np.isclose(r["ret_3"], 13 / 11 - 1)
    assert np.isclose(r["mfe_3"], 14 / 11 - 1)       # high of day2
    assert np.isclose(r["mae_3"], 10.5 / 11 - 1)     # low of day1
    assert np.isclose(T.iloc[2]["mae_3"], 8 / 12 - 1)
    # the last 2 rows lack a full window: every label NaN, never 0
    tail = T.iloc[-2:]
    for c in ["ret_3", "dir_ret_3", "mfe_3", "mae_3", "hit_up_10_3", "dir_excess_3"]:
        assert tail[c].isna().all(), c


def test_overlay_rule():
    d = pd.bdate_range("2024-01-01", periods=40)
    c = np.linspace(10, 20, 40)
    bars = pd.DataFrame({"date": d, "ticker": "X", "open": c, "high": c * 1.01, "low": c * .99, "close": c, "volume": 1e6})
    T = build_targets(bars, bars.assign(ticker="SPY"), [5])
    r = T.iloc[20]  # steady uptrend: still strong at exit, so held 5 more sessions
    assert r["still_strong_5"] == 1.0 and np.isclose(r["overlay_ret_5"], c[29] / c[20] - 1, rtol=1e-5)


# ------------------------------------------------------------------ IC
def test_daily_ic_matches_scipy(built):
    _, _, F, T = built
    p = prepare(F, T, [5, 21])
    feats = model_feature_cols(p)[:5]
    dic = DailyIC(p, feats, min_names=10)
    ic = dic.ic("ret_5")
    day = ic.dropna().index[len(ic.dropna()) // 2]
    g = dic.df[dic.df["date"] == day][[feats[0], "ret_5"]].dropna()
    ref = spearmanr(g[feats[0]], g["ret_5"]).correlation
    assert abs(ic.loc[day, feats[0]] - ref) < 0.02


# ------------------------------------------------------------------ harness
def test_folds_purged():
    cfg = Config()
    for h in (3, 21):
        for f in make_folds(1000, cfg, h):
            P = cfg.purge(h)
            assert f.test_start - f.train_end > P          # train labels end before test
            assert f.cal_start - f.fit_end > P             # fit labels end before calibration
            assert P >= h + 1


def test_no_skill_on_pure_noise():
    """Random walk: if anything leaks, OOS IC of p_up would be clearly positive."""
    bars, spy, sh, sic, etfs = make_market(n_tickers=80, n_days=560, signal=0.0, seed=11)
    cfg = Config(horizons=[5], test_sessions=126, train_min_sessions=200, max_survivors=12)
    tg = build_universe(append_next_session(bars), cfg.luk, sh, sic)
    p = prepare(build_features(tg, spy), build_targets(tg, spy, [5], etfs=etfs), [5])
    o = walk_forward(p, cfg, log=lambda *_: None)["oos"]
    o = o[o["y_mag"].notna()]
    ic = o.groupby("date").apply(lambda g: g["p_up"].corr(g["y_mag"], method="spearman"))
    assert abs(ic.mean()) < 0.03, ic.mean()


def test_planted_signal_recovered():
    bars, spy, sh, sic, etfs = make_market(n_tickers=80, n_days=560, signal=1.0, seed=12)
    cfg = Config(horizons=[5], test_sessions=126, train_min_sessions=200, max_survivors=12)
    tg = build_universe(append_next_session(bars), cfg.luk, sh, sic)
    p = prepare(build_features(tg, spy), build_targets(tg, spy, [5], etfs=etfs), [5])
    o = walk_forward(p, cfg, log=lambda *_: None)["oos"]
    o = o[o["y_mag"].notna()]
    ic = o.groupby("date").apply(lambda g: g["p_up"].corr(g["y_mag"], method="spearman"))
    assert ic.mean() > 0.03, ic.mean()
    assert (o["mag_q10"] <= o["mag_q50"]).all() and (o["mag_q50"] <= o["mag_q90"]).all()
    assert o["p_up"].between(0, 1).all()


# ------------------------------------------------------------------ skew / models / shap
def test_skew_score_floor_and_sign():
    s = skew_score([0.0, -0.02], [0.0, 0.0], [0.03, 0.02])
    assert np.isfinite(s).all() and (s >= 0).all()
    assert s[0] <= 0.03 / 0.003 + 1e-9         # floor = max(.002, .1 * .03)
    assert np.isclose(s[1], 1.0)
    f = positive_skew_flag([1.5, 1.5, 1.5, 1.1], [0.6, 0.4, 0.6, 0.6], ["normal", "normal", "low", "normal"])
    assert f.tolist() == [True, False, False, False]


def test_rf_shap_not_none():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((500, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(0, .3, 500) > .5).astype(int))
    rf = ForestDir().fit(X, y)
    sv = tree_shap(rf.estimator, X)
    assert sv is not None and sv.shape == X.shape
    q = QuantileSet([0.1, 0.5, 0.9]).fit(X, pd.Series(rng.normal(size=500)))
    s = skew_shap(q, X)
    assert s is not None and s["skew"].shape == X.shape


# ------------------------------------------------------------------ Massive client (offline)
class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, json.dumps(body)

    def json(self):
        return self._b


class _Sess:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.script.pop(0)


def test_client_grouped_parse_retry_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cfg = Config(cache_dir=str(tmp_path))
    row = {"T": "XOM", "v": 1.1e7, "vw": 170.18, "o": 171.43, "c": 170.59, "h": 171.71, "l": 169.21,
           "t": 1794603600000, "n": 150164}
    sess = _Sess([_Resp(429, {}), _Resp(200, {"results": [row], "resultsCount": 1})])
    cl = MassiveClient(cfg, api_key="k", session=sess)
    df = cl.grouped_day("2024-03-01")
    assert list(df.columns) == ["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "trades"]
    assert df.iloc[0]["close"] == 170.59 and sess.calls[-1][1]["adjusted"] == "true"
    cl.grouped_day("2024-03-01")            # served from cache: no new HTTP call
    assert len(sess.calls) == 2


def test_client_paginates_stock_set(tmp_path):
    cfg = Config(cache_dir=str(tmp_path), stock_types=["CS"], include_delisted=False)
    sess = _Sess([_Resp(200, {"results": [{"ticker": "A", "type": "CS", "active": True}],
                              "next_url": "https://api.massive.com/v3/reference/tickers?cursor=x"}),
                  _Resp(200, {"results": [{"ticker": "B", "type": "CS", "active": True}]})])
    s = MassiveClient(cfg, api_key="k", session=sess).stock_set()
    assert s["ticker"].tolist() == ["A", "B"] and sess.calls[1][1]["apiKey"] == "k"


# ------------------------------------------------------------------ relative targets
def test_sic_to_etf():
    assert sic_to_etf(3674) == "XLK" and sic_to_etf(7372) == "XLK" and sic_to_etf(2834) == "XLV"
    assert sic_to_etf(6022) == "XLF" and sic_to_etf(6798) == "XLRE" and sic_to_etf(1311) == "XLE"
    assert sic_to_etf(4911) == "XLU" and sic_to_etf(5812) == "XLY" and sic_to_etf(3711) == "XLY"
    assert sic_to_etf(3561) == "XLI" and sic_to_etf(None) == "SPY" and sic_to_etf(9999) == "SPY"


def test_relative_target_by_hand():
    d = pd.bdate_range("2024-01-01", periods=6)
    stock = pd.DataFrame({"date": d, "ticker": "X", "open": [10, 10, 10, 10, 10, 10], "high": 12.0, "low": 9.0,
                          "close": [10, 10, 11, 12, 12, 12], "volume": 1e6, "bench_etf": "XLK"})
    etf = pd.DataFrame({"date": d, "ticker": "XLK", "open": 100.0, "high": 101.0, "low": 99.0,
                        "close": [100, 100, 101, 102, 102, 102], "volume": 1e6})
    spy = etf.assign(ticker="SPY", close=100.0)
    T = build_targets(stock, spy, [2], etfs=etf)
    r = T.iloc[1]   # entry open day1, exit close day2: stock 11/10-1 = 10%, XLK 101/100-1 = 1%
    assert np.isclose(r["ret_2"], 0.10) and np.isclose(r["rel_2"], 0.09) and np.isclose(r["excess_2"], 0.10)
    # missing ETF -> falls back to SPY
    T2 = build_targets(stock.assign(bench_etf="XLZ"), spy, [2], etfs=etf)
    assert np.isclose(T2.iloc[1]["rel_2"], 0.10)


def test_dir_xs_is_half_per_day():
    p = pd.DataFrame({"date": np.repeat(pd.bdate_range("2024-01-01", periods=3), 10),
                      "rel_5": np.random.default_rng(0).normal(size=30)})
    q = add_cross_sectional_labels(p, [5])
    assert (q.groupby("date")["dir_xs_5"].mean() == 0.5).all()


def test_regime_overlay():
    live = pd.DataFrame({"positive_skew": [True, False]})
    assert apply_overlay(live, {"posture": "AVOID"})["trade_view"].tolist() == ["BEST OF WEAK TAPE", ""]
    assert apply_overlay(live, {"posture": "BUY_DIP"})["trade_view"].iloc[0] == "ACTIONABLE"
    assert apply_overlay(live, {"posture": "BUY_RIP", "entry_mode": "CASH"})["trade_view"].iloc[0] == "BEST OF WEAK TAPE"
    assert apply_overlay(live, {})["trade_view"].iloc[0] == "NO REGIME DATA"


# ------------------------------------------------------------------ dashboard metrics
def _fake_oos(n_days=60, n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2024-01-01", periods=n_days).repeat(n)
    p = rng.random(len(d))
    y = 0.02 * (p - 0.5) + rng.normal(0, 0.01, len(d))   # higher score -> higher label
    return pd.DataFrame({"date": d, "horizon": 5, "p_up": p, "y_mag": y, "y_ret": y, "y_mfe": np.abs(y) + .01,
                         "y_mae": -np.abs(y) - .01, "skew_score": 0.5 + 2 * p, "y_hit_up_20": 0.0, "y_hit_dn_20": 0.0})


def test_edge_timeseries_positive_when_score_works():
    from pipeline.evaluate import edge_timeseries
    e = edge_timeseries(_fake_oos())
    assert len(e) == 3 and (e["spread"] > 0).all() and set(e.columns) >= {"month", "spread", "n_days"}


def test_skew_buckets_carry_quantiles():
    from pipeline.evaluate import skew_buckets
    t = skew_buckets(_fake_oos())
    assert {"p10", "p25", "p50", "p75", "p90"} <= set(t.columns)
    assert (t["p10"] <= t["p25"]).all() and (t["p25"] <= t["p50"]).all() and (t["p75"] <= t["p90"]).all()


def test_waterfall_ends_add_up():
    """skew_base + sum of all SkewSHAP = raw q90 + q10 - 2 q50 for each row."""
    from pipeline.explain import skew_base, skew_shap
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((400, 3)), columns=list("abc"))
    q = QuantileSet([0.1, 0.5, 0.9]).fit(X, pd.Series(rng.normal(size=400) * (1 + X["a"])))
    raw = {k: q.models[k].predict(X) for k in (0.1, 0.5, 0.9)}
    total = skew_base(q) + skew_shap(q, X)["skew"].sum(axis=1).values
    assert np.allclose(total, raw[0.9] + raw[0.1] - 2 * raw[0.5], atol=1e-6)


# ------------------------------------------------------------------ kurtosis
def test_tail_weight_scale():
    from pipeline.skew import tail_weight
    rng = np.random.default_rng(0)
    q = lambda x: np.quantile(x, [.1, .25, .75, .9])
    assert abs(tail_weight(*q(rng.normal(size=200_000))) - 1.0) < 0.02        # normal = 1
    assert tail_weight(*q(rng.standard_t(3, size=200_000))) > 1.1             # fat tails > 1
    assert tail_weight(*q(rng.uniform(size=200_000))) < 0.9                   # thin tails < 1


def test_five_quantiles_ordered_and_tail_output(built):
    cfg, tg, F, T = built
    from pipeline.harness import prepare, fit_live, predict_bundle
    p = prepare(F, T, [5, 21])
    dic = DailyIC(p, model_feature_cols(p), 10)
    b = fit_live(dic.df, dic, Config(horizons=[5], max_survivors=10), 5)
    o = predict_bundle(b, dic.df[dic.df["date"] == dic.df["date"].max()], Config())
    q = o[["mag_q10", "mag_q25", "mag_q50", "mag_q75", "mag_q90"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (o["tail_weight"] > 0).all()
    assert "kurt60_pr" in F.columns


# ------------------------------------------------------------------ short plan history
class _PlanSess:
    """Fake Massive: 403 for grouped days before `first`, otherwise a tiny payload."""
    def __init__(self, first):
        self.first, self.calls = first, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        day = url.rsplit("/", 1)[-1]
        if day < self.first:
            return _Resp(403, {"status": "NOT_AUTHORIZED"})
        return _Resp(200, {"results": [{"T": "XOM", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "vw": 1, "n": 1}]})


def test_plan_probe_finds_earliest_day(tmp_path):
    cfg = Config(cache_dir=str(tmp_path))
    sess = _PlanSess("2024-11-18")
    cl = MassiveClient(cfg, api_key="k", session=sess)
    assert cl.earliest_available("2019-01-01", "2026-11-13") == "2024-11-18"
    assert sess.calls < 16                                   # binary search, not a crawl
    n = sess.calls
    cl.earliest_available("2019-01-01", "2026-11-13")        # cached
    assert sess.calls == n


def test_splice_scales_yfinance_to_massive():
    from pipeline.backfill import splice
    d = pd.bdate_range("2024-01-01", periods=40)
    first = d[20]
    px = np.linspace(50, 60, 40)
    massive = pd.DataFrame({"date": d[20:], "ticker": "X", "open": px[20:], "high": px[20:], "low": px[20:],
                            "close": px[20:], "volume": 1e6, "vwap": px[20:], "trades": 1})
    yfb = pd.DataFrame({"date": d, "ticker": "X", "open": px * .97, "high": px * .97, "low": px * .97,
                        "close": px * .97, "volume": 1e6})          # dividend-adjusted 3% lower
    out = splice(massive, yfb, first)
    assert out["date"].min() == d[0] and (out["date"] < first).sum() == 20
    r = out.set_index("date")["close"].pct_change().dropna()
    assert r.abs().max() < 0.01                                      # no jump at the join
    assert (out.loc[out["date"] < first, "source"] == "yfinance").all()


def test_load_market_backfills_short_history(monkeypatch):
    """End to end on fake data: Massive only has the last 300 sessions; the warm-up comes from yfinance."""
    from pipeline import universe, backfill as bf
    bars, spy, sh, sic, etfs = make_market(n_tickers=30, n_days=600, signal=0.0, seed=5, start="2022-01-03")
    dates = sorted(bars["date"].unique())
    first = pd.Timestamp(dates[300])

    class FakeClient:
        def stock_set(self):
            return pd.DataFrame({"ticker": bars["ticker"].unique(), "active": True})
        def grouped_range(self, start, end, tickers=None):
            allb = pd.concat([bars, spy, etfs], ignore_index=True)
            return allb[(allb["date"] >= first) & (allb["date"] <= pd.Timestamp(end))]
        def ticker_details(self, cand, asof=None):
            return pd.DataFrame({"ticker": list(cand), "shares": sh.reindex(list(cand)).values,
                                 "sic_code": sic.reindex(list(cand)).values})

    def fake_yf(names, start, end):
        allb = pd.concat([bars, spy, etfs], ignore_index=True)
        y = allb[allb["ticker"].isin(names) & (allb["date"] <= pd.Timestamp(end))].copy()
        for c in ("open", "high", "low", "close"):
            y[c] = y[c] * 0.98
        return y[["date", "ticker", "open", "high", "low", "close", "volume"]]

    monkeypatch.setattr(bf, "yf_download", fake_yf)
    cfg = Config(start=str(pd.Timestamp(dates[280]).date()), end=str(pd.Timestamp(dates[-1]).date()), backfill="auto")
    m = universe.load_market(cfg, FakeClient(), next_session=False, log=lambda *_: None)
    assert m["backfilled"] and m["first_massive"] == first
    b = m["bars"]
    assert b["date"].min() < first and (b.loc[b["date"] < first, "source"] == "yfinance").all()
    j = b[b["ticker"] == b["ticker"].iloc[0]].set_index("date")["close"].pct_change()
    assert abs(j.loc[first]) < 0.2                                  # spliced, not a 2% level jump artefact
    assert m["spy"]["date"].min() < first


def test_short_profile():
    cfg = Config()
    assert cfg.apply_profile(1200) == "standard" and cfg.test_sessions == 63
    cfg = Config()
    assert cfg.apply_profile(480) == "short"
    assert cfg.train_min_sessions == 192 and cfg.test_sessions == 42


def test_auto_start_is_massive_only(monkeypatch):
    """Default config: start = plan's earliest date + warm-up, no yfinance."""
    from pipeline import universe, backfill as bf
    bars, spy, sh, sic, etfs = make_market(n_tickers=20, n_days=500, signal=0.0, seed=6, start="2022-01-03")
    dates = sorted(bars["date"].unique())
    allb = pd.concat([bars, spy, etfs], ignore_index=True)

    class FakeClient:
        def earliest_available(self, lo, end):
            return str(pd.Timestamp(dates[0]).date())
        def stock_set(self):
            return pd.DataFrame({"ticker": bars["ticker"].unique(), "active": True})
        def grouped_range(self, start, end, tickers=None):
            return allb[(allb["date"] >= pd.Timestamp(start)) & (allb["date"] <= pd.Timestamp(end))]
        def ticker_details(self, cand, asof=None):
            return pd.DataFrame({"ticker": list(cand), "shares": sh.reindex(list(cand)).values,
                                 "sic_code": sic.reindex(list(cand)).values})

    monkeypatch.setattr(bf, "yf_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no yfinance")))
    cfg = Config(end=str(pd.Timestamp(dates[-1]).date()))
    assert cfg.start == "auto" and cfg.backfill == "none"
    m = universe.load_market(cfg, FakeClient(), next_session=False, log=lambda *_: None)
    assert not m["backfilled"] and "source" not in m["bars"]
    assert pd.Timestamp(cfg.start) > pd.Timestamp(dates[0]) + pd.tseries.offsets.BDay(cfg.warmup_sessions)


def test_out_of_plan_day_is_skipped(tmp_path):
    cfg = Config(cache_dir=str(tmp_path), probe_history=False)
    cl = MassiveClient(cfg, api_key="k", session=_PlanSess("2024-11-18"))
    assert cl.grouped_day("2024-11-15").empty
    assert not (tmp_path / "grouped" / "2024-11-15.parquet").exists()
