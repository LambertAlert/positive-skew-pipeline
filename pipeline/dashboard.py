"""
pipeline/dashboard.py
Builds dashboard.html from the run's own outputs (nothing hard-coded). War Room styling.

Every panel carries two reads, toggled in the header:
  Trader          what it means for picking and managing trades
  Data scientist  what was computed, on what, and how to judge it
Several of the reads are generated from the run's numbers (spreads, coverage, gaps), so they
change with the data.

Themes (header switch, remembered per browser): Midnight (default), Graphite, Paper (light) and
War Room. Each theme's three data colours passed the colour-blind / contrast validator against its
own surface. Positive vs negative always uses the theme's first two data colours (warm vs cool);
green/red are kept for confidence status, always with a ✓/✕ label. The reads name colours from the
active theme.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ETF_NAMES = {"XLK": "Technology", "XLV": "Health care", "XLF": "Financials", "XLE": "Energy",
             "XLI": "Industrials", "XLY": "Consumer discretionary", "XLP": "Consumer staples",
             "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real estate", "XLC": "Communication",
             "SPY": "Unmapped (vs SPY)"}


def _records(df: pd.DataFrame, cols=None, round_=5):
    if df is None or len(df) == 0:
        return []
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    d = d.replace([np.inf, -np.inf], np.nan)
    return json.loads(d.to_json(orient="records", double_precision=round_, date_format="iso"))


def write_dashboard(out: Path, meta: dict, metrics: dict, live: pd.DataFrame, shap_res: dict, cfg) -> Path:
    out = Path(out)
    gates = pd.read_csv(out / "gates.csv") if (out / "gates.csv").exists() else pd.DataFrame()
    surv_freq, gate_grid = {}, {}
    if len(gates):
        for h, g in gates.groupby("horizon"):
            s = g["survivors"].fillna("").str.split("|").explode()
            s = s[s != ""]
            surv_freq[int(h)] = (s.value_counts() / len(g)).head(15).round(3).to_dict()
            feats = (s.value_counts() / len(g)).head(20)
            sets = [set(str(x).split("|")) for x in g["survivors"].fillna("")]
            gate_grid[int(h)] = {"folds": [str(x)[:7] for x in g["test_start"]],
                                 "relaxed": [bool(x) for x in g["gate_relaxed"]],
                                 "rows": [{"feature": f, "share": round(float(v), 3),
                                           "cells": [int(f in st) for st in sets]} for f, v in feats.items()]}
        meta["relaxed_folds"] = int(gates["gate_relaxed"].sum())
        meta["n_folds"] = int(len(gates))
    dep = {}
    for h in cfg.horizons:
        f = out / f"shap_dependence_h{h}.csv"
        if f.exists():
            d = pd.read_csv(f)
            if len(d):
                top = d["feature"].iloc[0]
                d = d[d["feature"] == top]
                dep[int(h)] = {"feature": top, "points": _records(d.sample(min(600, len(d)), random_state=1),
                                                                   ["x", "shap", "post_climax", "hv_elevated"], 4)}
    live = live.copy()
    live["date"] = live["date"].astype(str).str[:10]
    live_cols = ["date", "ticker", "horizon", "sector", "bench_etf", "p_up", "p_linear", "p_rf", "p_gbm",
                 "p_bin_iso", "disagreement", "mag_q10", "mag_q25", "mag_q50", "mag_q75", "mag_q90", "tail_weight", "mfe_q50", "mfe_q90", "mae_q10",
                 "mae_q50", "skew_score", "path_skew", "bowley", "confidence", "positive_skew", "hv_regime",
                 "post_climax", "scr_premarket", "scr_leader", "scr_potent", "skew_drivers", "dir_drivers",
                 "trade_view", "regime_posture", "skew_base", "skew_total", "dir_base", "dir_total"]
    payload = {
        "meta": meta,
        "live": _records(live.sort_values(["horizon", "skew_score"], ascending=[True, False]), live_cols),
        "direction": _records(metrics.get("direction")),
        "calibration": _records(metrics.get("calibration")),
        "lift": _records(metrics.get("lift")),
        "quantiles": _records(metrics.get("quantiles")),
        "skew_buckets": _records(metrics.get("skew_buckets")),
        "flags": _records(metrics.get("flags")),
        "overlay": _records(metrics.get("overlay")),
        "ic_oos": _records(metrics.get("ic_oos")),
        "surv_freq": surv_freq,
        "gate_grid": gate_grid,
        "edge": _records(metrics.get("edge")),
        "tails": _records(metrics.get("tails")),
        "dependence": dep,
        "shap": {int(h): {k: (_records(v, round_=4) if k in ("interactions", "beeswarm")
                              else _records(v.reset_index().rename(columns={"index": "feature"})))
                          for k, v in r.items()} for h, r in shap_res.items()},
        "skew_threshold": cfg.skew_threshold,
        "etf_names": ETF_NAMES,
    }
    html = TEMPLATE.replace("/*__DATA__*/", "const D = " + json.dumps(payload, default=str) + ";")
    f = out / "dashboard.html"
    f.write_text(html)
    return f


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Positive-Skew Pipeline — War Room</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Orbitron:wght@500;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root,body[data-theme=midnight]{--bg:#0b0f17;--panel:#121722;--panel2:#171d2a;--line:#232b3b;--grid:#1a2130;--axis:#2c3547;
 --accent:#7fb2ff;--accent2:#9fb0c8;--flag:#7fb2ff;--text:#e6ebf2;--ink2:#aab4c3;--mute:#6f7b8e;
 --s1:#3b8fd9;--s2:#e0703a;--s3:#8b74e8;--pos:#3b8fd9;--neg:#e0703a;--neutral:#3a4457;--good:#3fbf7f;--bad:#e5534b;
 --hover:#161c28;--selrow:#1b2333;--tipbg:#1a2030;--tipline:#2c3547;--topbg:rgba(11,15,23,.94);--onaccent:#0b0f17;
 --f-ui:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;--f-num:'JetBrains Mono',ui-monospace,monospace;--f-head:'Inter',system-ui,sans-serif;--head-track:.02em;--head-case:none}
body[data-theme=graphite]{--bg:#101114;--panel:#17191e;--panel2:#1c1f25;--line:#2a2d35;--grid:#202329;--axis:#33363f;
 --accent:#e8e8ea;--accent2:#a3a6ad;--flag:#4a90e2;--text:#ececee;--ink2:#b4b7be;--mute:#777b84;
 --s1:#4a90e2;--s2:#d9682f;--s3:#22a08a;--pos:#4a90e2;--neg:#d9682f;--neutral:#3a3d46;--good:#3fbf7f;--bad:#e5534b;
 --hover:#1d2026;--selrow:#23262e;--tipbg:#1f2228;--tipline:#33363f;--topbg:rgba(16,17,20,.94);--onaccent:#101114}
body[data-theme=paper]{--bg:#f4f3ef;--panel:#ffffff;--panel2:#f8f7f4;--line:#e3e1db;--grid:#eeece7;--axis:#cfccc4;
 --accent:#1d2a44;--accent2:#5b6475;--flag:#2563c9;--text:#1c2230;--ink2:#4a5263;--mute:#868c99;
 --s1:#2563c9;--s2:#d4622a;--s3:#0e8f7a;--pos:#2563c9;--neg:#d4622a;--neutral:#d9d6ce;--good:#16803c;--bad:#c0362c;
 --hover:#f3f1ec;--selrow:#e9eef9;--tipbg:#ffffff;--tipline:#d9d6ce;--topbg:rgba(244,243,239,.95);--onaccent:#ffffff}
body[data-theme=warroom]{--bg:#09090e;--panel:#111119;--panel2:#16161f;--line:#23232e;--grid:#1b1b25;--axis:#2a2a35;
 --accent:#f5a524;--accent2:#c8743a;--flag:#f5a524;--text:#ebe6dc;--ink2:#b9b2a5;--mute:#7f796d;
 --s1:#c97f17;--s2:#5b84d6;--s3:#2a9d7e;--pos:#c97f17;--neg:#5b84d6;--neutral:#3a3a46;--good:#4fbf8a;--bad:#e0605a;
 --hover:#181822;--selrow:#1e1e2a;--tipbg:#1d1d28;--tipline:#2c2c38;--topbg:rgba(9,9,14,.94);--onaccent:#111;
 --f-ui:'Fira Code',ui-monospace,monospace;--f-num:'Fira Code',ui-monospace,monospace;--f-head:Orbitron,sans-serif;--head-track:.1em;--head-case:uppercase}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:13.5px/1.55 var(--f-ui);transition:background .2s}
a{color:var(--accent)}
.top{position:sticky;top:0;z-index:5;background:var(--topbg);backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}
.top .in{max-width:1400px;margin:0 auto;padding:14px 20px 10px}
h1{font:600 18px var(--f-head);color:var(--accent);margin:0;letter-spacing:var(--head-track)}
.sub{color:var(--mute);font-size:11.5px;margin-top:3px}
.warn{color:var(--bad)}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{background:transparent;color:var(--ink2);border:0;border-right:1px solid var(--line);padding:5px 11px;font:inherit;font-size:12px;cursor:pointer}
.seg button:last-child{border-right:0}.seg button.on{background:var(--accent);color:var(--onaccent);font-weight:500}
input#search{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:12px var(--f-num);width:110px;text-transform:uppercase}
input#search::placeholder{text-transform:none;color:var(--mute)}
th.s{cursor:pointer;user-select:none}th.s:hover{color:var(--text)}th .ar{color:var(--accent);margin-left:3px}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit;font-size:12px}
.lbl{color:var(--mute);font-size:11px;margin-right:2px}
.grp{display:inline-flex;align-items:center;gap:6px}
@media(max-width:700px){.top{position:static}.top .in,main{padding-left:12px;padding-right:12px}.card{padding:12px}
 .sect{flex-direction:column;gap:2px}.sgrid{grid-template-columns:none;grid-auto-flow:column;grid-auto-columns:82%;overflow-x:auto;scroll-snap-type:x mandatory;padding-bottom:4px}.scard{scroll-snap-align:start}.kpis{grid-template-columns:repeat(2,1fr)}.ch.tall{height:360px}.big{gap:12px}.big b{font-size:18px}}
.regime{margin-top:8px;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);font-size:12px;color:var(--ink2)}
.regime b{font-weight:500}
main{max-width:1400px;margin:0 auto;padding:6px 20px 40px}
.sect{margin:26px 0 10px;display:flex;align-items:baseline;gap:12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sect h2{font:600 13px var(--f-head);color:var(--accent2);letter-spacing:.1em;margin:0;text-transform:uppercase}
.sect span{color:var(--mute);font-size:11.5px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(12,1fr)}
.c12{grid-column:span 12}.c8{grid-column:span 8}.c6{grid-column:span 6}.c4{grid-column:span 4}
@media(max-width:1100px){.c8,.c6,.c4{grid-column:span 12}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h3{font:600 13px var(--f-ui);color:var(--text);margin:0 0 10px}
.reads{display:grid;gap:8px;margin-top:12px}
.read{border-left:2px solid var(--line);padding:2px 0 2px 10px;color:var(--ink2);font-size:12px}
.read .who{display:block;font:600 10.5px var(--f-ui);letter-spacing:.1em;text-transform:uppercase;margin-bottom:2px}
.read.tr{border-color:var(--s1)}.read.tr .who{color:var(--s1)}
.read.ds{border-color:var(--s3)}.read.ds .who{color:var(--s3)}
body.m-tr .read.ds,body.m-ds .read.tr{display:none}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{display:block;font:500 20px var(--f-num);color:var(--text)}
.kpi span{color:var(--mute);font-size:11px}
.tw{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:8px;margin-top:12px}
table{width:100%;border-collapse:collapse;font:12px var(--f-num);font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--mute);font-weight:400;position:sticky;top:0;background:var(--panel2);z-index:1}
td.l,th.l{text-align:left}tbody tr:hover{background:var(--hover)}tr.sel{background:var(--selrow)!important}
#live tbody tr{cursor:pointer}
.tk{color:var(--text);font-weight:500}.tk.f{color:var(--flag)}
.etf{color:var(--mute);font-size:11px;margin-left:4px}
.chip{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:10.5px;color:var(--ink2);margin-left:3px}
.st{font-size:11.5px}.st.good{color:var(--good)}.st.bad{color:var(--bad)}.st.mid{color:var(--accent)}
.rng{position:relative;width:150px;height:14px;display:inline-block;vertical-align:middle}
.rng .z{position:absolute;top:0;bottom:0;width:1px;background:var(--mute);opacity:.6}
.rng .lo{position:absolute;top:4px;height:6px;background:var(--neg);border-radius:3px 0 0 3px}
.rng .hi{position:absolute;top:4px;height:6px;background:var(--pos);border-radius:0 3px 3px 0}
.rng .m{position:absolute;top:1px;width:2px;height:12px;background:var(--text)}
.rng .iq{position:absolute;top:2px;height:10px;border:1.5px solid var(--text);border-radius:2px;opacity:.6}
.bigrng .iq{position:absolute;top:10px;height:20px;border:2px solid var(--text);border-radius:3px;opacity:.7}
.tl{font-size:10.5px;border-radius:4px;padding:0 5px;margin-left:5px;border:1px solid var(--line);color:var(--mute)}
.tl.fat{color:var(--s3);border-color:var(--s3)}.tl.thin{color:var(--ink2)}
.ch.third{height:96px}
.mini{display:inline-block;height:6px;border-radius:3px;background:var(--pos);vertical-align:middle;margin-right:6px}
.sgrid{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}
.scard{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.scard h4{margin:0 0 6px;font:600 12px var(--f-ui);letter-spacing:.04em;color:var(--text)}
.scard h4 small{color:var(--mute);font:400 11px var(--f-ui);letter-spacing:0;margin-left:6px}
.shd{display:grid;grid-template-columns:1fr 64px 52px;color:var(--mute);font-size:10.5px;padding-bottom:3px;border-bottom:1px solid var(--grid)}
.srow{display:grid;grid-template-columns:1fr 64px 52px;align-items:center;padding:4px 0;border-bottom:1px solid var(--grid);font-size:12px;cursor:pointer}
.srow span:nth-child(n+2),.shd span:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}
.srow:last-child{border-bottom:0}.srow:hover .tk{color:var(--accent)}
.big{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;margin-bottom:8px}
.big b{font:500 22px var(--f-num)}.big span{color:var(--mute);font-size:11.5px;display:block}
.bigrng{position:relative;height:50px;margin:14px 0 4px}
.bigrng .axis{position:absolute;left:0;right:0;top:30px;height:1px;background:var(--line)}
.bigrng .seg2{position:absolute;top:14px;height:12px}
.bigrng .tick{position:absolute;top:8px;width:2px;height:24px;background:var(--text)}
.bigrng .lab{position:absolute;top:34px;font-size:10.5px;color:var(--mute);transform:translateX(-50%)}
.bigrng .z{position:absolute;top:6px;height:30px;width:1px;background:var(--mute)}
.ch{position:relative;height:250px}.ch.tall{height:320px}
.note{color:var(--mute);font-size:11px;margin-top:6px}
.empty{color:var(--mute);padding:20px 0;text-align:center}
.ramp{display:flex;align-items:center;gap:8px;color:var(--mute);font-size:11px;margin-top:6px}
.ramp i{display:inline-block;width:120px;height:8px;border-radius:4px}
.gate{overflow-x:auto}.gate table{font-size:11px;width:auto}.gate .cell{min-width:22px}.gate td,.gate th{padding:2px 2px;border:0}
.gate th{position:static;background:none;font-size:10px;color:var(--mute);writing-mode:vertical-rl;transform:rotate(180deg);height:58px;text-align:left;font-weight:400}
.gate td.f{text-align:left;padding-right:8px;color:var(--ink2);white-space:nowrap}.gate td.sh{color:var(--mute);padding-left:8px}
.gate .cell{width:100%;min-width:12px;height:14px;border-radius:3px;background:var(--grid)}.gate .cell.on{background:var(--s1)}
.tools{display:flex;align-items:center;gap:6px;margin-top:12px}
.ch.half{height:125px}
.hz{display:inline-flex;gap:3px;vertical-align:middle}.hz i{display:inline-block;width:11px;height:11px;border-radius:3px;border:1.5px solid var(--axis)}
.hz i.on{background:var(--s1);border-color:var(--s1)}.hz i.lean{background:transparent;border-color:var(--s1)}.hz i.cur{outline:1.5px solid var(--text);outline-offset:1px}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--panel2);border:1px solid var(--line);color:var(--text);padding:8px 14px;border-radius:8px;font-size:12px;opacity:0;transition:opacity .2s;pointer-events:none;z-index:9}
.toast.on{opacity:1}
.toggle{background:transparent;border:1px solid var(--line);color:var(--ink2);border-radius:5px;padding:3px 8px;font:inherit;font-size:11px;cursor:pointer;margin-left:6px}
</style></head>
<body class="m-both" data-theme="midnight">
<div class="top"><div class="in">
 <h1>POSITIVE-SKEW PATH PIPELINE</h1>
 <div class="sub" id="meta"></div>
 <div class="bar">
  <span class="grp"><span class="lbl">Horizon</span><div class="seg" id="htabs"></div></span>
  <span class="grp"><span class="lbl">Sector</span><select id="sector"></select></span>
  <span class="grp"><span class="lbl">Show</span><div class="seg" id="flagseg"><button data-v="1" class="on">Flagged</button><button data-v="0">All names</button></div></span>
  <span class="grp"><span class="lbl">Ticker</span><input id="search" placeholder="search…" autocomplete="off" spellcheck="false"></span>
  <span class="grp"><span class="lbl">Reads</span><div class="seg" id="modeseg"><button data-v="both" class="on">Both</button><button data-v="tr">Trader</button><button data-v="ds">Data scientist</button></div></span>
  <span class="grp"><span class="lbl">Theme</span><div class="seg" id="themeseg"><button data-v="midnight">Midnight</button><button data-v="graphite">Graphite</button><button data-v="paper">Paper</button><button data-v="warroom">War Room</button></div></span>
 </div>
 <div class="regime" id="regime"></div>
</div></div>
<main>
 <div class="sect"><h2>Today's shortlist</h2><span id="asof"></span></div>
 <div class="grid">
  <div class="card c12">
   <div class="kpis" id="kpis"></div>
   <div class="tools"><span class="note" id="rowcount"></span><span style="flex:1"></span>
    <button class="toggle" id="copyt" title="copy the tickers in the table, comma-separated">Copy tickers</button>
    <button class="toggle" id="dlcsv" title="download the table as CSV">Download CSV</button></div>
   <div class="tw" style="margin-top:6px"><table id="live"></table></div>
   <div class="reads" id="r_live"></div>
  </div>
  <div class="card c6"><h3>Upside vs downside map <span class="note">click a dot</span></h3><div class="ch tall"><canvas id="c_map"></canvas></div><div class="reads" id="r_map"></div></div>
  <div class="card c6"><h3>Sector strip · every name by P(beat) <span class="note">click a dot</span></h3><div class="ch tall"><canvas id="c_strip"></canvas></div><div class="reads" id="r_strip"></div></div>
  <div class="card c12"><h3>Best in each sector <span class="note" style="margin-left:6px">top 3 by P(beat) within each benchmark ETF · click a name</span></h3>
   <div class="sgrid" id="sectors"></div><div class="reads" id="r_sect"></div></div>
 </div>

 <div class="sect"><h2>Name drill-down</h2><span>select a row, dot or name above</span></div>
 <div class="grid">
  <div class="card c12"><div id="dhead"></div><div class="bigrng" id="drng"></div><div class="reads" id="r_drng"></div></div>
  <div class="card c6"><h3>Range by holding period</h3><div class="ch"><canvas id="c_fan"></canvas></div><div class="reads" id="r_fan"></div></div>
  <div class="card c6"><h3>Skew, P(beat) and tails by holding period</h3><div class="ch third"><canvas id="c_hsk"></canvas></div><div class="ch third"><canvas id="c_hp"></canvas></div><div class="ch third"><canvas id="c_htl"></canvas></div><div class="reads" id="r_hz"></div></div>
  <div class="card c6"><h3>Why the skew · SkewSHAP waterfall</h3><div class="ch tall"><canvas id="c_local"></canvas></div><div class="reads" id="r_local"></div></div>
  <div class="card c6"><h3>Why the direction · GBM SHAP waterfall</h3><div class="ch tall"><canvas id="c_local_dir"></canvas></div><div class="reads" id="r_local_dir"></div></div>
 </div>

 <div class="sect"><h2>Does it work?</h2><span>out-of-sample, purged walk-forward</span></div>
 <div class="grid">
  <div class="card c6"><h3>Decile lift · ranked by P(beat)</h3><div class="ch"><canvas id="c_lift"></canvas></div><div class="reads" id="r_lift"></div></div>
  <div class="card c6"><h3>Edge over time · top minus bottom decile, by month</h3><div class="ch"><canvas id="c_edge"></canvas></div><div class="reads" id="r_edge"></div></div>
  <div class="card c6"><h3>Calibration · predicted vs realized</h3><div class="ch"><canvas id="c_cal"></canvas></div><div class="reads" id="r_cal"></div></div>
  <div class="card c6"><h3>Range honesty · target vs realized coverage</h3><div class="ch"><canvas id="c_cov"></canvas></div><div class="reads" id="r_cov"></div></div>
  <div class="card c6"><h3>Skew buckets · best move ÷ worst move</h3><div class="ch"><canvas id="c_sk1"></canvas></div><div class="reads" id="r_sk1"></div></div>
  <div class="card c6"><h3>Skew buckets · distribution of outcomes</h3><div class="ch"><canvas id="c_sk2"></canvas></div><div class="reads" id="r_sk2"></div></div>
  <div class="card c6"><h3>Flagged vs not · hold-if-still-strong</h3><div class="tw" style="max-height:none"><table id="t_flags"></table></div><div class="tw" style="max-height:none"><table id="t_overlay"></table></div><div class="reads" id="r_flags"></div></div>
  <div class="card c6"><h3>Score IC · daily, out-of-sample</h3><div class="tw" style="max-height:none"><table id="t_ic"></table></div><div class="reads" id="r_ic"></div></div>
  <div class="card c12"><h3>Tails (kurtosis) · predicted vs realized</h3><div class="grid" style="gap:14px"><div class="c6"><div class="ch"><canvas id="c_tail"></canvas></div></div><div class="c6"><div class="ch"><canvas id="c_big"></canvas></div></div></div><div class="reads" id="r_tail"></div></div>
  <div class="card c12"><h3>Direction scorecard <button class="toggle" id="allmodels">show all models</button></h3><div class="tw"><table id="t_dir"></table></div><div class="reads" id="r_dir"></div></div>
 </div>

 <div class="sect"><h2>What drives it</h2><span>live model and the in-fold IC gate</span></div>
 <div class="grid">
  <div class="card c6"><h3>SkewSHAP beeswarm · every dot is a name</h3><div class="ch tall"><canvas id="c_skshap"></canvas></div><div class="ramp" id="ramp"></div><div class="reads" id="r_skshap"></div></div>
  <div class="card c6"><h3>Global direction SHAP · GBM</h3><div class="ch tall"><canvas id="c_gbshap"></canvas></div><div class="reads" id="r_gbshap"></div></div>
  <div class="card c6"><h3 id="deptitle">Dependence · top skew driver</h3><div class="ch"><canvas id="c_dep"></canvas></div><div class="reads" id="r_dep"></div></div>
  <div class="card c6"><h3>IC gate · which features passed, fold by fold</h3><div id="gate" class="gate"></div><div class="reads" id="r_surv"></div></div>
 </div>
</main>
<div class="toast" id="toast"></div>
<script>
/*__DATA__*/
let S1,S2,S3,MUTE,GRID,INK,INK2,AXIS,NEU,PANEL,yGrid,xNo;
function applyTheme(name){
 document.body.dataset.theme=name;const cs=getComputedStyle(document.body),v=k=>cs.getPropertyValue(k).trim();
 S1=v("--s1");S2=v("--s2");S3=v("--s3");MUTE=v("--mute");GRID=v("--grid");INK=v("--text");INK2=v("--ink2");AXIS=v("--axis");NEU=v("--neutral");PANEL=v("--panel");
 Chart.defaults.color=MUTE;Chart.defaults.font.family=v("--f-num");Chart.defaults.font.size=11;Chart.defaults.borderColor=GRID;
 const tt=Chart.defaults.plugins.tooltip;tt.backgroundColor=v("--tipbg");tt.borderColor=v("--tipline");tt.borderWidth=1;tt.titleColor=INK;tt.bodyColor=INK2;tt.padding=8;
 yGrid={grid:{color:GRID},border:{display:false}};xNo={grid:{display:false},border:{color:AXIS}};
 [...document.querySelectorAll("#themeseg button")].forEach(b=>b.classList.toggle("on",b.dataset.v===name));
 [C1N,C2N,C3N]=CNAMES[name]||CNAMES.midnight;
 try{localStorage.setItem("psp_theme",name)}catch(e){}
}
const CNAMES={midnight:["blue","orange","violet"],graphite:["blue","orange","teal"],paper:["blue","orange","teal"],warroom:["amber","blue","teal"]};
let C1N="blue",C2N="orange",C3N="violet";
let THEME="midnight";try{THEME=localStorage.getItem("psp_theme")||"midnight"}catch(e){}
applyTheme(THEME);
const charts={};function mk(id,cfg){if(charts[id])charts[id].destroy();const el=document.getElementById(id);if(!el)return;charts[id]=new Chart(el,cfg);}
const f=(x,d=3)=>x==null||isNaN(x)?"—":Number(x).toFixed(d), pc=(x,d=1)=>x==null||isNaN(x)?"—":(x>0?"+":"")+(100*x).toFixed(d)+"%", pp=(x,d=1)=>x==null||isNaN(x)?"—":(100*x).toFixed(d)+"%";
const m=D.meta,H=m.horizons,REL=m.target_mode!=="absolute",R=m.regime||{},EN=D.etf_names||{};
let h=H.includes(5)?5:H[0],onlyFlag=true,sel=null,sector="ALL",allModels=false,query="",sortK=null,sortDir=-1;
const MAGW=REL?"vs its sector ETF":"absolute", PNAME=REL?"P(beat)":"P(up)";

document.getElementById("meta").innerHTML=`${m.source==="synthetic"?'<span class="warn">SYNTHETIC DATA — layout only, not market evidence</span> · ':''}Massive · ${m.start} → ${m.end} · universe: Luk ${m.screens.join(" ∪ ")}, close &gt; $${m.price_floor} · ${m.n_names.toLocaleString()} names (~${Math.round(m.avg_universe)}/day) · target: ${REL?"beat sector ETF":"absolute return"} · ${m.n_folds??0} folds${m.relaxed_folds?` (${m.relaxed_folds} relaxed gate)`:""} · built ${m.generated}`;
const pcol=p=>p==="AVOID"?"var(--bad)":(p==="NEUTRAL"?"var(--accent)":"var(--good)");
document.getElementById("regime").innerHTML=R.stale?`<span class="warn">War Room regime snapshot is ${R.age_days} days old</span> (as of ${String(R.generated_at).slice(0,16)}), so the View column ignores it. Last posture: ${R.posture}.`:R.posture?`War Room posture <b style="color:${pcol(R.posture)}">${R.posture}</b>${R.posture_confidence!=null?` (${(100*R.posture_confidence).toFixed(0)}% conf.)`:""}${R.regime_score!=null?` · regime score ${f(R.regime_score,2)}`:""}${R.entry_mode?` · entry mode <b>${R.entry_mode}</b>`:""}${R.current_state_name?` · ${R.current_state_name}`:""}${R.generated_at?` · as of ${String(R.generated_at).slice(0,16)}`:""} — flagged names are labelled for this tape in the View column.${R.entry_mode_reason?`<br><i>${R.entry_mode_reason}</i>`:""}`
 :`<span class="warn">No War Room regime snapshot</span> — rankings show which names should beat their sector, not whether to be active. Run with <code>--regime-path</code> to add the posture.`;

function seg(id,on){const el=document.getElementById(id);[...el.children].forEach(b=>b.onclick=()=>{[...el.children].forEach(x=>x.classList.remove("on"));b.classList.add("on");on(b.dataset.v)})}
const ht=document.getElementById("htabs");H.forEach(x=>{const b=document.createElement("button");b.textContent=x+"D";b.dataset.v=x;if(x===h)b.classList.add("on");ht.appendChild(b)});
seg("htabs",v=>{h=+v;sel=null;render()});seg("flagseg",v=>{onlyFlag=v==="1";render()});
seg("modeseg",v=>{document.body.className="m-"+v});
seg("themeseg",v=>{applyTheme(v);render()});
document.getElementById("search").oninput=e=>{query=e.target.value.trim().toUpperCase();sel=null;render()};
const etfs=[...new Set(D.live.map(r=>r.bench_etf||"SPY"))].sort();
const ss=document.getElementById("sector");ss.innerHTML=`<option value="ALL">All sectors</option>`+etfs.map(e=>`<option value="${e}">${e} · ${EN[e]||e}</option>`).join("");
ss.onchange=()=>{sector=ss.value;sel=null;render()};
document.getElementById("copyt").onclick=()=>copyText(CUR.map(r=>r.ticker).join(","));
document.getElementById("dlcsv").onclick=()=>{const k=["ticker","bench_etf","horizon","p_up","skew_score","tail_weight","mag_q10","mag_q25","mag_q50","mag_q75","mag_q90","mfe_q50","mae_q50","confidence","positive_skew","trade_view"];
 const csv=[k.join(",")+",horizons_flagged",...CUR.map(r=>k.map(c=>r[c]??"").join(",")+","+nFlag(r.ticker))].join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download=`positive_skew_${m.end}_${h}D${sector!=="ALL"?"_"+sector:""}.csv`;a.click();toast(`Saved ${CUR.length} rows`)};
document.getElementById("allmodels").onclick=e=>{allModels=!allModels;e.target.textContent=allModels?"ensemble only":"show all models";render()};

function reads(id,tr,ds){document.getElementById(id).innerHTML=`<div class="read tr"><span class="who">Trader</span>${tr}</div><div class="read ds"><span class="who">Data scientist</span>${ds}</div>`}
function table(id,rows,cols,empty,sortable){const t=document.getElementById(id);if(!rows.length){t.innerHTML=`<tbody><tr><td class="l empty">${empty||"No rows yet — run the walk-forward (without --live-only)."}</td></tr></tbody>`;return}
 t.innerHTML="<thead><tr>"+cols.map((c,i)=>`<th class="${c[2]||''}${sortable&&c[3]?' s':''}" data-i="${i}" title="${sortable&&c[3]?'sort':''}">${c[0]}${sortable&&sortK===i?`<span class="ar">${sortDir<0?'▼':'▲'}</span>`:''}</th>`).join("")+"</tr></thead><tbody>"+rows.map(r=>"<tr>"+cols.map(c=>`<td class="${c[2]||''}">${c[1](r)}</td>`).join("")+"</tr>").join("")+"</tbody>";}
const nm=x=>String(x).replace(/__x__/g," × ").replace(/_pr/g,"");
function hbars(id,labels,vals,signed,fmt,tick){labels=labels.map(nm);mk(id,{type:"bar",data:{labels,datasets:[{data:vals,backgroundColor:vals.map(v=>signed&&v<0?S2:S1),borderRadius:4,borderSkipped:"start",barPercentage:.72,categoryPercentage:.9}]},
 options:{indexAxis:"y",maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>fmt?fmt(c.raw):f(c.raw,4)}}},
 scales:{x:{...yGrid,ticks:{maxTicksLimit:5,callback:tick||(v=>v)}},y:{...xNo,ticks:{autoSkip:false,font:{size:10.5},color:INK2}}}}});}
const cap=x=>x[0].toUpperCase()+x.slice(1);
function hex2rgb(h){h=h.replace("#","");return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16))}
function lerpHex(a,b,t){const A=hex2rgb(a),B=hex2rgb(b);return "#"+A.map((v,i)=>Math.round(v+(B[i]-v)*t).toString(16).padStart(2,"0")).join("")}
function diverge(t){return t<.5?lerpHex(S2,NEU,t*2):lerpHex(NEU,S1,(t-.5)*2)}   // low -> grey -> high
function hash(str){let x=0;for(const c of str)x=(x*31+c.charCodeAt(0))%9973;return x/9973}
function pick(t){const r=D.live.find(r=>r.horizon===h&&r.ticker===t);if(!r)return;sel=r;drill();mark();document.getElementById("dhead").scrollIntoView({behavior:"smooth",block:"start"})}
// draws a short text label beside chosen points (selective direct labels)
const pointLabels={id:"pointLabels",afterDatasetsDraw(c){const ctx=c.ctx,boxes=[];c.data.datasets.forEach((ds,di)=>{if(!ds.labelKey)return;const meta=c.getDatasetMeta(di);if(meta.hidden)return;
 ctx.save();ctx.font=`500 10.5px ${Chart.defaults.font.family}`;ctx.fillStyle=INK;ctx.textBaseline="middle";
 meta.data.forEach((pt,i)=>{const d=ds.data[i];if(!d||!d[ds.labelKey])return;const t=d[ds.labelKey],w=ctx.measureText(t).width,b={x:pt.x+7,y:pt.y-13,w,h:13};
  if(boxes.some(o=>b.x<o.x+o.w+3&&b.x+b.w+3>o.x&&b.y<o.y+o.h&&b.y+b.h>o.y))return;   // skip a label that would collide
  boxes.push(b);ctx.fillText(t,b.x,pt.y-7)});ctx.restore()})}};
function waterfall(id,base,items,total,k,unit,startLabel,endLabel,dp){
 const labels=[startLabel],data=[[0,base*k]],colors=[NEU],delta=[base*k],run=[base*k];let cum=base*k;
 items.forEach(([n,v])=>{v*=k;labels.push(nm(n));data.push([cum,cum+v]);colors.push(v>=0?S1:S2);delta.push(v);cum+=v;run.push(cum)});
 const other=total*k-cum;if(Math.abs(other)>1e-9){labels.push("all other features");data.push([cum,cum+other]);colors.push(other>=0?S1+"88":S2+"88");delta.push(other);cum+=other;run.push(cum)}
 labels.push(endLabel);data.push([0,total*k]);colors.push(INK2);delta.push(total*k);run.push(total*k);
 const last=labels.length-1;
 mk(id,{type:"bar",data:{labels,datasets:[{data,backgroundColor:colors,borderRadius:3,borderSkipped:false,barPercentage:.7,categoryPercentage:.9}]},
  options:{indexAxis:"y",maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>{const i=c.dataIndex;return i===0?`typical name: ${delta[0].toFixed(dp)} ${unit}`:i===last?`this name: ${delta[i].toFixed(dp)} ${unit}`:`${delta[i]>=0?"+":""}${delta[i].toFixed(dp)} ${unit} → running ${run[i].toFixed(dp)}`}}}},
  scales:{x:{...yGrid,ticks:{maxTicksLimit:6,callback:v=>+v.toFixed(3)},title:{display:true,text:unit==="pp"?"tail asymmetry, pp (upside width − downside width)":"log-odds of beating the sector"}},y:{...xNo,ticks:{autoSkip:false,font:{size:10.5},color:INK2}}}}});
}
const tlab=t=>t==null?"":t>=1.15?"fat":t<=0.9?"thin":"normal";
const tailCell=t=>t==null?"—":`${f(t,2)}<span class="tl ${tlab(t)}">${tlab(t)}</span>`;
const view=v=>!v?"":`<span class="st ${v.startsWith("ACTIONABLE")?"good":(v==="BEST OF WEAK TAPE"||v.startsWith("STALE")?"bad":"mid")}">${v==="BEST OF WEAK TAPE"?"▼ ":v.startsWith("ACTIONABLE")?"▲ ":"● "}${v}</span>`;
function rangeBar(r,lo,hi){const sc=x=>Math.max(0,Math.min(100,(x-lo)/(hi-lo)*100));const z=sc(0),a=sc(r.mag_q10),b=sc(r.mag_q50),c=sc(r.mag_q90);
 return `<span class="rng" title="q10 ${pc(r.mag_q10)} · q25 ${pc(r.mag_q25)} · q50 ${pc(r.mag_q50)} · q75 ${pc(r.mag_q75)} · q90 ${pc(r.mag_q90)}"><span class="z" style="left:${z}%"></span><span class="lo" style="left:${a}%;width:${Math.max(0,b-a)}%"></span><span class="hi" style="left:${b}%;width:${Math.max(0,c-b)}%"></span>${r.mag_q25!=null?`<span class="iq" style="left:${sc(r.mag_q25)}%;width:${Math.max(1,sc(r.mag_q75)-sc(r.mag_q25))}%"></span>`:""}<span class="m" style="left:calc(${b}% - 1px)"></span></span>`;}

const LIVECOLS=()=>[
  ["Ticker",r=>`<span class="tk ${r.positive_skew?'f':''}">${r.ticker}</span><span class="etf">${r.bench_etf||""}</span>`,"l",r=>r.ticker],
  [PNAME,r=>`<span class="mini" style="width:${Math.max(2,(r.p_up-0.4)*200)}px"></span>${f(r.p_up)}`,"",r=>r.p_up],
  ["SkewScore",r=>f(r.skew_score,2),"",r=>r.skew_score],
  ["Tails",r=>tailCell(r.tail_weight),"",r=>r.tail_weight],
  ["Horizons",r=>hzCell(r.ticker),"l",r=>nFlag(r.ticker)+r.p_up/10],
  [`Range ${MAGW} (q10 · q50 · q90)`,r=>rangeBar(r,RLO,RHI),"l",r=>r.mag_q90-r.mag_q10],
  ["q50",r=>pc(r.mag_q50),"",r=>r.mag_q50],["Best move q50",r=>pc(r.mfe_q50),"",r=>r.mfe_q50],["Worst move q50",r=>pc(r.mae_q50),"",r=>r.mae_q50],
  ["Confidence",r=>r.confidence==="low"?'<span class="st bad">✕ low</span>':'<span class="st good">✓ normal</span>',"",r=>r.confidence==="low"?0:1],
  ["View",r=>view(r.trade_view),"l",r=>r.trade_view||""],
  ["Setup",r=>`${r.hv_regime} vol${r.post_climax?'<span class="chip">post-climax</span>':''}${[r.scr_premarket?"PM":"",r.scr_leader?"LDR":"",r.scr_potent?"POT":""].filter(Boolean).map(s=>`<span class="chip">${s}</span>`).join("")}`,"l"]];
let RLO=-.02,RHI=.02;
const BYT={};D.live.forEach(r=>{(BYT[r.ticker]=BYT[r.ticker]||{})[r.horizon]=r});
const nFlag=t=>H.filter(x=>BYT[t]?.[x]?.positive_skew).length;
function hzCell(t){return `<span class="hz" title="${H.map(x=>{const r=BYT[t]?.[x];return `${x}D: ${r?(r.positive_skew?"flagged":r.p_up>=.5?`${PNAME} ${f(r.p_up,2)}, not flagged`:"not flagged"):"—"}`}).join(" · ")}">${H.map(x=>{const r=BYT[t]?.[x];return `<i class="${r?.positive_skew?"on":(r&&r.p_up>=.5?"lean":"")} ${x===h?"cur":""}"></i>`}).join("")}</span>`}
function toast(msg){const t=document.getElementById("toast");t.textContent=msg;t.classList.add("on");setTimeout(()=>t.classList.remove("on"),1800)}
function copyText(txt){const done=()=>toast(`Copied ${txt.split(",").filter(Boolean).length} tickers`);
 try{navigator.clipboard.writeText(txt).then(done,()=>fallback())}catch(e){fallback()}
 function fallback(){const ta=document.createElement("textarea");ta.value=txt;document.body.appendChild(ta);ta.select();try{document.execCommand("copy");done()}catch(e){toast("Copy blocked — select the text")}ta.remove()}}
let CUR=[];
function render(){
 const L=D.live.filter(r=>r.horizon===h&&(sector==="ALL"||(r.bench_etf||"SPY")===sector)), F=L.filter(r=>r.positive_skew);
 let base=query?L.filter(r=>r.ticker.includes(query)):(onlyFlag?F:L);
 const cols=LIVECOLS();
 if(sortK!=null&&cols[sortK][3]){const acc=cols[sortK][3];base=[...base].sort((a,b)=>{const x=acc(a),y=acc(b);return (x>y?1:x<y?-1:0)*sortDir})}
 const rows=base.slice(0,200);
 document.getElementById("asof").textContent=L.length?`scores for the session after ${m.end} · ${h}-session horizon`:"";
 const med=a=>{const s=a.filter(x=>x!=null).sort((x,y)=>x-y);return s.length?s[Math.floor(s.length/2)]:null};
 document.getElementById("kpis").innerHTML=[[L.length,"names in universe"+(sector!=="ALL"?` (${sector})`:"")],[F.length,"positive-skew flags"],
  [L.filter(r=>nFlag(r.ticker)>=3).length,"flagged on 3+ horizons"],[f(med(L.map(r=>r.skew_score)),2),"median SkewScore"],[L.filter(r=>r.confidence==="low").length,"low-confidence (unusual setups)"],
  [R.posture||"—","War Room posture"]].map(k=>`<div class="kpi"><b>${k[0]}</b><span>${k[1]}</span></div>`).join("");
 const all=rows.flatMap(r=>[r.mag_q10,r.mag_q90]).filter(x=>x!=null);RLO=Math.min(-0.02,...all);RHI=Math.max(0.02,...all);
 CUR=base;document.getElementById("rowcount").textContent=`${base.length} name${base.length===1?"":"s"}${base.length>rows.length?` (showing ${rows.length})`:""}`;
 table("live",rows,cols,query?`No ticker matching “${query}” in this horizon/sector.`:(onlyFlag?"No flagged names for this horizon/sector — switch Show to All names.":"No names."),true);
 [...document.querySelectorAll("#live th.s")].forEach(th=>th.onclick=()=>{const i=+th.dataset.i;if(sortK===i)sortDir=-sortDir;else{sortK=i;sortDir=i===0?1:-1}render()});
 [...document.querySelectorAll("#live tbody tr")].forEach((tr,i)=>tr.onclick=()=>{sel=rows[i];drill();mark()});
 if(!sel&&rows.length)sel=rows[0];mark();drill();
 const top=[...F].sort((a,b)=>b.skew_score-a.skew_score)[0];
 const tbh=(D.tails||[]).filter(r=>r.horizon===h),tailOK=tbh.length?tbh[tbh.length-1].realized_tail>tbh[0].realized_tail+.03:null;
 const tailNote=tailOK===false?` <b>Caution:</b> at this horizon the backtest hasn't confirmed that predicted tails show up afterwards (see "Tails · predicted vs realized"), so read the Tails column as the model's view, not a proven signal.`:tailOK?` The backtest confirms predicted tails at this horizon: names marked fat did have fatter tails afterwards.`:"";
 reads("r_live",
  `${F.length} of ${L.length} names are flagged for the next ${h} sessions: the model expects them to beat ${REL?"their sector ETF":"zero"} with a wider upside than downside${top?` (strongest asymmetry: <b>${top.ticker}</b>, SkewScore ${f(top.skew_score,2)})`:""}. The Horizons squares show the same name at 3/5/7/10/21 sessions (filled = flagged, ${C1N} ring = ${PNAME} ≥ 0.5 but not flagged, grey = neither; the outlined square is the horizon you're viewing); a name flagged across several horizons is a sturdier read than a one-horizon flag. Sort by that column to put them first. The range bar is the model's middle-80% outcome: ${C2N} = downside to the median, ${C1N} = upside, the tick is the median, the thin grey line is zero. The outlined box inside the bar is the middle half (q25–q75); Tails compares the full bar to that box — "fat" means outcomes cluster near the median with a few big moves, so size and stops need to allow for the occasional large one.${tailNote} Use it to build the chart-review list, then let the View column (the regime) decide whether to act and how big.`,
  `Live model fit on every labelled date (same purged calibration tail as the folds). ${PNAME} = isotonic-calibrated mean of logistic, RF and LightGBM for <code>${REL?"dir_xs_h":"dir_ret_h"}</code>${REL?" (beat the day's median universe member on return minus sector ETF; 50% base rate by construction)":""}. Range = LightGBM quantiles of <code>${REL?"rel_h":"ret_h"}</code>, rearranged so they can't cross. SkewScore = (q90−q50)/max(q50−q10, floor). Tails = quantile kurtosis [(q90−q10)/(q75−q25)]/1.90, so a normal distribution = 1.00 (fat ≥ 1.15, thin ≤ 0.90). Flag = SkewScore &gt; ${D.skew_threshold}, P ≥ 0.5, Isolation Forest not anomalous. Best/worst move = MFE/MAE medians, absolute.`);
 renderMap(L);renderStrip();renderSectors();renderEvidence();renderDrivers();
}
function renderMap(L){
 const pts=L.filter(r=>r.mag_q10!=null).map(r=>({x:(r.mag_q50-r.mag_q10)*100,y:(r.mag_q90-r.mag_q50)*100,t:r.ticker,p:r.p_up,f:r.positive_skew,sk:r.skew_score}));
 const fl=pts.filter(p=>p.f).sort((a,b)=>b.sk-a.sk),ot=pts.filter(p=>!p.f);fl.forEach((p,i)=>{if(i<6)p.lab=p.t});
 const mx=Math.max(1,...pts.map(p=>Math.max(p.x,p.y)))*1.05,lo=Math.max(0,Math.min(...pts.map(p=>Math.min(p.x,p.y)))*.85),rad=p=>Math.max(3,Math.min(9,3+(p.p-.45)*60));
 mk("c_map",{type:"scatter",plugins:[pointLabels],data:{datasets:[
  {label:"even (SkewScore 1.0)",type:"line",data:[{x:lo,y:lo},{x:mx,y:mx}],borderColor:NEU,borderWidth:1.5,pointRadius:0},
  {label:`flag line (${D.skew_threshold}×)`,type:"line",data:[{x:lo/D.skew_threshold,y:lo},{x:mx/D.skew_threshold,y:mx}],borderColor:MUTE,borderWidth:1,pointRadius:0},
  {label:"not flagged",data:ot,backgroundColor:MUTE+"70",borderColor:PANEL,borderWidth:1,pointRadius:ot.map(rad),pointHoverRadius:9},
  {label:"positive-skew flag",data:fl,labelKey:"lab",backgroundColor:S1,borderColor:PANEL,borderWidth:1.5,pointRadius:fl.map(rad),pointHoverRadius:10}]},
  options:{maintainAspectRatio:false,onClick:(e,els,c)=>{const el=els.find(x=>x.datasetIndex>=2);if(el)pick(c.data.datasets[el.datasetIndex].data[el.index].t)},
   plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{filter:i=>i.datasetIndex>=2,callbacks:{label:c=>`${c.raw.t}: up ${c.raw.y.toFixed(1)} pp / down ${c.raw.x.toFixed(1)} pp · skew ${f(c.raw.sk,2)} · ${PNAME} ${f(c.raw.p)}`}}},
   scales:{x:{...yGrid,min:lo,max:mx,title:{display:true,text:"downside width, q50 − q10 (pp)"}},y:{...yGrid,min:lo,max:mx,title:{display:true,text:"upside width, q90 − q50 (pp)"}}}}});
 const above=pts.filter(p=>p.y>p.x).length;
 reads("r_map",
  `Each dot is a name: how far the model thinks it could fall to its median (left–right) against how far it could rise above it (up–down). Above the diagonal = more upside than downside; above the steeper line = flagged. ${above} of ${pts.length} sit above even. Bigger dots have a higher ${PNAME}. The best setups are high and to the left: big upside, small downside.`,
  `x = q50 − q10, y = q90 − q50 of the predicted ${REL?"rel_h":"ret_h"} distribution, in percentage points; dot radius ∝ ${PNAME}. SkewScore = y / max(x, floor), so the iso-SkewScore lines through the origin are y = x and y = ${D.skew_threshold}x. The floor only matters near the y-axis. Points must also pass the ${PNAME} and confidence filters to be flagged.`);
}
function renderStrip(){
 const L=D.live.filter(r=>r.horizon===h),by={};L.forEach(r=>{const e=r.bench_etf||"SPY";(by[e]=by[e]||[]).push(r)});
 const keys=Object.keys(by).sort((a,b)=>by[a].length-by[b].length);  // largest at top
 const mk2=(r,i)=>({x:r.p_up,y:i+(hash(r.ticker)-.5)*.5,t:r.ticker,sk:r.skew_score,e:r.bench_etf||"SPY"});
 const pts=keys.flatMap((e,i)=>by[e].map(r=>({...mk2(r,i),f:r.positive_skew})));
 const meds=keys.map((e,i)=>{const v=by[e].map(r=>r.p_up).sort((a,b)=>a-b);return {x:v[Math.floor(v.length/2)],y:i,e,n:v.length}});
 const dim=sector!=="ALL";
 mk("c_strip",{type:"scatter",data:{datasets:[
  {label:"not flagged",data:pts.filter(p=>!p.f),backgroundColor:pts.filter(p=>!p.f).map(p=>dim&&p.e!==sector?MUTE+"30":MUTE+"90"),borderColor:PANEL,borderWidth:1,pointRadius:4,pointHoverRadius:7},
  {label:"positive-skew flag",data:pts.filter(p=>p.f),backgroundColor:pts.filter(p=>p.f).map(p=>dim&&p.e!==sector?S1+"40":S1),borderColor:PANEL,borderWidth:1.5,pointRadius:5.5,pointHoverRadius:8},
  {label:"sector median",data:meds,pointStyle:"line",rotation:90,pointRadius:11,borderColor:INK,borderWidth:2,backgroundColor:INK}]},
  options:{maintainAspectRatio:false,onClick:(e,els,c)=>{const el=els.find(x=>x.datasetIndex<2);if(el)pick(c.data.datasets[el.datasetIndex].data[el.index].t)},
   plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{callbacks:{label:c=>c.datasetIndex===2?`${c.raw.e} median ${f(c.raw.x)} (${c.raw.n} names)`:`${c.raw.t} (${c.raw.e}): ${PNAME} ${f(c.raw.x)} · skew ${f(c.raw.sk,2)}`}}},
   scales:{x:{...yGrid,title:{display:true,text:PNAME}},y:{...xNo,min:-.6,max:keys.length-.4,ticks:{stepSize:1,autoSkip:false,color:INK2,callback:v=>Number.isInteger(v)&&keys[v]?`${keys[v]}`:""}}}}});
 const best=meds.slice().sort((a,b)=>b.x-a.x)[0];
 reads("r_strip",
  `Every name in the universe, one row per sector, placed by how likely it is to beat that sector. The vertical tick is the sector's median. A sector whose dots sit to the right has more names the model likes — ${best?`<b>${best.e}</b> has the highest median today`:""}. Use it to see whether a sector has several good candidates or just one. Picking a sector in the header dims the rest.`,
  `Dots are jittered vertically by a hash of the ticker (stable between renders); x = calibrated ${PNAME}. Rows are benchmark ETFs, ordered by name count. Because ${PNAME} is relative to each row's own ETF, it measures within-sector opportunity, not which sector will win.`);
}
function mark(){const rows=[...document.querySelectorAll("#live tbody tr")];rows.forEach(tr=>tr.classList.toggle("sel",tr.querySelector(".tk")?.textContent===sel?.ticker))}

function renderSectors(){
 const L=D.live.filter(r=>r.horizon===h);const by={};L.forEach(r=>{const e=r.bench_etf||"SPY";(by[e]=by[e]||[]).push(r)});
 const keys=Object.keys(by).sort((a,b)=>by[b].length-by[a].length);
 document.getElementById("sectors").innerHTML=keys.map(e=>{const t=by[e].sort((a,b)=>b.p_up-a.p_up).slice(0,3);
  return `<div class="scard"><h4>${e}<small>${EN[e]||""} · ${by[e].length} names</small></h4><div class="shd"><span>ticker</span><span>${PNAME}</span><span>skew</span></div>`+t.map(r=>`<div class="srow" data-t="${r.ticker}"><span><span class="tk ${r.positive_skew?'f':''}">${r.ticker}</span>${r.positive_skew?' <span class="chip">flag</span>':''}</span><span>${f(r.p_up)}</span><span>${f(r.skew_score,2)}</span></div>`).join("")+`</div>`}).join("")||'<div class="empty">No live rows.</div>';
 [...document.querySelectorAll(".srow")].forEach(el=>el.onclick=()=>{sel=L.find(r=>r.ticker===el.dataset.t);drill();mark();document.getElementById("dhead").scrollIntoView({behavior:"smooth",block:"start"})});
 reads("r_sect",
  `If you already have the sector view — bullish semis, or defensive and want health care — start here. Each box is the names most likely to beat that sector's ETF. Highlighted tickers also carry the skew flag. Note that semis sit in XLK, so a semi here is ranked against tech broadly, and the Luk screens favour volatile names, so defensive sectors can be thin.`,
  `Groups = benchmark ETF from the SIC→SPDR map. Ranked within group by calibrated ${PNAME}. Because each label is already relative to its own ETF, scores are comparable across groups; within-group rank is the right read for sector selection. Groups are sorted by size.`);
}

function drill(){
 const head=document.getElementById("dhead"),rng=document.getElementById("drng");
 if(!sel){head.innerHTML='<div class="empty">No name selected.</div>';rng.innerHTML="";return}
 const r=sel,etf=r.bench_etf||"SPY";
 head.innerHTML=`<div class="big"><div><b style="color:${r.positive_skew?'var(--accent)':'var(--text)'}">${r.ticker}</b><span>${etf} · ${EN[etf]||""} · ${r.date} · ${h}D</span></div>
  <div><b>${f(r.p_up)}</b><span>${PNAME}</span></div><div><b>${f(r.skew_score,2)}</b><span>SkewScore</span></div><div><b>${pc(r.mag_q50)}</b><span>median ${MAGW}</span></div>
  <div><b>${f(r.tail_weight,2)}</b><span>tail weight (${tlab(r.tail_weight)})</span></div><div><b>${pc(r.mfe_q50)} / ${pc(r.mae_q50)}</b><span>typical best / worst move</span></div><div><b>${f(r.disagreement)}</b><span>model disagreement</span></div><div>${view(r.trade_view)||'<span class="note">not flagged</span>'}<span>${r.confidence==="low"?'<span class="st bad">✕ low confidence</span>':'<span class="st good">✓ normal confidence</span>'}</span></div></div>`;
 const lo=Math.min(-0.02,r.mag_q10*1.3,r.mae_q50*1.1),hi=Math.max(0.02,r.mag_q90*1.3,r.mfe_q50*1.1),sc=x=>(x-lo)/(hi-lo)*100;
 rng.innerHTML=`<div class="axis"></div><div class="z" style="left:${sc(0)}%"></div>
  <div class="seg2" style="left:${sc(r.mag_q10)}%;width:${sc(r.mag_q50)-sc(r.mag_q10)}%;background:${S2};border-radius:4px 0 0 4px"></div>
  <div class="seg2" style="left:${sc(r.mag_q50)}%;width:${sc(r.mag_q90)-sc(r.mag_q50)}%;background:${S1};border-radius:0 4px 4px 0"></div>
  ${r.mag_q25!=null?`<div class="iq" style="left:${sc(r.mag_q25)}%;width:${sc(r.mag_q75)-sc(r.mag_q25)}%"></div>`:""}
  <div class="tick" style="left:calc(${sc(r.mag_q50)}% - 1px)"></div>
  ${[["q10",r.mag_q10],["q50",r.mag_q50],["q90",r.mag_q90]].map(([k,v])=>`<div class="lab" style="left:${sc(v)}%">${k} ${pc(v)}</div>`).join("")}<div class="lab" style="left:${sc(0)}%;top:-8px">0</div>`;
 const up=r.mag_q90-r.mag_q50,dn=r.mag_q50-r.mag_q10;
 reads("r_drng",
  `${r.ticker}: in 8 of 10 outcomes the model sees the ${h}-session return ${MAGW} landing between ${pc(r.mag_q10)} and ${pc(r.mag_q90)}, median ${pc(r.mag_q50)}. The upside half is ${f(up/Math.max(dn,1e-9),1)}× the downside half. During the trade, a typical best move is ${pc(r.mfe_q50)} and a typical worst ${pc(r.mae_q50)} (absolute) — a stop inside that worst-move figure will get hit often. ${r.tail_weight!=null?`The outlined box is the middle half of outcomes (${pc(r.mag_q25)} to ${pc(r.mag_q75)}); tail weight ${f(r.tail_weight,2)} means the full range is ${f(r.tail_weight*1.9,2)}× that box, versus 1.90× for a normal distribution — ${r.tail_weight>=1.15?"fat tails: most outcomes are modest but the occasional move is large, so allow room in stops and size for it":r.tail_weight<=0.9?"thin tails: outcomes are spread fairly evenly, with fewer surprise moves":"roughly normal tails"}.`:""}`,
  `q10/q50/q90 from three LightGBM quantile regressions on <code>${REL?"rel_h":"ret_h"}</code>, rearranged. Up-width ${pp(up,2)}, down-width ${pp(dn,2)}; SkewScore applies a floor of max(0.2%, 10% of q10–q90 width) to the down-width. Disagreement = std of the four calibrated direction models (${f(r.p_linear)} / ${f(r.p_rf)} / ${f(r.p_gbm)} / ${f(r.p_bin_iso)}).`);
 renderHorizons(r);
 const sd=JSON.parse(r.skew_drivers||"[]"),dd=JSON.parse(r.dir_drivers||"[]");
 if(r.skew_base!=null)waterfall("c_local",r.skew_base,sd,r.skew_total,100,"pp","typical name","this name",3);
 else hbars("c_local",sd.map(x=>x[0]),sd.map(x=>x[1]*100),true,v=>(v>0?"+":"")+v.toFixed(3)+" pp",v=>v+" pp");
 if(r.dir_base!=null)waterfall("c_local_dir",r.dir_base,dd,r.dir_total,1,"log-odds","typical name",`this name (raw P ${f(1/(1+Math.exp(-r.dir_total)),3)})`,3);
 else hbars("c_local_dir",dd.map(x=>x[0]),dd.map(x=>x[1]),true,v=>(v>0?"+":"")+f(v,3)+" log-odds");
 const t1=sd[0],d1=dd[0];
 reads("r_local",
  t1?`The biggest reason for ${r.ticker}'s skew read is <b>${nm(t1[0])}</b>, which ${t1[1]>0?"widens the upside relative to the downside":"widens the downside relative to the upside"}. Read it top to bottom: it starts at the typical name's tilt, each bar adds or removes upside skew (${C1N} adds, ${C2N} removes), and the last bar is ${r.ticker}'s own tilt. If the big push comes from something you'd discount on the chart, discount the score.`:"No drivers stored for this row.",
  `Waterfall of local SkewSHAP = SHAP(q90)+SHAP(q10)−2·SHAP(q50), in pp of (upper − lower) tail width. Start = base value E[q90+q10−2q50] over the training data; end = this row's raw-model asymmetry (before rearrangement); "all other features" closes the gap left by the top 6. Feature names are peer percentiles within SIC group; × marks a gated interaction.`);
 reads("r_local_dir",
  d1?`For the direction call, <b>${nm(d1[0])}</b> matters most here and ${d1[1]>0?"raises":"lowers"} the odds that ${r.ticker} beats ${REL?"its sector":"zero"}. Same reading as the skew waterfall: from the typical name, step by step, to this one.`:"No drivers stored for this row.",
  `Waterfall of TreeSHAP on the live LightGBM classifier, in log-odds: base value (average log-odds) → top 6 contributions → remainder → this row's raw log-odds. Raw P is before isotonic calibration and before averaging with the other models, so it differs from the headline ${PNAME}.`);
}

function renderHorizons(r){
 const rows=H.map(x=>BYT[r.ticker]?.[x]).filter(Boolean),xs=rows.map(z=>z.horizon),pc100=v=>v==null?null:v*100;
 const pt=(k)=>rows.map(z=>({x:z.horizon,y:pc100(z[k])}));
 mk("c_fan",{type:"line",data:{datasets:[
   {label:"q10",data:pt("mag_q10"),borderColor:S2,borderWidth:1.5,pointRadius:0,fill:false},
   {label:"downside half (q10–q50)",data:pt("mag_q50"),borderColor:INK,borderWidth:2,backgroundColor:S2+"33",fill:"-1",pointRadius:rows.map(z=>z.horizon===h?5:3),pointBackgroundColor:INK},
   {label:"upside half (q50–q90)",data:pt("mag_q90"),borderColor:S1,borderWidth:1.5,backgroundColor:S1+"33",fill:"-1",pointRadius:0},
   {label:"typical best / worst move",data:pt("mfe_q50"),borderColor:MUTE,backgroundColor:MUTE,borderWidth:1.25,pointRadius:2,fill:false},
   {label:"typical worst move",data:pt("mae_q50"),borderColor:MUTE,backgroundColor:MUTE,borderWidth:1.25,pointRadius:2,fill:false}]},
  options:{maintainAspectRatio:false,interaction:{mode:"index",intersect:false},plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10,filter:l=>l.text!=="q10"&&l.text!=="typical worst move"}},
   tooltip:{callbacks:{title:c=>`${c[0].raw.x} sessions`,label:c=>{const n=["q10","median (q50)","q90","typical best move","typical worst move"][c.datasetIndex];return `${n}: ${c.raw.y>=0?"+":""}${c.raw.y.toFixed(2)}%`}}}},
   scales:{x:{type:"linear",...yGrid,min:Math.min(...xs)-1,max:Math.max(...xs)+1,ticks:{callback:v=>xs.includes(v)?v+"D":""},afterBuildTicks:a=>{a.ticks=xs.map(v=>({value:v}))},title:{display:true,text:"holding period (sessions)"}},
    y:{...yGrid,ticks:{callback:v=>v+"%"},grid:{color:c=>c.tick.value===0?AXIS:GRID}}}}});
 const hx={type:"linear",...xNo,min:Math.min(...xs)-1,max:Math.max(...xs)+1,ticks:{callback:v=>xs.includes(v)?v+"D":""},afterBuildTicks:a=>{a.ticks=xs.map(v=>({value:v}))}};
 const flagCol=rows.map(z=>z.positive_skew?S1:MUTE);
 mk("c_hsk",{type:"line",data:{datasets:[
   {label:"flag line",data:xs.map(x=>({x,y:D.skew_threshold})),borderColor:NEU,borderWidth:1.25,pointRadius:0},
   {label:"SkewScore",data:rows.map(z=>({x:z.horizon,y:z.skew_score})),borderColor:INK2,borderWidth:1.75,pointRadius:rows.map(z=>z.horizon===h?6:4.5),pointBackgroundColor:flagCol,pointBorderColor:PANEL,pointBorderWidth:1.5}]},
  options:{maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{filter:i=>i.datasetIndex===1,callbacks:{label:c=>`${c.raw.x}D SkewScore ${f(c.raw.y,2)}${rows[c.dataIndex].positive_skew?" · flagged":""}`}}},
   scales:{x:hx,y:{...yGrid,title:{display:true,text:"SkewScore"},suggestedMin:.8,suggestedMax:1.5}}}});
 mk("c_hp",{type:"line",data:{datasets:[
   {label:"coin flip",data:xs.map(x=>({x,y:.5})),borderColor:NEU,borderWidth:1.25,pointRadius:0},
   {label:PNAME,data:rows.map(z=>({x:z.horizon,y:z.p_up})),borderColor:INK2,borderWidth:1.75,pointRadius:rows.map(z=>z.horizon===h?6:4.5),pointBackgroundColor:flagCol,pointBorderColor:PANEL,pointBorderWidth:1.5}]},
  options:{maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{filter:i=>i.datasetIndex===1,callbacks:{label:c=>`${c.raw.x}D ${PNAME} ${f(c.raw.y,3)}`}}},
   scales:{x:hx,y:{...yGrid,title:{display:true,text:PNAME},suggestedMin:.45,suggestedMax:.6}}}});
 mk("c_htl",{type:"line",data:{datasets:[
   {label:"normal",data:xs.map(x=>({x,y:1})),borderColor:NEU,borderWidth:1.25,pointRadius:0},
   {label:"tail weight",data:rows.map(z=>({x:z.horizon,y:z.tail_weight})),borderColor:INK2,borderWidth:1.75,pointRadius:rows.map(z=>z.horizon===h?6:4.5),pointBackgroundColor:S3,pointBorderColor:PANEL,pointBorderWidth:1.5}]},
  options:{maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{filter:i=>i.datasetIndex===1,callbacks:{label:c=>`${c.raw.x}D tail weight ${f(c.raw.y,2)} (${tlab(c.raw.y)})`}}},
   scales:{x:{...hx,title:{display:true,text:"holding period (sessions)"}},y:{...yGrid,title:{display:true,text:"tails"},suggestedMin:.85,suggestedMax:1.25}}}});
 const bs=rows.slice().sort((a,b)=>b.skew_score-a.skew_score)[0],bp=rows.slice().sort((a,b)=>b.p_up-a.p_up)[0],fl=rows.filter(z=>z.positive_skew).map(z=>z.horizon+"D");
 const mfeRate=rows.map(z=>({h:z.horizon,v:z.mfe_q50/Math.max(Math.abs(z.mae_q50),1e-6)})).sort((a,b)=>b.v-a.v)[0];
 reads("r_fan",
  `How ${r.ticker}'s expected range changes with how long you hold. The shaded band is the middle 80% of outcomes ${MAGW} — ${C2N} below the median, ${C1N} above — and the grey lines are the typical best and worst move during the trade (absolute). Ranges widen with time; what you want is the ${C1N} half widening faster than the ${C2N} half. ${mfeRate?(mfeRate.v>=1?`The typical best move beats the typical worst by the most at ${mfeRate.h} sessions (${f(mfeRate.v,2)}×).`:`At no horizon does the typical best move exceed the typical worst (closest: ${mfeRate.h} sessions, ${f(mfeRate.v,2)}×), so expect to sit through drawdowns as large as the gains.`):""}`,
  `Each horizon has its own quantile models, fit independently, so the fan is five marginal distributions joined for display — not a simulated path, and the bands need not widen smoothly. x is spaced by actual sessions. The larger median dot is the horizon selected in the header.`);
 reads("r_hz",
  `${fl.length?`Flagged at ${fl.join(", ")}.`:"Not flagged at any horizon."} Skew peaks at ${bs.horizon} sessions (${f(bs.skew_score,2)}) and ${PNAME} at ${bp.horizon} (${f(bp.p_up,3)}). If both peak at the same horizon, that's the natural holding period to plan around; if they disagree, the skew horizon is the one that suits a positive-skew trade with a hold-if-still-strong exit. The bottom chart is tail weight: where it rises, that holding period carries more risk of a single large move.`,
  `SkewScore and calibrated ${PNAME} from each horizon's live models. Dots in ${C1N} mark horizons where every flag condition passed; the reference lines are the ${D.skew_threshold} SkewScore threshold, 0.5 and tail weight 1.0 (normal). Separate charts, not two axes, since the scales differ.`);
}

function renderEvidence(){
 const li=D.lift.filter(r=>r.horizon===h);
 mk("c_lift",{type:"bar",data:{labels:li.map(r=>"D"+r.dec),datasets:[
   {label:`mean return ${MAGW}`,data:li.map(r=>(REL?r.mean_mag:r.mean_ret)*100),backgroundColor:S1,borderRadius:4,borderSkipped:"start",barPercentage:.8,categoryPercentage:.8},
   {label:"mean absolute return",data:li.map(r=>r.mean_ret*100),backgroundColor:S2,borderRadius:4,borderSkipped:"start",barPercentage:.8,categoryPercentage:.8}]},
  options:{maintainAspectRatio:false,plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.raw.toFixed(2)}%`}}},scales:{x:xNo,y:{...yGrid,ticks:{callback:v=>v+"%"}}}}});
 const d1=li.find(r=>r.dec===1),d10=li.find(r=>r.dec===10);const mono=li.length?li.slice(1).filter((r,i)=>(REL?r.mean_mag:r.mean_ret)>=(REL?li[i].mean_mag:li[i].mean_ret)).length:0;
 reads("r_lift",
  d1&&d10?`Each day the universe is split into 10 buckets by ${PNAME}. Historically the top bucket returned ${pc(REL?d10.mean_mag:d10.mean_ret,2)} ${MAGW} over ${h} sessions and the bottom ${pc(REL?d1.mean_mag:d1.mean_ret,2)} — a spread of ${pp((REL?d10.mean_mag-d1.mean_mag:d10.mean_ret-d1.mean_ret),2)}. That's the edge from picking the top of the list instead of the bottom; it is per trade, before costs, and small, so use it to rank the review list.`:"Run the walk-forward to see this.",
  d1&&d10?`Per-date deciles of out-of-sample ${PNAME}, pooled across folds. ${mono} of 9 steps are non-decreasing — monotonicity matters more than the endpoints. ${C1N[0].toUpperCase()+C1N.slice(1)} is the trained label, ${C2N} the absolute return for context.`:"");
 const ed=(D.edge||[]).filter(r=>r.horizon===h);
 const roll=ed.map((r,i)=>{const w=ed.slice(Math.max(0,i-5),i+1);return w.reduce((a,x)=>a+x.spread,0)/w.length*100});
 mk("c_edge",{type:"bar",data:{labels:ed.map(r=>r.month),datasets:[
   {type:"line",label:"6-month average",data:roll,borderColor:INK2,borderWidth:2,pointRadius:0,tension:.25},
   {label:"monthly spread",data:ed.map(r=>r.spread*100),backgroundColor:ed.map(r=>r.spread<0?S2:S1),borderRadius:3,borderSkipped:"start",barPercentage:.85,categoryPercentage:.9}]},
  options:{maintainAspectRatio:false,plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.raw.toFixed(2)}%${c.datasetIndex===1?` (${ed[c.dataIndex].n_days} days)`:""}`}}},
  scales:{x:{...xNo,ticks:{maxRotation:0,autoSkipPadding:10}},y:{...yGrid,ticks:{callback:v=>v+"%"},grid:{color:c=>c.tick.value===0?AXIS:GRID}}}}});
 const posM=ed.filter(r=>r.spread>0).length;
 reads("r_edge",
  ed.length?`The lift chart's average can hide one lucky stretch, so this splits it by month: each bar is how much the top tenth beat the bottom tenth that month. ${posM} of ${ed.length} months were positive. What you want is mostly ${C1N} bars with the line staying above zero; long ${C2N} runs mean the edge comes and goes.`:"Run the walk-forward to see this.",
  ed.length?`Daily spread = mean ${REL?"rel_h":"ret_h"} of per-date decile 10 minus decile 1 on OOS ${PNAME}; bars are monthly means, the line a trailing 6-month mean. Labels overlap for h &gt; 1, so neighbouring months share information; treat the hit rate (${pp(posM/ed.length,0)}) as descriptive, not a significance test.`:"");
 const cal=D.calibration.filter(r=>r.horizon===h);
 mk("c_cal",{type:"scatter",data:{datasets:[
   {label:"perfect calibration",type:"line",data:[{x:.3,y:.3},{x:.7,y:.7}],borderColor:NEU,borderWidth:1.5,pointRadius:0},
   {label:"95% range",type:"line",data:cal.flatMap(r=>{const e=1.96*Math.sqrt(Math.max(r.hit*(1-r.hit),1e-4)/r.n);return [{x:r.mean_p,y:Math.max(.2,r.hit-e)},{x:r.mean_p,y:Math.min(.8,r.hit+e)},{x:NaN,y:NaN}]}),borderColor:MUTE,borderWidth:1.5,pointRadius:0,spanGaps:false},
   {label:"bins",data:cal.map(r=>({x:r.mean_p,y:r.hit,n:r.n})),backgroundColor:S1,borderColor:PANEL,borderWidth:2,pointRadius:cal.map(r=>Math.min(14,4+Math.log10(r.n+1)*2)),pointHoverRadius:12}]},
  options:{maintainAspectRatio:false,plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{filter:i=>i.datasetIndex===2,callbacks:{label:c=>c.raw.n?`predicted ${f(c.raw.x)} → realized ${f(c.raw.y)} (n=${c.raw.n.toLocaleString()})`:""}}},
  scales:{x:{...yGrid,min:.2,max:.8,title:{display:true,text:`predicted ${PNAME}`}},y:{...yGrid,min:.2,max:.8,title:{display:true,text:"realized rate"}}}}});
 const big=cal.filter(r=>r.n>=500),gap=big.length?Math.max(...big.map(r=>Math.abs(r.hit-r.mean_p))):null;
 const dr=D.direction.find(r=>r.horizon===h&&r.regime==="all"&&r.model==="ensemble");
 reads("r_cal",
  `When the model says 55%, does it happen about 55% of the time? Dots on the line mean you can read ${PNAME} at face value. The whiskers show how far off a dot could be by luck alone: a dot whose whisker crosses the line isn't a real miss. Most names sit near 50%: the model rarely has a strong opinion, which is normal for stock selection.`,
  `Reliability diagram: bins of ensemble ${PNAME} vs realized ${REL?"dir_xs":"dir_ret"} rate; dot size ∝ log n; whiskers = ±1.96·√(p(1−p)/n), a binomial 95% interval. ${gap!=null?`Largest gap among bins with n ≥ 500: ${f(gap,3)}.`:""} ${dr?`Brier skill vs base rate ${pc(dr.brier_skill,2)}.`:""}`);
 const sb=D.skew_buckets.filter(r=>r.horizon===h);
 const sbl=sb.map(r=>r.bucket);
 mk("c_sk1",{type:"bar",data:{labels:sbl,datasets:[{label:"mean best move ÷ |mean worst move|",data:sb.map(r=>r.mfe_mae_ratio),base:1,backgroundColor:sb.map(r=>r.mfe_mae_ratio<1?S2:S1),borderRadius:4,borderSkipped:"start",barPercentage:.7}]},
  options:{maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`ratio ${f(c.raw,3)} · n=${sb[c.dataIndex].n.toLocaleString()}`}}},scales:{x:{...xNo,title:{display:true,text:"predicted SkewScore bucket"}},y:{...yGrid,beginAtZero:false,min:Math.min(.9,Math.floor((Math.min(...sb.map(r=>r.mfe_mae_ratio))-.03)*20)/20),suggestedMax:1.1,title:{display:true,text:"ratio (bars start at 1.0 = even)"}}}}});
 const hasQ=sb.length&&sb[0].p10!=null;
 if(hasQ)mk("c_sk2",{type:"bar",data:{labels:sbl,datasets:[
   {label:"10th–90th percentile",data:sb.map(r=>[r.p10*100,r.p90*100]),backgroundColor:MUTE+"90",grouped:false,barPercentage:.06,borderSkipped:false},
   {label:"middle half (25th–75th)",data:sb.map(r=>[r.p25*100,r.p75*100]),backgroundColor:S1+"40",borderColor:S1,borderWidth:1.5,grouped:false,barPercentage:.45,borderSkipped:false,borderRadius:3},
   {label:"median",type:"line",data:sb.map(r=>r.p50*100),showLine:false,pointStyle:"line",pointRadius:13,pointHoverRadius:13,borderColor:INK,borderWidth:2.5},
   {label:"mean",type:"line",data:sb.map(r=>(REL?r.mean_mag:r.mean_ret)*100),showLine:false,pointStyle:"circle",pointRadius:4.5,backgroundColor:S1,borderColor:PANEL,borderWidth:1.5}]},
  options:{maintainAspectRatio:false,plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{mode:"index",intersect:false,callbacks:{title:c=>`SkewScore ${c[0].label} · n=${sb[c[0].dataIndex].n.toLocaleString()}`,label:c=>{const r=sb[c.dataIndex];return c.datasetIndex===0?`p10 ${pc(r.p10,2)} · p90 ${pc(r.p90,2)}`:c.datasetIndex===1?`p25 ${pc(r.p25,2)} · p75 ${pc(r.p75,2)}`:c.datasetIndex===2?`median ${pc(r.p50,2)}`:`mean ${pc(REL?r.mean_mag:r.mean_ret,2)}`}}}},
  scales:{x:{...xNo,title:{display:true,text:"predicted SkewScore bucket"}},y:{...yGrid,ticks:{callback:v=>v+"%"},grid:{color:c=>c.tick.value===0?AXIS:GRID}}}}});
 const lowB=sb[0],hiB=sb.filter(r=>r.n>=200).slice(-1)[0];
 const verdict=lowB&&hiB?(hiB.mfe_mae_ratio>lowB.mfe_mae_ratio+0.03?`It does here: from ${f(lowB.mfe_mae_ratio,2)} in the lowest bucket to ${f(hiB.mfe_mae_ratio,2)} in the highest bucket with at least 200 names.`:`It doesn't clearly do that here (${f(lowB.mfe_mae_ratio,2)} → ${f(hiB.mfe_mae_ratio,2)}), so treat SkewScore with suspicion until it does.`):"";
 reads("r_sk1",
  `This is the test of the whole idea. Names with a higher predicted SkewScore should, after the fact, have run further in your favour than against you. A ratio above 1 means the average best move beat the average worst move. ${verdict}`,
  `OOS rows bucketed by predicted SkewScore; mean MFE / |mean MAE| per bucket (absolute path). Buckets with few rows are noisy — check n in the tooltip. Realized Bowley skew of the trained label is in the flags table below.`);
 reads("r_sk2",
  hasQ?`What actually happened in each bucket, return ${MAGW}. The box is the middle half of outcomes, the whisker the 10th–90th percentile, the line the median, the dot the average. Positive skew looks like a top whisker longer than the bottom one and a dot above the line. Skew without a positive average is a lottery ticket.`:"Re-run the walk-forward to add the outcome distribution.",
  `Realized <code>${REL?"rel_h":"ret_h"}</code> quantiles (p10/p25/p50/p75/p90) and mean per predicted-SkewScore bucket, OOS. Asymmetry shows as (p90 − p50) vs (p50 − p10) and mean − median; compare with the Bowley column in the flags table. Separate from the ratio chart rather than a second axis.`);
 table("t_flags",D.flags.filter(r=>r.horizon===h),[["Group",r=>r.group.replace(/_/g," "),"l"],["n",r=>r.n.toLocaleString()],[REL?"Vs sector":"Return",r=>pc(REL?r.mean_mag:r.mean_ret,2)],["Abs return",r=>pc(r.mean_ret,2)],["Hit",r=>pp(r.hit)],["Best÷worst",r=>f(r.mfe_mae_ratio,2)],["Bowley",r=>f(r.realized_bowley,3)],["Tails",r=>f(r.realized_tail,2)]]);
 table("t_overlay",D.overlay.filter(r=>r.horizon===h),[["Hold rule",r=>r.group.replace(/_/g," "),"l"],["n",r=>r.n.toLocaleString()],["Extended",r=>pp(r.share_extended)],["Fixed exit",r=>pc(r.fixed_ret,2)],["Hold if strong",r=>pc(r.overlay_ret,2)]]);
 const fl=D.flags.filter(r=>r.horizon===h),fp=fl.find(r=>r.group==="positive_skew"),fa=fl.find(r=>r.group==="all"),ov=D.overlay.find(r=>r.horizon===h&&r.group==="all_eligible");
 reads("r_flags",
  `${fp&&fa?`Historically the flagged names returned ${pc(REL?fp.mean_mag:fp.mean_ret,2)} ${MAGW} vs ${pc(REL?fa.mean_mag:fa.mean_ret,2)} for everything, with a best÷worst of ${f(fp.mfe_mae_ratio,2)} vs ${f(fa.mfe_mae_ratio,2)}.`:""} ${ov?`Holding winners another ${h} sessions when still above the 10-day SMA averaged ${pc(ov.overlay_ret,2)} vs ${pc(ov.fixed_ret,2)} for exiting on schedule.`:""}`,
  `Flag groups are mutually comparable OOS slices; low-confidence = Isolation Forest score above the training-fold 95th percentile. The hold rule is absolute return, decided at the exit close with data through that day, and excludes low-confidence rows.`);
 const q=D.quantiles.filter(r=>r.horizon===h);const qs=q.filter(r=>r.tau!=="q10-q90");const band=q.find(r=>r.tau==="q10-q90");
 const qrows=[...qs.map(r=>({lab:`${r.target==="mag"?(REL?"vs sector":"return"):r.target==="mfe"?"best move":"worst move"} q${Math.round(r.tau*100)}`,t:r.tau*100,v:r.coverage_below*100})),
   ...(band?[{lab:"inside q10–q90",t:80,v:band.coverage_below*100}]:[])];
 mk("c_cov",{type:"bar",data:{labels:qrows.map(r=>r.lab),datasets:[
   {label:"gap",data:qrows.map(r=>[r.t,r.v]),backgroundColor:NEU,grouped:false,barPercentage:.18,borderSkipped:false},
   {label:"target",type:"line",data:qrows.map(r=>r.t),showLine:false,pointRadius:6,pointHoverRadius:7,backgroundColor:PANEL,borderColor:MUTE,borderWidth:2},
   {label:"realized",type:"line",data:qrows.map(r=>r.v),showLine:false,pointRadius:6,pointHoverRadius:7,backgroundColor:S3,borderColor:PANEL,borderWidth:1.5}]},
  options:{indexAxis:"y",maintainAspectRatio:false,plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10,filter:l=>l.text!=="gap"}},tooltip:{filter:i=>i.datasetIndex>0,callbacks:{label:c=>`${c.dataset.label}: ${c.raw.toFixed(1)}%`}}},
  scales:{x:{...yGrid,min:0,max:100,ticks:{callback:v=>v+"%"},title:{display:true,text:"share of outcomes below the predicted quantile"}},y:{...xNo,ticks:{color:INK2,font:{size:10.5}}}}}});
 const mq=q.filter(r=>r.target==="mag"&&r.tau!=="q10-q90");
 reads("r_cov",
  band?`Are the ranges honest? About 80% of outcomes should land inside q10–q90; historically ${pp(band.coverage_below)} did. ${band.coverage_below<0.75?"The ranges run a bit narrow — real outcomes are wider than shown.":band.coverage_below>0.85?"The ranges run a bit wide.":"Close enough to trust the range bars."}`:"Run the walk-forward to see this.",
  `Dumbbell: hollow dot = target share below the quantile (tau), ${C3N} dot = realized share OOS, bar = the gap. The last row is the q10–q90 band against its 80% target. ${mq.length?`Pinball skill vs an expanding unconditional quantile: ${mq.map(r=>`q${Math.round(r.tau*100)} ${pc(r.pinball_skill,1)}`).join(", ")}.`:""}`);
 const tb=(D.tails||[]).filter(r=>r.horizon===h);
 if(tb.length){
  const lo=Math.min(.8,...tb.map(r=>Math.min(r.pred_tail,r.realized_tail)))-.03,hi=Math.max(1.2,...tb.map(r=>Math.max(r.pred_tail,r.realized_tail)))+.03;
  mk("c_tail",{type:"scatter",plugins:[pointLabels],data:{datasets:[
    {label:"predicted = realized",type:"line",data:[{x:lo,y:lo},{x:hi,y:hi}],borderColor:NEU,borderWidth:1.5,pointRadius:0},
    {label:"normal (1.0)",type:"line",data:[{x:lo,y:1},{x:hi,y:1}],borderColor:AXIS,borderWidth:1,pointRadius:0},
    {label:"tail-weight quintile",data:tb.map(r=>({x:r.pred_tail,y:r.realized_tail,q:"Q"+r.bucket,n:r.n})),labelKey:"q",backgroundColor:S3,borderColor:PANEL,borderWidth:1.5,pointRadius:7,pointHoverRadius:9}]},
   options:{maintainAspectRatio:false,plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{filter:i=>i.datasetIndex===2,callbacks:{label:c=>`${c.raw.q}: predicted ${f(c.raw.x,2)} → realized ${f(c.raw.y,2)} (n=${c.raw.n.toLocaleString()})`}}},
    scales:{x:{...yGrid,min:lo,max:hi,title:{display:true,text:"predicted tail weight (bucket mean)"}},y:{...yGrid,min:lo,max:hi,title:{display:true,text:"realized tail weight"}}}}});
  mk("c_big",{type:"bar",data:{labels:tb.map(r=>"Q"+r.bucket),datasets:[{label:"share of big moves",data:tb.map(r=>r.big_move_share*100),backgroundColor:S3,borderRadius:4,borderSkipped:"start",barPercentage:.6}]},
   options:{maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.raw.toFixed(1)}% of outcomes were big moves · n=${tb[c.dataIndex].n.toLocaleString()}`}}},
    scales:{x:{...xNo,title:{display:true,text:"predicted tail-weight quintile (Q5 = fattest)"}},y:{...yGrid,ticks:{callback:v=>v+"%"},title:{display:true,text:"big-move share"}}}}});
  const q1=tb[0],q5=tb[tb.length-1],rising=q5.realized_tail>q1.realized_tail+.03;
  reads("r_tail",
   `Kurtosis is about how often a name makes an unusually large move compared with its normal range. Each day names are split into five groups by predicted tail weight. Left: did the fattest-tailed group (Q5) really have fatter tails afterwards? ${rising?`Yes — realized tail weight rose from ${f(q1.realized_tail,2)} (Q1) to ${f(q5.realized_tail,2)} (Q5).`:`Not clearly here (${f(q1.realized_tail,2)} → ${f(q5.realized_tail,2)}), so treat the Tails column as descriptive.`} Right: the share of outcomes that were big moves, ${pp(q1.big_move_share)} in Q1 vs ${pp(q5.big_move_share)} in Q5. For positive-skew trades a fat right tail is the payoff you're after; fat tails with negative skew are the setups that gap against you.`,
   `Per-date quintiles of predicted quantile kurtosis T = [(q90−q10)/(q75−q25)]/1.90 (normal = 1). Left: bucket-mean predicted T vs realized T of <code>${REL?"rel_h":"ret_h"}</code> in that bucket; points on the diagonal = calibrated, above = tails underestimated. Right: share of outcomes more than 2 × IQR from the horizon median (pooled IQR). Quantile kurtosis ignores everything beyond the 10th/90th percentiles — robust, but blind to the extreme 1-in-20 moves; moment kurtosis of daily returns enters the model separately as the kurt60 feature.`);
 } else {["c_tail","c_big"].forEach(id=>{if(charts[id]){charts[id].destroy();delete charts[id]}});reads("r_tail","Re-run the walk-forward to add the tail check.","");}
 const dirRows=D.direction.filter(r=>r.horizon===h&&(allModels||r.model==="ensemble"));
 table("t_dir",dirRows,[["Model",r=>r.model,"l"],["HV regime",r=>r.regime,"l"],["n",r=>r.n.toLocaleString()],["AUC",r=>f(r.auc)],["AUC same-day",r=>f(r.auc_xs)],["Brier",r=>f(r.brier,4)],["Brier skill",r=>pc(r.brier_skill,2)],["Log loss",r=>f(r.logloss,4)]]);
 reads("r_dir",
  dr?`Scorecard for the direction call. AUC 0.50 is a coin flip; ${f(dr.auc,3)} here. In stock ranking, 0.52–0.55 applied across hundreds of names is a usable edge; above 0.60 is suspicious. Look at the HV-regime rows to see whether the edge lives in calm or volatile tapes.`:"Run the walk-forward to see this.",
  `Pooled AUC, per-date AUC (ranks names within a day), Brier and log loss, overall and by the stock's own HV20-percentile regime. ${REL?"With the relative label the base rate is 50% every day, so pooled and same-day AUC should agree.":"With the absolute label, market-wide days dominate pooled AUC; trust the same-day column."}`);
 const ic=D.ic_oos.filter(r=>r.horizon===h);
 const lab=y=>({y_mag:REL?"vs sector":"return",y_mfe:"best move",y_mae:"worst move"}[y]||y);
 table("t_ic",ic,[["Score",r=>r.score.replace("mag_q50","median").replace("p_up",PNAME).replace("_score","").replace("mfe_q50","best move q50").replace("mae_q50","worst move q50"),"l"],["vs",r=>lab(r.label),"l"],["IC",r=>`<span style="color:${r.mean_ic>0?'var(--text)':'var(--mute)'}">${f(r.mean_ic,3)}</span>`],["IR",r=>f(r.ir,2)]]);
 const icp=ic.find(r=>r.score==="p_up");
 reads("r_ic",
  icp?`How well each score lined up with what happened, day by day. ${PNAME} scored ${f(icp.mean_ic,3)}; above about 0.05 is strong for stock ranking. The best/worst-move rows are usually much higher because volatility is easier to predict than direction.`:"",
  `Mean daily Spearman between score and realized label, OOS; IR = mean/std of the daily series. Overlapping h-day labels autocorrelate the series, so IR overstates significance for h &gt; 1.`);
}

function renderDrivers(){
 const S=D.shap[h]||{},sk=(S.skew||[]).slice(0,12),gb=(S.gbm||[]).slice(0,12);
 const bee=S.beeswarm||[];
 if(bee.length){
  const feats=[...new Set(bee.map(b=>b.feature))],nF=feats.length,pts=[];
  feats.forEach((fe,fi)=>{const rows=bee.filter(b=>b.feature===fe).map(b=>({...b,v:b.shap*100}));const lo=Math.min(...rows.map(r=>r.v)),hi=Math.max(...rows.map(r=>r.v)),w=(hi-lo)/50||1,bins={};
   rows.sort((a,b)=>a.v-b.v).forEach(r=>{const k=Math.round((r.v-lo)/w);const n=bins[k]=(bins[k]||0)+1;const off=(n%2?1:-1)*Math.ceil((n-1)/2)*.07;pts.push({x:r.v,y:(nF-1-fi)+Math.max(-.42,Math.min(.42,off)),c:diverge(r.x),fe,xv:r.x})})});
  mk("c_skshap",{type:"scatter",data:{datasets:[{data:pts,backgroundColor:pts.map(p=>p.c),borderWidth:0,pointRadius:2.6,pointHoverRadius:5}]},
   options:{maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${nm(c.raw.fe)}: feature ${c.raw.xv<.33?"low":c.raw.xv>.66?"high":"mid"} → ${c.raw.x>=0?"+":""}${c.raw.x.toFixed(3)} pp`}}},
   scales:{x:{...yGrid,title:{display:true,text:"SkewSHAP (pp of tail asymmetry)"},grid:{color:c=>c.tick.value===0?AXIS:GRID}},y:{...xNo,min:-.6,max:nF-.4,ticks:{stepSize:1,autoSkip:false,color:INK2,font:{size:10.5},callback:v=>Number.isInteger(v)&&feats[nF-1-v]?nm(feats[nF-1-v]):""}}}}});
  document.getElementById("ramp").innerHTML=`feature value <span>low</span><i style="background:linear-gradient(90deg,${S2},${NEU},${S1})"></i><span>high</span>`;
 } else {hbars("c_skshap",sk.map(r=>r.feature),sk.map(r=>r.skew_mean*100),true,v=>(v>0?"+":"")+v.toFixed(4)+" pp",v=>v+" pp");document.getElementById("ramp").innerHTML="";}
 hbars("c_gbshap",gb.map(r=>r.feature),gb.map(r=>r.mean_abs),false,v=>f(v,4)+" log-odds");
 const ups=sk.filter(r=>r.skew_mean>0).slice(0,2).map(r=>nm(r.feature)),dns=sk.filter(r=>r.skew_mean<0).slice(0,2).map(r=>nm(r.feature));
 reads("r_skshap",
  `Each row is an input, each dot a name. Dots to the right widened the upside relative to the downside; to the left, the opposite. Colour is the name's own value on that input: ${C1N} = high versus peers, ${C2N} = low. So "${C1N} dots on the right" reads as "being high on this helps upside skew". Rows near the top matter most. ${ups.length?`On average, upside comes from ${ups.join(" and ")}`:""}${dns.length?`; downside from ${dns.join(" and ")}.`:"."}`,
  `SHAP summary (beeswarm) of local SkewSHAP on a 250-row training sample of the live quantile models; top 10 features by mean |SkewSHAP|. Colour = feature value min-max scaled within the sample (diverging, grey at mid). Dots are stacked within 50 bins per row to show density. Spread along x = heterogeneity; colour sorting along x = monotone effect; mixed colours = interactions or non-monotone effects (see the dependence plot).`);
 reads("r_gbshap",
  gb.length?`The inputs that move the direction call most: ${gb.slice(0,3).map(r=>nm(r.feature)).join(", ")}. This is size of influence, not direction.`:"",
  `Mean |TreeSHAP| on the live LightGBM classifier's log-odds. Correlated features share credit, so a family (e.g. several volatility measures) can look less important one by one than it is together.`);
 const dp=D.dependence[h];
 if(dp){document.getElementById("deptitle").textContent=`Dependence · ${nm(dp.feature)} (top skew driver)`;
  const pts=dp.points,a=pts.filter(p=>!p.post_climax),b=pts.filter(p=>p.post_climax);
  mk("c_dep",{type:"scatter",data:{datasets:[
    {label:"normal",data:a.map(p=>({x:p.x,y:p.shap*100})),backgroundColor:MUTE+"80",borderColor:PANEL,borderWidth:1,pointRadius:3.5},
    {label:"post-climax (≤5 sessions after extreme volume)",data:b.map(p=>({x:p.x,y:p.shap*100})),backgroundColor:S1,borderColor:PANEL,borderWidth:1.5,pointRadius:5}]},
   options:{maintainAspectRatio:false,interaction:{mode:"nearest",intersect:false},plugins:{legend:{position:"top",align:"start",labels:{boxWidth:10,boxHeight:10}},tooltip:{callbacks:{label:c=>`rank ${f(c.raw.x,2)} → ${c.raw.y.toFixed(3)} pp`}}},
   scales:{x:{...yGrid,min:0,max:1,title:{display:true,text:"peer percentile (0 = lowest in group, 1 = highest)"}},y:{...yGrid,title:{display:true,text:"SkewSHAP, pp"}}}}});
  reads("r_dep",
   `How the top skew driver behaves across its range: left = the stock is low on ${nm(dp.feature)} versus peers, right = high. Points above zero widen the upside. ${C1N[0].toUpperCase()+C1N.slice(1)} points are post-climax setups — if they sit apart from the grey ones, the volume event changes how the feature works.`,
   `Peer-percentile of ${dp.feature} vs local SkewSHAP on a training sample (≤600 points shown), coloured by the post_climax flag. Curvature or a split by colour is what the trees use that a linear model wouldn't.`);
 } else {document.getElementById("c_dep").parentElement.innerHTML='<div class="empty">No dependence sample for this horizon.</div>';reads("r_dep","","");}
 const gg=(D.gate_grid||{})[h];const ks=gg?gg.rows:[];
 document.getElementById("gate").innerHTML=gg&&ks.length?`<table><thead><tr><th style="writing-mode:horizontal-tb;transform:none;height:auto"></th>${gg.folds.map((fd,i)=>`<th title="test from ${fd}${gg.relaxed[i]?" · gate relaxed":""}">${fd}${gg.relaxed[i]?" †":""}</th>`).join("")}<th style="writing-mode:horizontal-tb;transform:none;height:auto">share</th></tr></thead><tbody>`+
  ks.map(r=>`<tr><td class="f">${nm(r.feature)}</td>${r.cells.map((c,i)=>`<td><div class="cell ${c?"on":""}" title="${nm(r.feature)} · ${gg.folds[i]}: ${c?"passed":"not used"}"></div></td>`).join("")}<td class="sh">${(r.share*100).toFixed(0)}%</td></tr>`).join("")+`</tbody></table>`
  :'<div class="empty">Run the walk-forward to see the gate history.</div>';
 reads("r_surv",
  ks.length?`Each column is one retraining (labelled by the month its test period starts); a filled square means that input passed the screen and was used. Solid rows are inputs the model trusts in every period. Rows that fill in only recently, or drop out, show where the market changed what works — worth knowing before you lean on that input.`:"Run the walk-forward to see this.",
  `Features × folds, top 20 by pass frequency; † marks folds where fewer than 3 features passed and the gate fell back to top-10 by |t|. Share of folds where the feature passed the in-fold IC gate (|mean daily IC| ≥ 0.015, |IR| ≥ 0.15, quarterly sign stability ≥ 55%, any core target or HV/post-climax condition), capped at 30 survivors per fold. ${m.relaxed_folds?`${m.relaxed_folds} folds fell back to the top-10-by-|t| rule.`:""}`);
}
render();
</script></body></html>"""
