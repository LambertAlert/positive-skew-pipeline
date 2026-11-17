"""
psp_dashboard.py — Streamlit dashboard for the positive-skew pipeline.

    streamlit run psp_dashboard.py            (or press F5 in VS Code: .vscode/launch.json is included)

Reads a finished run's files (default data/psp_out/: live_scores.parquet, metrics_*.csv, gates.csv,
shap_*.csv, meta.json) and draws every panel with Plotly. The Run tab can also start the pipeline
(full backtest, live-only, or a synthetic demo) in the background and show its log.

Every panel has a Trader read and a Data-scientist read (sidebar: Reads). Themes: Midnight,
Graphite, Paper and War Room; each theme's data colours passed the colour-blind/contrast validator
against its own background. Positive vs negative always uses the theme's warm/cool pair; green and
red are kept for confidence status and always come with a ✓/✕ label.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Positive-Skew Pipeline", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------------------------- themes
THEMES = {
    "Midnight": dict(bg="#0b0f17", panel="#121722", panel2="#171d2a", line="#232b3b", grid="#1a2130", axis="#2c3547",
                     text="#e6ebf2", ink2="#aab4c3", mute="#6f7b8e", accent="#7fb2ff", s1="#3b8fd9", s2="#e0703a",
                     s3="#8b74e8", neutral="#3a4457", good="#3fbf7f", bad="#e5534b", names=("blue", "orange", "violet"),
                     font="Inter, system-ui, -apple-system, Segoe UI, sans-serif", mono="JetBrains Mono, ui-monospace, monospace",
                     head="Inter, system-ui, sans-serif"),
    "Graphite": dict(bg="#101114", panel="#17191e", panel2="#1c1f25", line="#2a2d35", grid="#202329", axis="#33363f",
                     text="#ececee", ink2="#b4b7be", mute="#777b84", accent="#e8e8ea", s1="#4a90e2", s2="#d9682f",
                     s3="#22a08a", neutral="#3a3d46", good="#3fbf7f", bad="#e5534b", names=("blue", "orange", "teal"),
                     font="Inter, system-ui, sans-serif", mono="JetBrains Mono, ui-monospace, monospace", head="Inter, system-ui, sans-serif"),
    "Paper": dict(bg="#f4f3ef", panel="#ffffff", panel2="#f8f7f4", line="#e3e1db", grid="#eeece7", axis="#cfccc4",
                  text="#1c2230", ink2="#4a5263", mute="#868c99", accent="#1d2a44", s1="#2563c9", s2="#d4622a",
                  s3="#0e8f7a", neutral="#d9d6ce", good="#16803c", bad="#c0362c", names=("blue", "orange", "teal"),
                  font="Inter, system-ui, sans-serif", mono="JetBrains Mono, ui-monospace, monospace", head="Inter, system-ui, sans-serif"),
    "War Room": dict(bg="#09090e", panel="#111119", panel2="#16161f", line="#23232e", grid="#1b1b25", axis="#2a2a35",
                     text="#ebe6dc", ink2="#b9b2a5", mute="#7f796d", accent="#f5a524", s1="#c97f17", s2="#5b84d6",
                     s3="#2a9d7e", neutral="#3a3a46", good="#4fbf8a", bad="#e0605a", names=("amber", "blue", "teal"),
                     font="Fira Code, ui-monospace, monospace", mono="Fira Code, ui-monospace, monospace", head="Orbitron, sans-serif"),
}
ETF_NAMES = {"XLK": "Technology", "XLV": "Health care", "XLF": "Financials", "XLE": "Energy", "XLI": "Industrials",
             "XLY": "Consumer discretionary", "XLP": "Consumer staples", "XLB": "Materials", "XLU": "Utilities",
             "XLRE": "Real estate", "XLC": "Communication", "SPY": "Unmapped (vs SPY)"}
METRIC_KEYS = ("direction", "calibration", "lift", "quantiles", "skew_buckets", "flags", "overlay", "ic_oos", "edge", "tails")


def css(T: dict) -> str:
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Orbitron:wght@500;700&family=Fira+Code:wght@400;500&display=swap');
.stApp, [data-testid="stHeader"] {{ background:{T['bg']}; color:{T['text']}; }}
[data-testid="stSidebar"] {{ background:{T['panel']}; border-right:1px solid {T['line']}; }}
.stApp {{ font-family:{T['font']}; }}
.stMarkdown p, .stMarkdown li, .stApp label, [data-testid="stMetricLabel"] p, [data-testid="stWidgetLabel"] p {{ font-family:{T['font']}; color:{T['text']}; }}
[data-testid="stAlert"] p, [data-testid="stAlert"] li {{ color:inherit !important; }}
[data-testid="stCaptionContainer"], .stApp small {{ color:{T['mute']} !important; }}
[data-testid="stMetricValue"] {{ font-family:{T['mono']}; color:{T['text']}; }}
h1, h2, h3 {{ font-family:{T['head']}; color:{T['accent']} !important; letter-spacing:.02em; }}
h3 {{ font-size:1.05rem !important; color:{T['text']} !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:{T['panel']}; border-color:{T['line']} !important; }}
.stTabs [data-baseweb="tab"] {{ color:{T['ink2']}; }}
.stTabs [aria-selected="true"] {{ color:{T['accent']} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{T['accent']}; }}
code {{ font-family:{T['mono']}; }}
</style>"""


# ----------------------------------------------------------------------------------------------- formatting
def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pc(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{'+' if x > 0 else ''}{100 * x:.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}%"


def nm(x: str) -> str:
    return str(x).replace("__x__", " × ").replace("_pr", "")


def tlab(t):
    if t is None or not np.isfinite(t):
        return ""
    return "fat" if t >= 1.15 else "thin" if t <= 0.9 else "normal"


# ----------------------------------------------------------------------------------------------- data
def _stamp(out: Path) -> float:
    return max((p.stat().st_mtime for p in out.glob("*") if p.is_file()), default=0.0)


@st.cache_data(show_spinner="Loading results…")
def load(out_dir: str, stamp: float) -> dict:
    out = Path(out_dir)
    d = {"meta": json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}}
    d["live"] = pd.read_parquet(out / "live_scores.parquet") if (out / "live_scores.parquet").exists() else pd.DataFrame()
    for k in METRIC_KEYS:
        p = out / f"metrics_{k}.csv"
        d[k] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    d["gates"] = pd.read_csv(out / "gates.csv") if (out / "gates.csv").exists() else pd.DataFrame()
    shap = {}
    for p in out.glob("shap_*_h*.csv"):
        kind, h = p.stem.rsplit("_h", 1)
        shap.setdefault(int(h), {})[kind.replace("shap_", "")] = p
    d["shap"] = {h: {k: (pd.read_csv(v, index_col=0) if k in ("gbm", "skew", "rf", "linear") else pd.read_csv(v))
                     for k, v in r.items()} for h, r in shap.items()}
    if len(d["live"]):
        d["live"]["date"] = pd.to_datetime(d["live"]["date"]).dt.strftime("%Y-%m-%d")
        if "bench_etf" not in d["live"]:
            d["live"]["bench_etf"] = "SPY"
        d["live"]["bench_etf"] = d["live"]["bench_etf"].fillna("SPY")
    return d


# ----------------------------------------------------------------------------------------------- plotting helpers
def base(T, height=300, **kw):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=T["mono"], size=11, color=T["mute"]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=T["ink2"])),
                      hoverlabel=dict(bgcolor=T["panel2"], bordercolor=T["line"], font=dict(color=T["text"], family=T["mono"])),
                      **kw)
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["axis"], linecolor=T["axis"], tickfont=dict(color=T["mute"]))
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["axis"], linecolor=T["axis"], tickfont=dict(color=T["mute"]))
    return fig


def show(fig, key=None, select=False):
    cfg = {"displayModeBar": False}
    if select:
        return st.plotly_chart(fig, width="stretch", config=cfg, key=key, on_select="rerun", selection_mode="points")
    st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    return None


def picked(event):
    try:
        pts = event.selection.points if hasattr(event, "selection") else event["selection"]["points"]
        if pts:
            cd = pts[0].get("customdata")
            return cd[0] if isinstance(cd, (list, tuple)) else cd
    except Exception:
        pass
    return None


def reads(tr: str, ds: str):
    mode = st.session_state.get("reads", "Both")
    if mode in ("Both", "Trader") and tr:
        st.markdown(f":blue[**TRADER**]  \n{tr}")
    if mode in ("Both", "Data scientist") and ds:
        st.markdown(f":violet[**DATA SCIENTIST**]  \n{ds}")


def hbars(T, labels, vals, signed=True, unit="", height=330, key=None):
    labels = [nm(x) for x in labels][::-1]
    vals = list(vals)[::-1]
    fig = base(T, height)
    fig.add_bar(y=labels, x=vals, orientation="h",
                marker=dict(color=[T["s2"] if signed and v < 0 else T["s1"] for v in vals], cornerradius=4),
                hovertemplate="%{y}: %{x:.4f} " + unit + "<extra></extra>")
    fig.update_yaxes(tickfont=dict(color=T["ink2"]), gridcolor="rgba(0,0,0,0)")
    fig.update_layout(showlegend=False)
    show(fig, key)


def waterfall(T, base_v, items, total, k, unit, end_label, key):
    names = ["typical name"] + [nm(n) for n, _ in items]
    vals = [base_v * k] + [v * k for _, v in items]
    other = total * k - sum(vals)
    if abs(other) > 1e-9:
        names.append("all other features")
        vals.append(other)
    names.append(end_label)
    vals.append(0)
    measure = ["absolute"] + ["relative"] * (len(names) - 2) + ["total"]
    fig = base(T, 360)
    fig.add_trace(go.Waterfall(orientation="h", y=names, x=vals, measure=measure,
                               increasing=dict(marker=dict(color=T["s1"])), decreasing=dict(marker=dict(color=T["s2"])),
                               totals=dict(marker=dict(color=T["ink2"])), connector=dict(line=dict(color=T["axis"], width=1)),
                               hovertemplate="%{y}: %{x:+.3f} " + unit + "<extra></extra>"))
    fig.update_yaxes(tickfont=dict(color=T["ink2"]), gridcolor="rgba(0,0,0,0)", autorange="reversed")
    fig.update_xaxes(title_text="tail asymmetry, pp" if unit == "pp" else "log-odds of beating the sector")
    fig.update_layout(showlegend=False)
    show(fig, key)


# ----------------------------------------------------------------------------------------------- run controls
def run_tab(T, out_dir: Path):
    st.subheader("Run the pipeline")
    st.caption("Starts `python -m pipeline.run` in the background from this folder and writes to the output folder above. "
               "A full run downloads Massive data and fits the backtest, which takes a while — you can keep using the dashboard; "
               "press Refresh results when the log says done.")
    log_p, pid_p = out_dir / "run.log", out_dir / "run.pid"
    running = False
    if pid_p.exists():
        try:
            pid = int(pid_p.read_text())
            os.kill(pid, 0)
            running = True
        except Exception:
            running = False
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("What to run", ["Live scores only (daily)", "Full backtest + live", "Synthetic demo (no key)"],
                        index=0, key="run_mode")
        horizons = st.multiselect("Horizons", [3, 5, 7, 10, 21], default=[3, 5, 7, 10, 21], key="run_h")
        quick = st.checkbox("Faster backtest (126-session test blocks, 300K training rows)", value=True, key="run_quick")
    with c2:
        key_set = bool(os.environ.get("MASSIVE_API_KEY", "").strip())
        api_key = st.text_input("Massive API key", type="password", key="run_key",
                                help="Used for this run only, not saved. Leave blank if MASSIVE_API_KEY is set.",
                                placeholder="set in environment" if key_set else "required for Massive runs")
        regime = st.text_input("War Room regime snapshot", value="", key="run_regime",
                               placeholder="default: latest from LambertAlert/pivot-scanner on GitHub",
                               help="Leave blank to download the latest narrative_regime.parquet from pivot-scanner each run. "
                                    "Or enter a local path, e.g. ../pivot-scanner/data/narrative_regime.parquet")
        backfill = st.checkbox("Backfill the feature warm-up from yfinance", value=False, key="run_bf",
                               help="More scored history on short plans; the backfill is used for warm-up only.")
    args = [sys.executable, "-m", "pipeline.run", "--out", str(out_dir)]
    if mode.startswith("Live"):
        args.append("--live-only")
    if mode.startswith("Synthetic"):
        args += ["--source", "synthetic"]
    if horizons:
        args += ["--horizons", *map(str, sorted(horizons))]
    if quick and not mode.startswith("Live"):
        args += ["--test-sessions", "126", "--max-train-rows", "300000"]
    if regime.strip():
        args += ["--regime-path", regime.strip()]
    if backfill:
        args += ["--backfill", "auto"]
    st.code(" ".join(a if " " not in a else f'"{a}"' for a in args[1:]), language="bash")
    b1, b2, b3 = st.columns([1, 1, 4])
    with b1:
        go_btn = st.button("Start", type="primary", disabled=running, width="stretch")
    with b2:
        if st.button("Refresh results", width="stretch"):
            st.cache_data.clear()
            st.rerun()
    if go_btn:
        env = dict(os.environ)
        if api_key.strip():
            env["MASSIVE_API_KEY"] = api_key.strip()
        if not mode.startswith("Synthetic") and not env.get("MASSIVE_API_KEY", "").strip():
            st.error("No Massive API key: enter it above or set MASSIVE_API_KEY.")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            fh = open(log_p, "w")
            p = subprocess.Popen(args, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
            pid_p.write_text(str(p.pid))
            time.sleep(1.0)
            st.rerun()
    st.markdown(f"**Status:** {'🟢 running' if running else '⚪ idle'}")
    if log_p.exists():
        lines = [l for l in log_p.read_text(errors="ignore").splitlines() if "Warning" not in l and "warnings.warn" not in l]
        st.code("\n".join(lines[-40:]) or "(no output yet)", language="text")
        if running:
            st.button("Update log")


# ----------------------------------------------------------------------------------------------- app
def main():
    # sidebar --------------------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Positive-skew pipeline")
        theme = st.selectbox("Theme", list(THEMES), index=0, key="theme",
                             help="Paper is the light theme: switch Streamlit to Light too (⋮ → Settings → Theme).")
        out_dir = Path(st.text_input("Results folder", value=str(HERE / "data" / "psp_out"), key="out_dir"))
        st.radio("Reads", ["Both", "Trader", "Data scientist"], horizontal=True, key="reads")
    T = THEMES[theme]
    st.markdown(css(T), unsafe_allow_html=True)
    if theme == "Paper":
        st.sidebar.caption("Paper is the light theme — for matching tables and inputs, switch Streamlit to Light too "
                           "(⋮ menu → Settings → Theme).")
    C1N, C2N, C3N = T["names"]

    have = (out_dir / "live_scores.parquet").exists()
    if not have:
        st.title("Positive-skew path pipeline")
        st.info(f"No results in `{out_dir}` yet. Use the Run tab to start a run (try the synthetic demo first), "
                "or point *Results folder* at an existing run.")
        run_tab(T, out_dir)
        return
    D = load(str(out_dir), _stamp(out_dir))
    m, live = D["meta"], D["live"]
    H = sorted(live["horizon"].unique().tolist())
    REL = m.get("target_mode", "relative") != "absolute"
    PN = "P(beat)" if REL else "P(up)"
    MAGW = "vs its sector ETF" if REL else "absolute"
    R = m.get("regime") or {}

    with st.sidebar:
        h = st.radio("Horizon (sessions)", H, index=H.index(5) if 5 in H else 0, horizontal=True, key="h")
        etfs = sorted(live["bench_etf"].unique())
        sector = st.selectbox("Sector", ["All sectors"] + etfs, key="sector",
                              format_func=lambda e: e if e == "All sectors" else f"{e} · {ETF_NAMES.get(e, e)}")
        show_all = st.toggle("Show all names (not just flagged)", value=False, key="all")
        query = st.text_input("Ticker search", key="q").strip().upper()

    # header ---------------------------------------------------------------------------------------
    st.title("Positive-skew path pipeline")
    bits = [f"Massive · {m.get('start', '?')} → {m.get('end', '?')}",
            f"universe: Luk {' ∪ '.join(m.get('screens', []))}, close > ${m.get('price_floor', 9):g}",
            f"{m.get('n_names', 0):,} names (~{round(m.get('avg_universe', 0))}/day)",
            f"target: {'beat sector ETF' if REL else 'absolute return'}"]
    if m.get("profile") == "short":
        bits.append("short-history backtest")
    if m.get("backfilled"):
        bits.append(f"yfinance warm-up before {m.get('first_massive')}")
    st.caption(" · ".join(bits) + f" · built {m.get('generated', '')}")
    if m.get("source") == "synthetic":
        st.error("SYNTHETIC DATA — layout only, not market evidence.")
    if R.get("posture") and R.get("stale"):
        st.warning(f"War Room regime snapshot is {R.get('age_days')} days old (as of {str(R.get('generated_at'))[:16]}), so the "
                   f"View column ignores it. Last posture was {R['posture']}. Check the War Room pipeline in pivot-scanner.")
    elif R.get("posture"):
        txt = (f"War Room posture **{R['posture']}**"
               + (f" ({100 * R['posture_confidence']:.0f}% conf.)" if R.get("posture_confidence") is not None else "")
               + (f" · regime score {f(R.get('regime_score'), 2)}" if R.get("regime_score") is not None else "")
               + (f" · entry mode **{R['entry_mode']}**" if R.get("entry_mode") else "")
               + (f" · {R['current_state_name']}" if R.get("current_state_name") else "")
               + (f" · as of {str(R['generated_at'])[:16]}" if R.get("generated_at") else "")
               + (f"  \n_{R['entry_mode_reason']}_" if R.get("entry_mode_reason") else "")
               + " — flagged names are labelled for this tape in the View column.")
        (st.error if R["posture"] == "AVOID" else st.warning if R["posture"] == "NEUTRAL" else st.success)(txt)
    else:
        st.warning("No War Room regime snapshot in this run — rankings show which names should beat their sector, not "
                   "whether to be active. Runs download it from pivot-scanner on GitHub by default; re-run live scores to add it.")

    # shared slices --------------------------------------------------------------------------------
    LH = live[live["horizon"] == h]
    L = LH if sector == "All sectors" else LH[LH["bench_etf"] == sector]
    F = L[L["positive_skew"].astype(bool)]
    byt = {t: g.set_index("horizon") for t, g in live.groupby("ticker")}
    nflag = lambda t: int(byt[t]["positive_skew"].astype(bool).sum()) if t in byt else 0
    base_rows = L[L["ticker"].str.contains(query, regex=False)] if query else (L if show_all else F)
    if "sel" not in st.session_state or st.session_state.sel not in set(LH["ticker"]):
        st.session_state.sel = (F if len(F) else LH).sort_values("skew_score", ascending=False)["ticker"].iloc[0] if len(LH) else None

    tabs = st.tabs(["Shortlist", "Name drill-down", "Does it work?", "What drives it", "Run"])

    # ============================================================================== shortlist
    with tabs[0]:
        c = st.columns(6)
        kp = [(len(L), "names in universe" + (f" ({sector})" if sector != "All sectors" else "")),
              (len(F), "positive-skew flags"), (sum(nflag(t) >= 3 for t in L["ticker"]), "flagged on 3+ horizons"),
              (f(L["skew_score"].median(), 2), "median SkewScore"), (int((L["confidence"] == "low").sum()), "low-confidence"),
              (R.get("posture", "—"), "War Room posture")]
        for col, (v, lab) in zip(c, kp):
            col.metric(lab, v)
        tb = base_rows.sort_values("skew_score", ascending=False).copy()
        hz = lambda t: " ".join(f"{x}D" for x in H if t in byt and x in byt[t].index and bool(byt[t].loc[x, "positive_skew"])) or "—"
        view = pd.DataFrame({
            "Ticker": tb["ticker"], "ETF": tb["bench_etf"], PN: tb["p_up"], "SkewScore": tb["skew_score"],
            "Tails": tb.get("tail_weight", np.nan), "Tail": [tlab(x) for x in tb.get("tail_weight", pd.Series(np.nan, index=tb.index))],
            "Flagged at": [hz(t) for t in tb["ticker"]], "Flags": [nflag(t) for t in tb["ticker"]],
            "q10 %": tb["mag_q10"] * 100, "q50 %": tb["mag_q50"] * 100, "q90 %": tb["mag_q90"] * 100,
            "Best move %": tb["mfe_q50"] * 100, "Worst move %": tb["mae_q50"] * 100,
            "Confidence": np.where(tb["confidence"] == "low", "✕ low", "✓ normal"),
            "View": tb.get("trade_view", ""),
            "Setup": [f"{r.hv_regime} vol" + (" · post-climax" if r.post_climax else "") + " · "
                      + " ".join(s for s, k in (("PM", "scr_premarket"), ("LDR", "scr_leader"), ("POT", "scr_potent")) if bool(getattr(r, k, False)))
                      for r in tb.itertuples()],
        }).reset_index(drop=True)
        cc = st.column_config
        ev = st.dataframe(view, hide_index=True, width="stretch", height=min(560, 38 + 35 * max(1, len(view))),
                          on_select="rerun", selection_mode="single-row", key="tbl",
                          column_config={
                              PN: cc.ProgressColumn(PN, min_value=0.4, max_value=0.65, format="%.3f"),
                              "SkewScore": cc.NumberColumn(format="%.2f"), "Tails": cc.NumberColumn(format="%.2f"),
                              **{k: cc.NumberColumn(format="%+.1f%%") for k in ("q10 %", "q50 %", "q90 %", "Best move %", "Worst move %")},
                          })
        rows = ev.selection.rows if hasattr(ev, "selection") else []
        if rows:
            st.session_state.sel = view.iloc[rows[0]]["Ticker"]
        st.caption(f"{len(view)} name{'s' if len(view) != 1 else ''} · click a row to open it in **Name drill-down** · "
                   "Flagged at = every horizon where the same name is flagged; Flags = how many")
        e1, e2 = st.columns([3, 1])
        with e1:
            st.code(",".join(view["Ticker"]) or "(none)", language="text")
        with e2:
            st.download_button("Download CSV", tb.to_csv(index=False).encode(), width="stretch",
                               file_name=f"positive_skew_{m.get('end', '')}_{h}D{'' if sector == 'All sectors' else '_' + sector}.csv")
        tbh = D["tails"][D["tails"]["horizon"] == h] if len(D["tails"]) else pd.DataFrame()
        tail_ok = None if tbh.empty else bool(tbh["realized_tail"].iloc[-1] > tbh["realized_tail"].iloc[0] + .03)
        top = F.sort_values("skew_score", ascending=False).head(1)
        reads(
            f"{len(F)} of {len(L)} names are flagged for the next {h} sessions: the model expects them to beat "
            f"{'their sector ETF' if REL else 'zero'} with a wider upside than downside"
            + (f" (strongest asymmetry: **{top.iloc[0]['ticker']}**, SkewScore {f(top.iloc[0]['skew_score'], 2)})" if len(top) else "")
            + ". *Flagged at* lists every holding period where the same name is flagged; a name flagged across several is a sturdier read (sort by *Flags*). "
              "q10/q50/q90 are the model's middle-80% range and median; Tails compares that full range to the middle half — "
              "\"fat\" means mostly modest outcomes with the occasional large one, so size and stops need room for it."
            + (" **Caution:** at this horizon the backtest hasn't confirmed that predicted tails show up afterwards, so read Tails as the model's view." if tail_ok is False else "")
            + " Copy the ticker list into TradingView or Barchart from the box above; let the View column (the regime) decide whether to act.",
            f"Live model fit on every labelled date (same purged calibration tail as the folds). {PN} = isotonic-calibrated mean of "
            f"logistic, RF and LightGBM for `{'dir_xs_h' if REL else 'dir_ret_h'}`"
            + (" (beat the day's median universe member on return minus sector ETF; 50% base rate by construction)" if REL else "")
            + f". Quantiles from LightGBM on `{'rel_h' if REL else 'ret_h'}`, rearranged. SkewScore = (q90−q50)/max(q50−q10, floor). "
              "Tails = [(q90−q10)/(q75−q25)]/1.90 (normal = 1). Flag = SkewScore > 1.2, P ≥ 0.5, Isolation Forest not anomalous. "
              "Best/worst move = MFE/MAE medians, absolute.")

        st.divider()
        a, b = st.columns(2)
        with a:
            st.markdown("### Upside vs downside map")
            P = L.dropna(subset=["mag_q10"]).assign(x=lambda d: (d.mag_q50 - d.mag_q10) * 100, y=lambda d: (d.mag_q90 - d.mag_q50) * 100)
            if len(P):
                lo = max(0, min(P.x.min(), P.y.min()) * .85)
                mx = max(P.x.max(), P.y.max()) * 1.05
                thr = 1.2
                fig = base(T, 420)
                fig.add_scatter(x=[lo, mx], y=[lo, mx], mode="lines", line=dict(color=T["neutral"], width=1.5), name="even (1.0)", hoverinfo="skip")
                fig.add_scatter(x=[lo / thr, mx / thr], y=[lo, mx], mode="lines", line=dict(color=T["mute"], width=1), name=f"flag line ({thr}×)", hoverinfo="skip")
                for flagged, name, col in ((False, "not flagged", T["mute"]), (True, "positive-skew flag", T["s1"])):
                    Q = P[P["positive_skew"].astype(bool) == flagged]
                    lab = Q.sort_values("skew_score", ascending=False)["ticker"].head(6).tolist() if flagged else []
                    fig.add_scatter(x=Q.x, y=Q.y, mode="markers+text" if flagged else "markers", name=name,
                                    text=[t if t in lab else "" for t in Q.ticker], textposition="top right",
                                    textfont=dict(color=T["text"], size=10),
                                    marker=dict(color=col, opacity=1 if flagged else .55, size=np.clip(6 + (Q.p_up - .45) * 110, 6, 18),
                                                line=dict(color=T["panel"], width=1)),
                                    customdata=Q[["ticker"]].to_numpy(),
                                    hovertemplate="%{customdata[0]}: up %{y:.1f} pp / down %{x:.1f} pp<extra></extra>")
                fig.update_xaxes(range=[lo, mx], title_text="downside width, q50 − q10 (pp)")
                fig.update_yaxes(range=[lo, mx], title_text="upside width, q90 − q50 (pp)")
                t = picked(show(fig, "map", select=True))
                if t:
                    st.session_state.sel = t
                reads(f"Each dot is a name: room to fall to its median (left–right) against room to rise above it (up–down). Above the "
                      f"diagonal = more upside than downside; above the steeper line = flagged. {int((P.y > P.x).sum())} of {len(P)} sit above even. "
                      f"Bigger dots have a higher {PN}. Best setups are high and to the left. Click a dot to open it.",
                      f"x = q50 − q10, y = q90 − q50 of predicted `{'rel_h' if REL else 'ret_h'}` (pp); size ∝ {PN}. "
                      f"SkewScore = y / max(x, floor), so iso-SkewScore lines through the origin are y = x and y = {thr}x.")
        with b:
            st.markdown(f"### Sector strip · every name by {PN}")
            by = LH.groupby("bench_etf")
            keys = sorted(by.groups, key=lambda e: len(by.get_group(e)))
            fig = base(T, 420)
            rng = lambda t: (hash(t) % 1000) / 1000 - .5
            for flagged, name, col, sz in ((False, "not flagged", T["mute"], 7), (True, "positive-skew flag", T["s1"], 10)):
                Q = LH[LH["positive_skew"].astype(bool) == flagged]
                yy = [keys.index(e) + rng(t) * .5 for e, t in zip(Q.bench_etf, Q.ticker)]
                dim = [1 if sector in ("All sectors", e) else .25 for e in Q.bench_etf]
                fig.add_scatter(x=Q.p_up, y=yy, mode="markers", name=name, customdata=Q[["ticker", "bench_etf"]].to_numpy(),
                                marker=dict(color=col, size=sz, opacity=dim, line=dict(color=T["panel"], width=1)),
                                hovertemplate="%{customdata[0]} (%{customdata[1]}): " + PN + " %{x:.3f}<extra></extra>")
            med = [by.get_group(e)["p_up"].median() for e in keys]
            fig.add_scatter(x=med, y=list(range(len(keys))), mode="markers", name="sector median",
                            marker=dict(symbol="line-ns", size=16, line=dict(color=T["text"], width=2.5)), hovertemplate="median %{x:.3f}<extra></extra>")
            fig.update_yaxes(tickvals=list(range(len(keys))), ticktext=keys, tickfont=dict(color=T["ink2"]), gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(title_text=PN)
            t = picked(show(fig, "strip", select=True))
            if t:
                st.session_state.sel = t
            best = keys[int(np.argmax(med))] if keys else ""
            reads(f"Every name, one row per sector, placed by how likely it is to beat that sector; the tick is the sector median. "
                  f"Rows whose dots sit to the right have more names the model likes — **{best}** has the highest median today. "
                  "Picking a sector in the sidebar dims the rest.",
                  f"Vertical jitter is a stable hash of the ticker; x = calibrated {PN}. Because {PN} is relative to each row's own ETF, "
                  "it measures within-sector opportunity, not which sector will win.")

        st.divider()
        st.markdown(f"### Best in each sector · top 3 by {PN}")
        groups = sorted(LH.groupby("bench_etf"), key=lambda kv: -len(kv[1]))
        cols = st.columns(4)
        for i, (e, g) in enumerate(groups):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{e}** · {ETF_NAMES.get(e, '')} · {len(g)} names")
                    t3 = g.sort_values("p_up", ascending=False).head(3)
                    st.dataframe(pd.DataFrame({"ticker": t3.ticker + np.where(t3.positive_skew.astype(bool), " ⚑", ""),
                                               PN: t3.p_up.round(3), "skew": t3.skew_score.round(2)}),
                                 hide_index=True, width="stretch")
        reads("If you already have the sector view — bullish semis, or defensive and want health care — start here. ⚑ = also "
              "carries the skew flag. Semis sit in XLK (ranked against tech broadly) and the Luk screens favour volatile names, "
              "so defensive sectors can be thin.",
              f"Groups = benchmark ETF from the SIC→SPDR map, ranked within group by calibrated {PN}.")

    # ============================================================================== drill-down
    with tabs[1]:
        opts = LH.sort_values("skew_score", ascending=False)["ticker"].tolist()
        if not opts:
            st.info("No live rows.")
        else:
            cur = st.session_state.sel if st.session_state.sel in opts else opts[0]
            sel = st.selectbox("Name", opts, index=opts.index(cur), key="sel_box")
            st.session_state.sel = sel
            r = LH[LH["ticker"] == sel].iloc[0]
            etf = r["bench_etf"]
            st.markdown(f"## {sel}{' ⚑' if r['positive_skew'] else ''}")
            st.caption(f"{etf} · {ETF_NAMES.get(etf, '')} · scored {r['date']} · {h}-session horizon · "
                       f"{'flagged' if r['positive_skew'] else 'not flagged'}"
                       + (f" · View: {r['trade_view']}" if r.get("trade_view") else "")
                       + f" · confidence {'✓ normal' if r['confidence'] != 'low' else '✕ low'}")
            c = st.columns(6)
            c[0].metric(PN, f(r["p_up"]))
            c[1].metric("SkewScore", f(r["skew_score"], 2))
            c[2].metric("median q50", pc(r["mag_q50"]), help=f"median return {MAGW}")
            c[3].metric(f"tails ({tlab(r.get('tail_weight', np.nan))})", f(r.get("tail_weight", np.nan), 2))
            c[4].metric("best move", pc(r["mfe_q50"]), help="typical best move during the trade (absolute)")
            c[5].metric("worst move", pc(r["mae_q50"]), help="typical worst move during the trade (absolute)")
            # big range bar
            q = {k: r[f"mag_{k}"] * 100 for k in ("q10", "q25", "q50", "q75", "q90") if f"mag_{k}" in r}
            fig = base(T, 120)
            fig.add_shape(type="rect", x0=q["q10"], x1=q["q50"], y0=-.18, y1=.18, fillcolor=T["s2"], line_width=0)
            fig.add_shape(type="rect", x0=q["q50"], x1=q["q90"], y0=-.18, y1=.18, fillcolor=T["s1"], line_width=0)
            if "q25" in q:
                fig.add_shape(type="rect", x0=q["q25"], x1=q["q75"], y0=-.36, y1=.36, line=dict(color=T["text"], width=2), fillcolor="rgba(0,0,0,0)")
            fig.add_shape(type="line", x0=q["q50"], x1=q["q50"], y0=-.45, y1=.45, line=dict(color=T["text"], width=3))
            fig.add_shape(type="line", x0=0, x1=0, y0=-.6, y1=.6, line=dict(color=T["mute"], width=1))
            fig.add_scatter(x=list(q.values()), y=[-.75] * len(q), mode="text", text=[f"{k} {v:+.1f}%" for k, v in q.items()],
                            textfont=dict(color=T["mute"], size=10), hoverinfo="skip")
            fig.update_yaxes(visible=False, range=[-1, .7])
            fig.update_xaxes(showgrid=False, zeroline=False, ticksuffix="%")
            fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=6, b=6))
            show(fig, "bigrng")
            up, dn = r["mag_q90"] - r["mag_q50"], r["mag_q50"] - r["mag_q10"]
            tw = r.get("tail_weight", np.nan)
            reads(f"{sel}: in 8 of 10 outcomes the model sees the {h}-session return {MAGW} between {pc(r['mag_q10'])} and "
                  f"{pc(r['mag_q90'])}, median {pc(r['mag_q50'])}; the upside half is {f(up / max(dn, 1e-9), 1)}× the downside half. "
                  f"Typical best move {pc(r['mfe_q50'])}, typical worst {pc(r['mae_q50'])} (absolute) — a stop inside that worst move will get hit often."
                  + (f" The outlined box is the middle half ({pc(r['mag_q25'])} to {pc(r['mag_q75'])}); tail weight {f(tw, 2)} means the full range is "
                     f"{f(tw * 1.9, 2)}× the box vs 1.90× for a normal distribution — "
                     + ("fat tails: allow room for the occasional large move." if tw >= 1.15 else "thin tails: fewer surprise moves." if tw <= .9 else "roughly normal tails.")
                     if np.isfinite(tw) else ""),
                  f"Quantiles from separate LightGBM regressions on `{'rel_h' if REL else 'ret_h'}`, rearranged. Up-width {pp(up, 2)}, "
                  f"down-width {pp(dn, 2)}. Disagreement (std of 4 calibrated direction models) = {f(r['disagreement'])}.")

            st.divider()
            a, b = st.columns(2)
            rows = byt[sel].sort_index()
            xs = rows.index.tolist()
            with a:
                st.markdown("### Range by holding period")
                fig = base(T, 340)
                fig.add_scatter(x=xs, y=rows.mag_q10 * 100, mode="lines", line=dict(color=T["s2"], width=1.5), name="q10", showlegend=False)
                fig.add_scatter(x=xs, y=rows.mag_q50 * 100, mode="lines+markers", fill="tonexty", fillcolor=_alpha(T["s2"], .2),
                                line=dict(color=T["text"], width=2), marker=dict(size=[11 if x == h else 6 for x in xs]), name="downside half (q10–q50)")
                fig.add_scatter(x=xs, y=rows.mag_q90 * 100, mode="lines", fill="tonexty", fillcolor=_alpha(T["s1"], .2),
                                line=dict(color=T["s1"], width=1.5), name="upside half (q50–q90)")
                fig.add_scatter(x=xs, y=rows.mfe_q50 * 100, mode="lines+markers", line=dict(color=T["mute"], width=1.2), marker=dict(size=4), name="typical best / worst move")
                fig.add_scatter(x=xs, y=rows.mae_q50 * 100, mode="lines+markers", line=dict(color=T["mute"], width=1.2), marker=dict(size=4), showlegend=False)
                fig.update_xaxes(tickvals=xs, ticktext=[f"{x}D" for x in xs], title_text="holding period (sessions)")
                fig.update_yaxes(ticksuffix="%")
                show(fig, "fan")
                ratio = (rows.mfe_q50 / rows.mae_q50.abs().clip(lower=1e-6))
                bh = int(ratio.idxmax())
                reads(f"How {sel}'s expected range changes with holding time: shaded band = middle 80% {MAGW} ({C2N} below the median, "
                      f"{C1N} above), grey lines = typical best and worst move during the trade. You want the {C1N} half widening faster. "
                      + (f"Best move beats worst move by the most at {bh} sessions ({f(ratio.max(), 2)}×)." if ratio.max() >= 1 else
                         f"At no horizon does the typical best move exceed the typical worst (closest {bh} sessions, {f(ratio.max(), 2)}×)."),
                      "Each horizon has its own quantile models fit independently, so the fan joins five marginal distributions — "
                      "not a simulated path. x is spaced by sessions; the larger dot is the horizon selected in the sidebar.")
            with b:
                st.markdown("### Skew, P(beat) and tails by holding period")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.06)
                cols_ = [T["s1"] if v else T["mute"] for v in rows.positive_skew.astype(bool)]
                sizes = [12 if x == h else 8 for x in xs]
                for i, (col, ref, lab) in enumerate((("skew_score", 1.2, "SkewScore"), ("p_up", .5, PN), ("tail_weight", 1.0, "tails")), 1):
                    if col not in rows:
                        continue
                    fig.add_scatter(x=xs, y=[ref] * len(xs), mode="lines", line=dict(color=T["neutral"], width=1.2), row=i, col=1, hoverinfo="skip")
                    fig.add_scatter(x=xs, y=rows[col], mode="lines+markers", line=dict(color=T["ink2"], width=1.8),
                                    marker=dict(color=cols_ if col != "tail_weight" else T["s3"], size=sizes, line=dict(color=T["panel"], width=1.5)),
                                    row=i, col=1, hovertemplate="%{x}D " + lab + " %{y:.3f}<extra></extra>")
                    fig.update_yaxes(title_text=lab, row=i, col=1)
                fig.update_layout(height=340, showlegend=False, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family=T["mono"], size=11, color=T["mute"]))
                fig.update_xaxes(gridcolor=T["grid"], tickvals=xs, ticktext=[f"{x}D" for x in xs])
                fig.update_yaxes(gridcolor=T["grid"])
                show(fig, "hz")
                fl = [f"{x}D" for x in xs if bool(rows.loc[x, "positive_skew"])]
                reads((f"Flagged at {', '.join(fl)}. " if fl else "Not flagged at any horizon. ")
                      + f"Skew peaks at {int(rows.skew_score.idxmax())} sessions ({f(rows.skew_score.max(), 2)}) and {PN} at "
                        f"{int(rows.p_up.idxmax())} ({f(rows.p_up.max())}). If both peak together, that's the natural holding period; "
                        "if not, the skew horizon suits a positive-skew trade with a hold-if-still-strong exit. Bottom: where tail weight rises, "
                        "that holding period carries more risk of a single large move.",
                      f"SkewScore, calibrated {PN} and tail weight from each horizon's live models; {C1N} dots mark horizons where every flag "
                      "condition passed. Reference lines: 1.2, 0.5, 1.0. Separate panels rather than two axes.")

            a, b = st.columns(2)
            sd = json.loads(r.get("skew_drivers") or "[]")
            dd = json.loads(r.get("dir_drivers") or "[]")
            with a:
                st.markdown("### Why the skew · SkewSHAP waterfall")
                if sd and pd.notna(r.get("skew_base", np.nan)):
                    waterfall(T, r["skew_base"], sd, r["skew_total"], 100, "pp", "this name", "wf_skew")
                elif sd:
                    hbars(T, [x[0] for x in sd], [x[1] * 100 for x in sd], unit="pp", key="wf_skew")
                if sd:
                    reads(f"Top to bottom: starts at the typical name's tilt, each bar adds ({C1N}) or removes ({C2N}) upside skew, and the last "
                          f"bar is {sel}'s own tilt. The biggest push is **{nm(sd[0][0])}**. If it comes from something you'd discount on the chart, discount the score.",
                          "SkewSHAP = SHAP(q90)+SHAP(q10)−2·SHAP(q50) in pp of (upper − lower) tail width; start = base value over training data, "
                          "end = this row's raw-model asymmetry; 'all other features' closes the gap left by the top 6.")
            with b:
                st.markdown("### Why the direction · GBM SHAP waterfall")
                if dd and pd.notna(r.get("dir_base", np.nan)):
                    waterfall(T, r["dir_base"], dd, r["dir_total"], 1, "log-odds", f"this name (raw P {1 / (1 + np.exp(-r['dir_total'])):.3f})", "wf_dir")
                elif dd:
                    hbars(T, [x[0] for x in dd], [x[1] for x in dd], unit="log-odds", key="wf_dir")
                if dd:
                    reads(f"**{nm(dd[0][0])}** matters most for {sel}'s direction call and {'raises' if dd[0][1] > 0 else 'lowers'} the odds it "
                          f"beats {'its sector' if REL else 'zero'}. Same reading as the skew waterfall.",
                          f"TreeSHAP on the live LightGBM classifier in log-odds. Raw P is before calibration and before averaging, so it differs from {PN}.")

    # ============================================================================== evidence
    with tabs[2]:
        if D["lift"].empty:
            st.info("No backtest results in this folder — run the full backtest (Run tab).")
        else:
            a, b = st.columns(2)
            with a:
                st.markdown(f"### Decile lift · ranked by {PN}")
                li = D["lift"][D["lift"]["horizon"] == h]
                magcol = "mean_mag" if "mean_mag" in li else "mean_ret"
                fig = base(T, 300)
                fig.add_bar(x=[f"D{d}" for d in li.dec], y=li[magcol] * 100, name=f"mean return {MAGW}", marker=dict(color=T["s1"], cornerradius=4))
                fig.add_bar(x=[f"D{d}" for d in li.dec], y=li["mean_ret"] * 100, name="mean absolute return", marker=dict(color=T["s2"], cornerradius=4))
                fig.update_yaxes(ticksuffix="%")
                show(fig, "lift")
                d1, d10 = li.iloc[0], li.iloc[-1]
                mono = int((li[magcol].diff().dropna() >= 0).sum())
                reads(f"Each day the universe is split into 10 buckets by {PN}. The top bucket returned {pc(d10[magcol], 2)} {MAGW} over {h} sessions "
                      f"and the bottom {pc(d1[magcol], 2)} — a {pp(d10[magcol] - d1[magcol], 2)} spread per trade, before costs. Use it to rank the review list.",
                      f"Per-date deciles of OOS {PN}, pooled across folds. {mono} of 9 steps are non-decreasing; monotonicity matters more than endpoints.")
            with b:
                st.markdown("### Edge over time · top minus bottom decile, by month")
                ed = D["edge"][D["edge"]["horizon"] == h] if len(D["edge"]) else pd.DataFrame()
                if len(ed):
                    fig = base(T, 300)
                    fig.add_bar(x=ed.month, y=ed.spread * 100, name="monthly spread",
                                marker=dict(color=[T["s2"] if v < 0 else T["s1"] for v in ed.spread], cornerradius=3))
                    fig.add_scatter(x=ed.month, y=ed.spread.rolling(6, min_periods=1).mean() * 100, mode="lines",
                                    line=dict(color=T["ink2"], width=2, shape="spline"), name="6-month average")
                    fig.update_yaxes(ticksuffix="%")
                    show(fig, "edge")
                    pos = int((ed.spread > 0).sum())
                    reads(f"Each bar is how much the top tenth beat the bottom tenth that month: {pos} of {len(ed)} months were positive. "
                          f"You want mostly {C1N} bars with the line above zero; long {C2N} runs mean the edge comes and goes.",
                          "Daily D10 − D1 spread on OOS scores, monthly means and a trailing 6-month mean. Overlapping labels for h > 1 "
                          "make neighbouring months share information; the hit rate is descriptive.")
            a, b = st.columns(2)
            with a:
                st.markdown("### Calibration · predicted vs realized")
                cal = D["calibration"][D["calibration"]["horizon"] == h]
                err = 1.96 * np.sqrt((cal.hit * (1 - cal.hit)).clip(lower=1e-4) / cal.n)
                fig = base(T, 320)
                fig.add_scatter(x=[.3, .7], y=[.3, .7], mode="lines", line=dict(color=T["neutral"], width=1.5), name="perfect calibration", hoverinfo="skip")
                fig.add_scatter(x=cal.mean_p, y=cal.hit, mode="markers", name="bins", error_y=dict(array=err, color=T["mute"], thickness=1.5, width=0),
                                marker=dict(color=T["s1"], size=np.clip(6 + np.log10(cal.n + 1) * 4, 6, 24), line=dict(color=T["panel"], width=2)),
                                customdata=cal[["n"]].to_numpy(), hovertemplate="predicted %{x:.3f} → realized %{y:.3f} (n=%{customdata[0]:,})<extra></extra>")
                fig.update_xaxes(range=[.2, .8], title_text=f"predicted {PN}")
                fig.update_yaxes(range=[.2, .8], title_text="realized rate")
                show(fig, "cal")
                dr = D["direction"].query("horizon == @h and regime == 'all' and model == 'ensemble'")
                big = cal[cal.n >= 500]
                reads(f"When the model says 55%, does it happen about 55% of the time? Dots on the line mean you can read {PN} at face value; "
                      "whiskers show how far off a dot could be by luck alone.",
                      "Reliability diagram with binomial 95% whiskers; dot size ∝ log n."
                      + (f" Largest gap among bins with n ≥ 500: {f((big.hit - big.mean_p).abs().max())}." if len(big) else "")
                      + (f" Brier skill vs base rate {pc(dr.iloc[0]['brier_skill'], 2)}." if len(dr) else ""))
            with b:
                st.markdown("### Range honesty · target vs realized coverage")
                qm = D["quantiles"][D["quantiles"]["horizon"] == h]
                band = qm[qm["tau"].astype(str) == "q10-q90"]
                qs = qm[qm["tau"].astype(str) != "q10-q90"].copy()
                qs["t"] = qs["tau"].astype(float)
                lab = [f"{'vs sector' if r_.target == 'mag' and REL else 'return' if r_.target == 'mag' else 'best move' if r_.target == 'mfe' else 'worst move'} q{round(r_.t * 100)}"
                       for r_ in qs.itertuples()]
                tt, vv = (qs.t * 100).tolist(), (qs.coverage_below * 100).tolist()
                if len(band):
                    lab.append("inside q10–q90"); tt.append(80); vv.append(float(band.coverage_below.iloc[0]) * 100)
                fig = base(T, 320)
                for y_, a_, b_ in zip(lab, tt, vv):
                    fig.add_scatter(x=[a_, b_], y=[y_, y_], mode="lines", line=dict(color=T["neutral"], width=4), showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=tt, y=lab, mode="markers", name="target", marker=dict(color=T["panel"], size=11, line=dict(color=T["mute"], width=2)))
                fig.add_scatter(x=vv, y=lab, mode="markers", name="realized", marker=dict(color=T["s3"], size=11, line=dict(color=T["panel"], width=1.5)),
                                hovertemplate="%{y}: realized %{x:.1f}%<extra></extra>")
                fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="share of outcomes below")
                fig.update_yaxes(autorange="reversed", tickfont=dict(color=T["ink2"]))
                show(fig, "cov")
                cov = float(band.coverage_below.iloc[0]) if len(band) else None
                reads((f"About 80% of outcomes should land inside q10–q90; historically {pp(cov)} did. "
                       + ("The ranges run narrow — real outcomes are wider." if cov < .75 else "The ranges run wide." if cov > .85 else "Close enough to trust the ranges.")) if cov else "",
                      f"Dumbbell: hollow = target share below each quantile, {C3N} = realized share OOS. The last row is the q10–q90 band vs 80%.")
            a, b = st.columns(2)
            sb = D["skew_buckets"][D["skew_buckets"]["horizon"] == h]
            with a:
                st.markdown("### Skew buckets · best move ÷ worst move")
                fig = base(T, 300)
                fig.add_bar(x=sb.bucket, y=sb.mfe_mae_ratio - 1, base=1, marker=dict(color=[T["s2"] if v < 1 else T["s1"] for v in sb.mfe_mae_ratio], cornerradius=4),
                            customdata=sb[["n", "mfe_mae_ratio"]].to_numpy(), hovertemplate="ratio %{customdata[1]:.3f} · n=%{customdata[0]:,}<extra></extra>")
                fig.update_yaxes(title_text="ratio (bars start at 1.0 = even)")
                fig.update_xaxes(title_text="predicted SkewScore bucket")
                fig.update_layout(showlegend=False)
                show(fig, "sk1")
                ok = sb[sb.n >= 200]
                verdict = ("" if ok.empty else f"It does here: {f(sb.iloc[0].mfe_mae_ratio, 2)} → {f(ok.iloc[-1].mfe_mae_ratio, 2)}."
                           if ok.iloc[-1].mfe_mae_ratio > sb.iloc[0].mfe_mae_ratio + .03 else
                           f"It doesn't clearly do that here ({f(sb.iloc[0].mfe_mae_ratio, 2)} → {f(ok.iloc[-1].mfe_mae_ratio, 2)}), so treat SkewScore with suspicion.")
                reads("The test of the whole idea: names with a higher predicted SkewScore should have run further in your favour than against "
                      "you. Above 1 = the average best move beat the average worst move. " + verdict,
                      "OOS rows bucketed by predicted SkewScore; mean MFE / |mean MAE| (absolute path). Check n in the tooltip for small buckets.")
            with b:
                st.markdown("### Skew buckets · distribution of outcomes")
                if "p10" in sb:
                    mg = "mean_mag" if "mean_mag" in sb else "mean_ret"
                    fig = base(T, 300)
                    fig.add_trace(go.Box(x=sb.bucket, q1=sb.p25 * 100, median=sb.p50 * 100, q3=sb.p75 * 100, lowerfence=sb.p10 * 100,
                                         upperfence=sb.p90 * 100, mean=sb[mg] * 100, name="outcomes", fillcolor=_alpha(T["s1"], .25),
                                         line=dict(color=T["s1"]), marker=dict(color=T["s1"]), boxmean=True))
                    fig.update_yaxes(ticksuffix="%")
                    fig.update_xaxes(title_text="predicted SkewScore bucket")
                    fig.update_layout(showlegend=False)
                    show(fig, "sk2")
                    reads(f"What actually happened per bucket, return {MAGW}: box = middle half, whiskers = 10th–90th percentile, solid line = median, "
                          "dashed = average. Positive skew looks like a longer top whisker and the average above the median. Skew without a positive "
                          "average is a lottery ticket.",
                          f"Realized `{'rel_h' if REL else 'ret_h'}` p10/p25/p50/p75/p90 and mean per bucket (Plotly box from precomputed stats).")
            a, b = st.columns(2)
            with a:
                st.markdown("### Flagged vs not · hold-if-still-strong")
                fl = D["flags"][D["flags"]["horizon"] == h]
                mg = "mean_mag" if "mean_mag" in fl else "mean_ret"
                st.dataframe(pd.DataFrame({"group": fl.group.str.replace("_", " "), "n": fl.n, "vs sector" if REL else "return": fl[mg].map(lambda x: pc(x, 2)),
                                           "abs return": fl.mean_ret.map(lambda x: pc(x, 2)), "hit": fl.hit.map(pp), "best÷worst": fl.mfe_mae_ratio.round(2),
                                           "Bowley": fl.realized_bowley.round(3), "tails": fl.get("realized_tail", pd.Series(np.nan, index=fl.index)).round(2)}),
                             hide_index=True, width="stretch")
                ov = D["overlay"][D["overlay"]["horizon"] == h]
                st.dataframe(pd.DataFrame({"hold rule": ov.group.str.replace("_", " "), "n": ov.n, "extended": ov.share_extended.map(pp),
                                           "fixed exit": ov.fixed_ret.map(lambda x: pc(x, 2)), "hold if strong": ov.overlay_ret.map(lambda x: pc(x, 2))}),
                             hide_index=True, width="stretch")
                fp, fa = fl[fl.group == "positive_skew"], fl[fl.group == "all"]
                oe = ov[ov.group == "all_eligible"]
                reads((f"Flagged names returned {pc(fp.iloc[0][mg], 2)} {MAGW} vs {pc(fa.iloc[0][mg], 2)} for everything, best÷worst "
                       f"{f(fp.iloc[0].mfe_mae_ratio, 2)} vs {f(fa.iloc[0].mfe_mae_ratio, 2)}. " if len(fp) and len(fa) else "")
                      + (f"Holding winners another {h} sessions while above the 10-day SMA averaged {pc(oe.iloc[0].overlay_ret, 2)} vs "
                         f"{pc(oe.iloc[0].fixed_ret, 2)} for exiting on schedule." if len(oe) else ""),
                      "OOS slices; low-confidence = Isolation Forest score above the training-fold 95th percentile. The hold rule is absolute "
                      "return, decided at the exit close, and excludes low-confidence rows.")
            with b:
                st.markdown("### Score IC · daily, out-of-sample")
                ic = D["ic_oos"][D["ic_oos"]["horizon"] == h]
                labmap = {"y_mag": "vs sector" if REL else "return", "y_mfe": "best move", "y_mae": "worst move"}
                st.dataframe(pd.DataFrame({"score": ic.score.str.replace("p_up", PN).str.replace("mag_q50", "median").str.replace("mfe_q50", "best move q50").str.replace("mae_q50", "worst move q50"),
                                           "vs": ic.label.map(lambda x: labmap.get(x, x)), "IC": ic.mean_ic.round(3), "IR": ic.ir.round(2)}),
                             hide_index=True, width="stretch")
                icp = ic[ic.score == "p_up"]
                reads((f"How well each score lined up with what happened, day by day. {PN} scored {f(icp.iloc[0].mean_ic)}; above ~0.05 is strong "
                       "for stock ranking. Best/worst-move rows run higher because volatility is easier to predict than direction.") if len(icp) else "",
                      "Mean daily Spearman between score and realized label, OOS; IR = mean/std. Overlapping labels overstate significance for h > 1.")

            st.markdown("### Tails (kurtosis) · predicted vs realized")
            tb_ = D["tails"][D["tails"]["horizon"] == h] if len(D["tails"]) else pd.DataFrame()
            if len(tb_):
                a, b = st.columns(2)
                with a:
                    lo = min(.8, tb_.pred_tail.min(), tb_.realized_tail.min()) - .03
                    hi = max(1.2, tb_.pred_tail.max(), tb_.realized_tail.max()) + .03
                    fig = base(T, 300)
                    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=T["neutral"], width=1.5), name="predicted = realized", hoverinfo="skip")
                    fig.add_scatter(x=[lo, hi], y=[1, 1], mode="lines", line=dict(color=T["axis"], width=1), name="normal (1.0)", hoverinfo="skip")
                    fig.add_scatter(x=tb_.pred_tail, y=tb_.realized_tail, mode="markers+text", text=[f"Q{b_}" for b_ in tb_.bucket], textposition="top right",
                                    textfont=dict(color=T["text"]), marker=dict(color=T["s3"], size=13, line=dict(color=T["panel"], width=1.5)),
                                    name="tail-weight quintile", hovertemplate="predicted %{x:.2f} → realized %{y:.2f}<extra></extra>")
                    fig.update_xaxes(range=[lo, hi], title_text="predicted tail weight (bucket mean)")
                    fig.update_yaxes(range=[lo, hi], title_text="realized tail weight")
                    show(fig, "tail")
                with b:
                    fig = base(T, 300)
                    fig.add_bar(x=[f"Q{b_}" for b_ in tb_.bucket], y=tb_.big_move_share * 100, marker=dict(color=T["s3"], cornerradius=4),
                                hovertemplate="%{y:.1f}% big moves<extra></extra>")
                    fig.update_yaxes(ticksuffix="%", title_text="big-move share")
                    fig.update_xaxes(title_text="predicted tail-weight quintile (Q5 = fattest)")
                    fig.update_layout(showlegend=False)
                    show(fig, "big")
                q1, q5 = tb_.iloc[0], tb_.iloc[-1]
                rising = q5.realized_tail > q1.realized_tail + .03
                reads("Kurtosis is how often a name makes an unusually large move relative to its normal range. Left: did the fattest-tailed group "
                      f"(Q5) really have fatter tails afterwards? " + (f"Yes — {f(q1.realized_tail, 2)} (Q1) to {f(q5.realized_tail, 2)} (Q5)." if rising else
                      f"Not clearly here ({f(q1.realized_tail, 2)} → {f(q5.realized_tail, 2)}), so treat the Tails column as descriptive.")
                      + f" Right: share of big moves, {pp(q1.big_move_share)} in Q1 vs {pp(q5.big_move_share)} in Q5. With positive skew a fat right "
                        "tail is the payoff; fat tails with negative skew are the setups that gap against you.",
                      "Per-date quintiles of predicted quantile kurtosis [(q90−q10)/(q75−q25)]/1.90; realized tail weight per bucket; big move = "
                      "> 2 × IQR from the horizon median. Quantile kurtosis is robust but blind beyond the 10th/90th percentiles; moment kurtosis enters as `kurt60`.")

            st.markdown("### Direction scorecard")
            allm = st.toggle("Show all models", value=False, key="allm")
            dr = D["direction"][(D["direction"]["horizon"] == h) & (D["direction"]["model"].eq("ensemble") | allm)]
            st.dataframe(dr[["model", "regime", "n", "auc", "auc_xs", "brier", "brier_skill", "logloss"]].rename(
                columns={"auc_xs": "AUC same-day", "auc": "AUC", "brier_skill": "Brier skill"}).round(4), hide_index=True, width="stretch")
            de = dr[(dr.model == "ensemble") & (dr.regime == "all")]
            reads((f"AUC 0.50 is a coin flip; {f(de.iloc[0].auc)} here. In stock ranking 0.52–0.55 across hundreds of names is usable; above 0.60 "
                   "is suspicious. The HV-regime rows show whether the edge lives in calm or volatile tapes.") if len(de) else "",
                  "Pooled AUC, per-date AUC, Brier and log loss, overall and by the stock's own HV20-percentile regime.")

    # ============================================================================== drivers
    with tabs[3]:
        S = D["shap"].get(h, {})
        a, b = st.columns(2)
        with a:
            st.markdown("### SkewSHAP beeswarm · every dot is a name")
            bee = S.get("beeswarm")
            if bee is not None and len(bee):
                feats = list(dict.fromkeys(bee.feature))
                xs_, ys_, cs_, tx_ = [], [], [], []
                for i, fe in enumerate(feats):
                    g = bee[bee.feature == fe].sort_values("shap")
                    v = g.shap.to_numpy() * 100
                    w = (v.max() - v.min()) / 50 or 1
                    bins = np.round((v - v.min()) / w).astype(int)
                    cnt = {}
                    for vv_, bb, xx in zip(v, bins, g.x.to_numpy()):
                        n = cnt[bb] = cnt.get(bb, 0) + 1
                        off = (1 if n % 2 else -1) * np.ceil((n - 1) / 2) * .07
                        xs_.append(vv_); ys_.append(len(feats) - 1 - i + float(np.clip(off, -.42, .42))); cs_.append(xx); tx_.append(nm(fe))
                fig = base(T, 440)
                fig.add_scatter(x=xs_, y=ys_, mode="markers", text=tx_, hovertemplate="%{text}: %{x:+.3f} pp<extra></extra>",
                                marker=dict(color=cs_, colorscale=[[0, T["s2"]], [.5, T["neutral"]], [1, T["s1"]]], cmin=0, cmax=1, size=5,
                                            colorbar=dict(title=dict(text="feature value", font=dict(color=T["mute"])), tickvals=[0, 1],
                                                          ticktext=["low", "high"], thickness=10, len=.6, tickfont=dict(color=T["mute"]))))
                fig.update_yaxes(tickvals=list(range(len(feats))), ticktext=[nm(x) for x in feats[::-1]], tickfont=dict(color=T["ink2"]),
                                 gridcolor="rgba(0,0,0,0)")
                fig.update_xaxes(title_text="SkewSHAP (pp of tail asymmetry)")
                fig.update_layout(showlegend=False)
                show(fig, "bee")
                sk = S.get("skew")
                ups = [nm(x) for x in sk.index[sk.skew_mean > 0][:2]] if sk is not None else []
                dns = [nm(x) for x in sk.index[sk.skew_mean < 0][:2]] if sk is not None else []
                reads(f"Each row is an input, each dot a name. Right = widened the upside relative to the downside; left = the opposite. Colour is the "
                      f"name's own value on that input ({C1N} = high vs peers, {C2N} = low), so {C1N} dots on the right read as 'being high on this "
                      f"helps upside skew'. On average, upside comes from {' and '.join(ups) or '—'}; downside from {' and '.join(dns) or '—'}.",
                      "SHAP summary of local SkewSHAP on 250 training rows of the live quantile models; top 10 by mean |SkewSHAP|. Colour = feature "
                      "value scaled within the sample (diverging, grey at mid); dots stacked within 50 bins per row. Colour sorted along x = monotone effect.")
        with b:
            st.markdown("### Global direction SHAP · GBM")
            gb = S.get("gbm")
            if gb is not None:
                gb = gb.head(12)
                hbars(T, gb.index, gb.mean_abs, signed=False, unit="log-odds", height=440, key="gbm")
                reads(f"Inputs that move the direction call most: {', '.join(nm(x) for x in gb.index[:3])}. Size of influence, not direction.",
                      "Mean |TreeSHAP| on the live LightGBM classifier's log-odds. Correlated features share credit.")
        a, b = st.columns(2)
        with a:
            dep = S.get("dependence")
            if dep is not None and len(dep):
                fe = dep.feature.iloc[0]
                dd_ = dep[dep.feature == fe].sample(min(600, (dep.feature == fe).sum()), random_state=1)
                st.markdown(f"### Dependence · {nm(fe)} (top skew driver)")
                fig = base(T, 320)
                for flag, name, col, sz in ((0, "normal", T["mute"], 5), (1, "post-climax (≤5 sessions after extreme volume)", T["s1"], 7)):
                    q_ = dd_[dd_.post_climax.fillna(0) == flag]
                    fig.add_scatter(x=q_.x, y=q_.shap * 100, mode="markers", name=name,
                                    marker=dict(color=col, size=sz, opacity=.55 if not flag else 1, line=dict(color=T["panel"], width=1)))
                fig.update_xaxes(range=[0, 1], title_text="peer percentile (0 = lowest in group, 1 = highest)")
                fig.update_yaxes(title_text="SkewSHAP, pp")
                show(fig, "dep")
                reads(f"Left = low on {nm(fe)} versus peers, right = high; points above zero widen the upside. {C1N.capitalize()} points are "
                      "post-climax setups — if they sit apart from the grey ones, the volume event changes how the feature works.",
                      f"Peer percentile of {fe} vs local SkewSHAP on a training sample, coloured by post_climax.")
        with b:
            st.markdown("### IC gate · which features passed, fold by fold")
            g_ = D["gates"][D["gates"]["horizon"] == h] if len(D["gates"]) else pd.DataFrame()
            if len(g_):
                sets = [set(str(x).split("|")) for x in g_.survivors.fillna("")]
                cnt = pd.Series([f_ for s_ in sets for f_ in s_ if f_]).value_counts() / len(sets)
                feats = cnt.index[:20].tolist()
                z = [[1 if f_ in s_ else 0 for s_ in sets] for f_ in feats]
                folds = [str(x)[:7] + (" †" if bool(rx) else "") for x, rx in zip(g_.test_start, g_.gate_relaxed)]
                fig = base(T, 440)
                fig.add_trace(go.Heatmap(z=z, x=folds, y=[f"{nm(f_)} · {cnt[f_] * 100:.0f}%" for f_ in feats], showscale=False, xgap=3, ygap=3,
                                         colorscale=[[0, T["grid"]], [1, T["s1"]]], zmin=0, zmax=1,
                                         hovertemplate="%{y} · fold %{x}: %{z}<extra></extra>"))
                fig.update_yaxes(autorange="reversed", tickfont=dict(color=T["ink2"]), gridcolor="rgba(0,0,0,0)")
                fig.update_xaxes(title_text="retraining (test start)", gridcolor="rgba(0,0,0,0)")
                show(fig, "gate")
                reads("Each column is one retraining; a filled square means that input passed the screen and was used. Solid rows are trusted in "
                      "every period; rows that fill in only recently, or drop out, show where the market changed what works.",
                      "Features × folds, top 20 by pass frequency (share in the label); † = gate fell back to top-10 by |t|. Gate: |mean daily IC| ≥ 0.015, "
                      "|IR| ≥ 0.15, quarterly sign stability ≥ 55%, any core target or HV/post-climax condition.")
            else:
                st.info("Run the full backtest to see the gate history.")

    with tabs[4]:
        run_tab(T, out_dir)


def _alpha(hex_color: str, a: float) -> str:
    h_ = hex_color.lstrip("#")
    r_, g_, b_ = (int(h_[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r_},{g_},{b_},{a})"


main()
