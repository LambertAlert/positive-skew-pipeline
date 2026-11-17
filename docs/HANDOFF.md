# Positive-Skew Pipeline v0.3 — Massive + Luk universe, sector-relative targets (Nov 14 2026)

Neil uploaded `Positive_Skew_Pipeline.zip` (v0.1 "locked") for review. The review found that it was a
scaffold with bugs that bias results. Neil asked for the fixes, Massive data, and the Luk screeners as
the universe with **close > $9**.

## What v0.1 got wrong (all fixed)

**Leakage and labels:**
- The last *h* rows' labels were 0 instead of NaN, and MFE/MAE came from partial windows near the end.
- `hv20_pct` was a full-history rank, so it looked ahead.
- The harness used one 70/30 split with no purge; `walk_forward_splits` was never called.

**Skew:**
- SkewScore came from quantiles of `mfe_5` (dispersion of MFE, not up vs down).
- It had no floor (scores up to 51), and quantiles could cross.

**IC screen:**
- It computed one pooled correlation, not a per-date IC, so IR and stability were impossible.
- It ran on the full sample and never actually gated the model inputs.

**Models and SHAP:**
- No calibration; unscaled logistic regression; mismatched NaN handling between training and scoring.
- SHAP silently returned None for RF and for the wrapper classes.

**Features and dashboard:**
- The AVWAPs weren't anchored, and many README features were missing.
- The dashboard numbers were hard-coded and inconsistent: T0 SkewScore 4.44 vs 1.45 from its own quantiles, and tickers that don't exist.

## v0.2 layout (`pipeline/`)

| Module | Role |
|---|---|
| config.py | every threshold; `LukFilters` (min_price 9.0) |
| massive_client.py | grouped daily bars (1 call/session, parquet cache), CS+ADRC stock set incl. delisted, ticker details (shares, SIC) |
| universe.py | Luk Premarket ∪ Leader ∪ Potent, point-in-time (shifted 1); SIC peer groups; `append_next_session` for live scoring |
| features.py | anchored VWAPs, gaps, volume/climax, OC, momentum (IBD RS, RS-line 8W, 1st and 2nd derivatives), MRV, vol regime; delay-1; peer ranks |
| targets.py | entry at the open of t; labels NaN unless the window is complete; hi/lo MFE/MAE; still-strong overlay |
| ic_screen.py | daily cross-sectional IC (vectorized); gate on fit dates only; regime + post-climax conditions; interaction search |
| models.py | logit+interactions, RF, LGBM, bin-isotonic; isotonic calibrators; LGBM quantile sets (rearranged); IsolationForest |
| harness.py | purged walk-forward `[fit] P [calib] P [test]`, P = max(21, h+1); `fit_live` |
| evaluate.py | Brier/skill, AUC + cross-sectional AUC, calibration, decile lift, pinball/coverage, skew buckets, flags, overlay, OOS IC |
| explain.py | TreeSHAP (any shape), linear SHAP, SkewSHAP = S(q90)+S(q10)−2S(q50), interactions, dependence |
| dashboard.py | HTML from the run's own outputs, War Room styling |
| run.py | CLI: `python -m pipeline.run [--live-only] [--horizons ..] [--source synthetic]` |

Also included: `tests/test_pipeline.py` (17 tests, including perturbation leakage, noise and planted-signal tests) and `.github/workflows/positive_skew.yml` (weekly full run, daily live-only).

## Verified here

- The tests pass.
- **Pure random walk:** OOS IC of P(up) was −0.024 and −0.005 on two seeds, i.e. no leak (a leak would show up as a positive IC).
- **Planted signal:** OOS IC was +0.082.
- On synthetic data, the pooled AUC stays near 0.5 while the cross-sectional IC is positive, because the absolute-return label carries market beta. That's why `auc_xs` was added.

## Not verified

- **It hasn't run on real Massive data.** The sandbox can't reach api.massive.com. The endpoint shapes were checked with the Massive connector (the grouped bars return T, v, vw, o, c, h, l, t, n; ticker details have `weighted_shares_outstanding` and `sic_code`).
- **Runtime of the full run:** expect hours for 5 horizons; use `--test-sessions 126` in Actions.

## Open

- **First real run:** check the universe size per day (expect several hundred to ~1,500 names), the share of folds with a relaxed gate, and the skew-bucket monotonicity. That last one is the real test of the idea.
- **Market cap** uses current share counts; consider quarterly as-of-date details if the cap filter matters.
- **The direction label** may be better as `dir_excess` given the beta noise.

## v0.3: sector-relative targets (Nov 14, later)

Neil asked to train on relative moves instead of absolute ones. The reason: no stock-level feature
can call the market or the sector, and the regime model already decides whether to be active.

**Targets:**
- `rel_h` = return minus the sector ETF over the same window. The ETF comes from the SIC code via `universe.sic_to_etf` (the 11 SPDRs), with SPY as the fallback.
- `dir_xs_h` = beat the day's median universe member on `rel_h`. That's 50% every day by construction.
- Direction models predict `dir_xs`, and the quantiles (now `mag_q10/50/90`, so SkewScore too) are of `rel_h`. MFE and MAE stay absolute.
- `--target-mode absolute` restores v0.2.

**Data:** the SPDR ETFs come from the same grouped bars; `load_market` returns `etfs`.

**Overlay (`overlay.py`):** live scores read `data/narrative_regime.parquet` (`posture`, plus `entry_mode` if present) and add `trade_view`:

| Posture | View |
|---|---|
| BUY_RIP, BUY_DIP | ACTIONABLE |
| NEUTRAL | SELECTIVE |
| AVOID, or entry mode CASH | BEST OF WEAK TAPE |
| no snapshot | NO REGIME DATA |

The dashboard shows the posture banner and a View column. The overlay isn't backtested.

**Evaluation** uses `y_dir`/`y_mag` (the trained labels); absolute return is shown alongside.

**Same synthetic market, h=5, absolute vs relative** (shared market and sector factors plus a planted idiosyncratic signal):

| | absolute | relative |
|---|---|---|
| pooled AUC | 0.485 | 0.529 |
| Brier skill | −2.0% | +0.2% |
| calibration | inverted (higher P, lower hit) | monotone |
| OOS IC | 0.059 | 0.063 |

- Decile lift vs sector runs monotone from −0.66% (D1) to +0.68% (D10).
- Skew buckets: MFE/|MAE| rises from about 1.0 to 1.25 in the top bucket.
- These are synthetic numbers: they show the mechanics, not market evidence.

**Tests:** 21 pass. New ones cover the SIC→ETF map, a hand-computed `rel_h` with the SPY fallback, `dir_xs` at 50% per day, and the overlay rules.

**Open:**
- Point `--regime-path` at the War Room repo's `data/narrative_regime.parquet` when running outside it.
- Spot-check the SIC→ETF map on real names (ADRs and conglomerates).

## v0.3.1: dashboard redesign (Nov 15)

- Every panel now carries a Trader read and a Data scientist read, toggled in the header. Several are computed from the run's numbers.
- New: per-sector top-3 cards (grouped by benchmark ETF), inline range bars, a large range bar in the drill-down, a dependence plot coloured by post-climax (fixed: colour keys now come from the panel, not the design matrix), and coverage as target-vs-realized bars.
- The skew buckets are split into two charts (no dual axis), and the ratio bars start at 1.0.
- The palette was validated for the dark surface (#c97f17 / #5b84d6 / #2a9d7e).
- Mobile layout fixed.
- `bench_etf` now flows into the live and OOS outputs.
- `python -m pipeline.run --dashboard-only` rebuilds from saved files; `meta.json` is written on each run.

## v0.3.2: themes, sort, search (Nov 15)

- Neil didn't like the amber/copper scheme on this dashboard. There are now four themes: Midnight (navy; blue/orange/violet; Inter + JetBrains Mono; the default), Graphite (neutral dark), Paper (light) and War Room (the old look). Each palette was validated against its own background.
- The reads name the active theme's colours.
- Dependence plot: non-climax points are grey, so post-climax points stand out.
- The shortlist sorts by any column and has a ticker search.

## v0.3.3: chart forms (Nov 15)

Neil asked for better-fitting visuals than bars everywhere.

**New panels:**
- Upside-vs-downside map (iso-SkewScore lines, click a dot to drill).
- Sector strip (every name by P(beat) per ETF, median tick, click a dot).
- Edge over time (monthly D10−D1 spread plus a 6-month line).

**Replaced:**
- Local SHAP bars → waterfalls. `skew_base`/`skew_total` and `dir_base`/`dir_total` are now in the live outputs; the ends add up (tested).
- Skew mean-return bars → box plot. `p10..p90` were added to `metrics_skew_buckets`.
- Coverage bars → dumbbell.
- Global SkewSHAP bars → beeswarm (`shap_beeswarm_h*.csv`).
- Survival bars → features × folds heatmap.

**Also:** calibration gets binomial whiskers; map labels skip collisions; `metrics_edge.csv` is new.

24 tests pass. Verified in all themes, at phone width, and with click-through.

## v0.3.4: horizons and export (Nov 15)

- **Shortlist:** a Horizons column (filled = flagged, ring = P ≥ .5 unflagged, outline = current horizon), sortable by agreement. New KPI: names flagged on 3+ horizons.
- **Drill-down:** range by holding period (fan across 3–21D, x spaced by sessions) and SkewScore / P(beat) by horizon (small multiples). The reads name the peak horizons.
- **Export:** Copy tickers (comma list, with a clipboard fallback) and Download CSV of the current filtered view, including horizons_flagged.
- **Phone:** sector cards become a horizontal swipe row.
- **Not done yet (offered earlier):** industry-level benchmarks such as SMH for semis and XBI for biotech.

## v0.3.5: kurtosis (Nov 16)

**Model:**
- The return quantile model now fits q10/q25/q50/q75/q90 (two extra LightGBM fits per fold), giving a per-name tail weight = [(q90−q10)/(q75−q25)]/1.90 (normal = 1).
- New feature `kurt60`: 60-session excess kurtosis of daily returns.
- `metrics_tails.csv`: per-date quintiles of predicted tail weight vs realized tail weight and big-move share (beyond 2×IQR of the horizon median).
- Realized tail weight was added to the skew buckets and the flag summary.

**Dashboard:**
- Shortlist: Tails column (fat/normal/thin), and a q25–q75 box inside each range bar.
- Drill-down: tail weight in the header and a box in the big bar; tails by holding period as a third mini chart.
- New "Tails · predicted vs realized" card; Tails column in the flags table; tail_weight in the CSV export.
- The q25/q75 quantile coverage rows appear automatically.

**Caveat:** quantile kurtosis is robust but doesn't see beyond the 10th/90th percentiles (a Student-t with 3 degrees of freedom scores ≈1.13). Tail weight is not part of the flag.

**Possible next:** q05/q95 for sharper tails; a TailSHAP waterfall, (q90−q10) − 1.9(q75−q25), which is additive.

## v0.3.6: short Massive history (Nov 16)

Neil said his Massive plan has 2 years.
- **What the connector shows:** grouped daily bars reach back to 2021-11-23 (2021-11-16 is NOT_ENTITLED), which is a rolling 5 years. His 2-year limit may be another data type or another key.
- **Bug this exposed:** the old default (start 2021-01-04 + 290-session warm-up) would have requested 2019 and crashed on the 403.

**Fixes:**
- `MassiveClient.earliest_available`: a binary-search probe, cached in `plan_history.json`; 403 raises `MassiveForbidden`.
- `--calls-per-minute` pacing.
- `pipeline/backfill.py`: batched yfinance download, auto_adjust, tz_localize(None), BRK.B → BRK-B; splice scaled to Massive over 10 overlap sessions; `source` column.
- `--backfill auto|yfinance|none`, `--backfill-use warmup|full` (default warmup: nothing is trained or scored on survivor-only yfinance dates). OOS metrics are restricted to Massive dates.
- `Config.apply_profile`: a short walk-forward profile below 700 sessions (minimum training 40% of dates, at least 126; 42-session tests). 2 years → 7 folds instead of 4.

30 tests pass. yfinance was added to requirements. The real yfinance path wasn't exercised here (no network), only mocked.

## v0.3.7: Massive-only defaults + launcher (Nov 16)

Neil asked for the file to run with Massive.
- Defaults are now `start="auto"` (the plan's earliest grouped date + 265 sessions) and `backfill="none"`; yfinance is opt-in with `--backfill auto`.
- 403 days inside a stale cached window are skipped and not cached.
- New `run_massive.py` at the package root: prompts for the key (not saved) if no env var is set, passes every option through to `pipeline.run`, chdirs to its own folder, and opens the dashboard.
- The workflow no longer hard-codes `--start`. 32 tests pass.
- On the connector's 5-year window (from 2021-11-23), auto start scores from about late 2022, roughly 4 years.

## v0.4.0: Streamlit dashboard (Nov 17)

Neil asked for a Python file that creates a Streamlit dashboard from VS Code.
- `psp_dashboard.py` (Streamlit ≥1.50 + Plotly) reads `data/psp_out` and ports every panel and read.
  - Tabs: Shortlist, Name drill-down, Does it work?, What drives it, Run.
  - Selection: dataframe row selection and Plotly point selection set `st.session_state.sel`.
  - Plotly specifics: SHAP waterfalls use `go.Waterfall` (order kept, y reversed); skew boxes use a `go.Box` built from precomputed quartiles; the gate heatmap uses `go.Heatmap`.
  - Reads use Streamlit `:blue[]` / `:violet[]` markdown (no HTML inside columns, per the War Room rule).
- **Run tab:** starts `python -m pipeline.run` with Popen into `out/run.log`, tracking the pid in `run.pid`; Refresh clears the cache.
- **Setup files:** `.streamlit/config.toml` (dark base, Midnight colours) and `.vscode/launch.json` (F5 launches the dashboard or the full run).
- **Tested:** AppTest across widgets (no exceptions), an empty-folder state, the Run tab launching a synthetic run, and a live-server screenshot of every tab.
- **Not done:** the Paper theme over Streamlit's dark base leaves widgets dark (the user switches Streamlit to Light); st.context theme detection was unreliable.
