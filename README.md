# Positive-Skew Path Prediction Pipeline

Stock-only, delay-1 research pipeline on Massive daily data: the Luk-screener universe ($9+), sector-relative targets, a purged walk-forward backtest, positive-skew and tail-weight scoring, SHAP explanations, and two dashboards (a standalone HTML page and a Streamlit app).

```bash
git clone https://github.com/LambertAlert/positive-skew-pipeline.git
cd positive-skew-pipeline
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests/                              # 32 tests
python run_massive.py                                   # full run on Massive (asks for the key)
streamlit run psp_dashboard.py                          # dashboard (or F5 in VS Code)
```

Run outputs go to `data/` and are git-ignored. `docs/HANDOFF.md` has the design history, and `docs/dashboard_SYNTHETIC_demo.html` is a demo built on synthetic data.

---

Stock-only, delay-1 system on **Massive** daily data. The universe is **Martin Luk's three morning
screeners** (Premarket ∪ Leader ∪ Potent) with a **$9 price floor**. Peer-ranked features pass a
purged cross-sectional IC gate. Calibrated direction models and quantile path models then score
3/5/7/10/21-session horizons, with an explicit positive-skew flag, an Isolation Forest confidence
check and SHAP explanations.

**v0.3: the models predict sector-relative moves.** They learn which names will beat their sector
ETF, not whether the market goes up. The War Room regime posture is laid on top of the live scores
to say whether to act.

## Streamlit dashboard (VS Code)

```bash
pip install -r requirements.txt
streamlit run psp_dashboard.py        # or press F5 in VS Code: "Positive-skew dashboard (Streamlit)"
```

`psp_dashboard.py` reads a finished run from `data/psp_out/` (changeable in the sidebar) and draws every panel with Plotly:
- **Shortlist:** a sortable table; click a row to drill in. A copyable ticker list and a CSV download. The upside-vs-downside map and the sector strip (click a dot to drill in). Best-in-sector cards.
- **Name drill-down:** the range bar with the q25–q75 box, range by holding period, skew/P(beat)/tails by horizon, and two SHAP waterfalls.
- **Does it work?** Decile lift, edge over time, calibration with error bars, coverage dumbbell, skew buckets (ratio + box plot), flagged-vs-not and the hold rule, score IC, tails check, direction scorecard.
- **What drives it:** SkewSHAP beeswarm, GBM importance, dependence plot, IC-gate heatmap.
- **Run:** starts the pipeline in the background (live-only, full backtest or the synthetic demo), takes the API key (not saved) and an optional regime snapshot, and shows the log.

The sidebar has the theme (Midnight, Graphite, Paper, War Room), the Trader / Data scientist reads, horizon, sector, show-all and ticker search. `.streamlit/config.toml` gives Streamlit's own widgets a dark base. For the Paper theme, also switch Streamlit to Light (⋮ → Settings).

The pipeline still writes the standalone `dashboard.html` as well.

## Quick start

```bash
pip install -r requirements.txt
python run_massive.py                      # full run on Massive; asks for the key if MASSIVE_API_KEY isn't set
python run_massive.py --live-only          # daily: score the next session (after one full run)
python run_massive.py --dashboard-only     # rebuild the dashboard from saved results
python run_massive.py --horizons 5 10 --test-sessions 126     # faster first run
python -m pytest -q tests/                 # 32 tests incl. leakage checks
```

`run_massive.py` is a thin wrapper around `python -m pipeline.run`: every option below passes through, and it opens the dashboard when it finishes.

**Defaults are Massive-only.** The first scored day is your plan's earliest date plus the ~260-session feature warm-up (`--start auto`), and nothing comes from yfinance. Other options:

```bash
python run_massive.py --start 2023-01-03                 # pick the first scored day yourself
python run_massive.py --backfill auto                    # also backfill the warm-up from yfinance (more scored history)
python run_massive.py --regime-path ../pivot-scanner/data/narrative_regime.parquet
python run_massive.py --target-mode absolute             # raw-return training (v0.2 behaviour)
python run_massive.py --calls-per-minute 5               # free tier
python -m pipeline.run --source synthetic                # offline smoke test (no key)
```

## Data (Massive)

| Endpoint | Use |
|---|---|
| `/v2/aggs/grouped/locale/us/market/stocks/{date}` (adjusted=true) | every US ticker's daily bar, 1 call per session, cached; includes the day's VWAP |
| `/v3/reference/tickers` type CS + ADRC, active **and delisted** | the stock set (ETFs, warrants and units are excluded), cached a week |
| `/v3/reference/tickers/{ticker}` | shares outstanding (for market cap) and SIC code (for peer groups and the sector ETF), cached |
| same grouped bars | SPY and the 11 SPDR sector ETFs (XLK XLV XLF XLE XLI XLY XLP XLB XLU XLRE XLC), the benchmarks |

- Bars are split-adjusted. Massive does not dividend-adjust daily aggregates.

### Plans with limited history
Massive plans differ in how far back daily bars go. The pipeline handles that on its own:
1. **Finds your plan's earliest date** with a binary search (about 12 calls, cached a week; out-of-plan days answer 403) and never asks for anything older.
2. **Optionally backfills the feature warm-up from yfinance** (`--backfill auto`; off by default) when Massive can't cover the ~260 sessions the 252-day features need. Without it, the first ~260 Massive sessions are used only for warm-up.
   - Only the candidate tickers, SPY and the sector ETFs are fetched, in batches with `auto_adjust=True`.
   - Each ticker's yfinance prices are scaled to Massive over the first 10 overlapping sessions. yfinance is dividend-adjusted and Massive isn't, so without this there'd be a fake jump at the join.
   - Rows are tagged `source="yfinance"`.
3. **Uses the backfill for warm-up only by default** (`--backfill-use warmup`): nothing is trained or scored on yfinance dates, because they only contain names alive when the Massive window starts (survivorship).
   - `--backfill-use full` also trains on them, but OOS metrics still count only Massive dates.
4. **Switches to a short walk-forward profile** when there are fewer than 700 scored sessions (`--profile auto`): the minimum training window is 40% of the dates (at least 126 sessions) and test blocks are 42 sessions. On 2 years that gives 7 folds instead of 4.

The free tier is rate-limited: add `--calls-per-minute 5`.
- Delisted names are included, so the universe isn't survivorship-filtered.
- The first full pull is about 1,500 grouped calls plus one details call per candidate ticker. Later runs only fetch new sessions.

## Universe — Luk screeners, point-in-time

Same thresholds as the lab's `LK_DEFAULTS`, except the price floor (**close > $9**; the lab's
whole-market panel used $1). Every field is measured on the completed session and shifted one day,
so membership on day *t* uses bars through *t−1* only.

| Screen | Filters |
|---|---|
| Premarket | close > $9; mcap > $100M; 10D avg vol > 500K; close×vol > $50M; vol 1W > 3.5% |
| Leader | close > $9; mcap > $100M; 10D avg vol > 500K; close×5D vol > $300M |
| Potent | close > $9; close > EMA50; close ≥ EMA21; mcap > $100M; vol 1W > 3.5%; close×vol > $50M; \|chg from open\| > 0.5% |

- **Pre-market change is a sort in his screens, not a filter**, so it plays no part in a daily universe.
- **Market cap** = today's share count × that day's close. Massive has no daily share history. With the $9 floor and $50M dollar-volume test, the cap filter rarely binds.
- **Unknown caps are kept** and marked (`mcap_known`).
- **Potent's industry list** isn't reproduced, as in the lab.

Thresholds are in `pipeline/config.py → LukFilters`. Change `screens` to use a single screener.

## Timing

Features on row *t* use bars through *t−1*. The trade enters at the **open of t**, and the
*h*-session window is days *t … t+h−1*. Set `entry="close"` to enter at the close of *t* instead.
Labels are NaN unless the full window exists. MFE and MAE use intraday highs and lows.

## Features (all delay-1, then peer-ranked)

Each feature is ranked as a percentile within date × SIC 2-digit group. Groups with fewer than 8
names fall back to the SIC division, then to the whole universe.

- **AVWAP:** anchored at the more recent of the N-session closing high/low (N = 21/63/126/252). Distance, anchor side, 5-day slope, stack count (0–4). Session VWAP distance uses Massive's daily VWAP.
- **Gaps:** gap, gap/ATR14, gap filled, gap × RVOL.
- **Volume:** RVOL (vs prior 21), log dollar volume, highest volume in 21/63/252 sessions, highest in the loaded history, days since climax (RVOL ≥ 3 or 63-session high volume), post-climax flag (≤5 sessions), post-climax drift.
- **Open-to-close:** today's, 5/21-day means, 10-day hit rate, OC × RVOL, close location in range.
- **Momentum:** 21/63/126/252, relative to SPY (63), IBD RS raw (0.4/0.2/0.2/0.2), 52-week-high proximity, RS line vs its 8-week high, distance to the 8-week high, 1st and 2nd derivatives of mom_21, momentum category score.
- **Mean reversion:** 5-day reversal, distance to 5/10/20 MA, Bollinger z (20), MRV category score.
- **Volatility:** HV5/20/60, vol-of-vol, Parkinson range vol, 60-session excess kurtosis (`kurt60`), **HV20 percentile vs the stock's own past 252 sessions (past-only)**, elevated flag, HV5 > HV20.
- **Calendar:** day of week, month.

## Targets

**Why relative.** On any day much of a stock's move is the market and its sector, and no
stock-level feature can call that. Training on raw up/down spends the models on noise; the score
swings with the tape, and pooled AUC sits near 0.5 even when the model ranks names well. The real
decision is which names to own today; the regime layer already decides whether to be active.

**What the models learn (default `target_mode="relative"`):**
- `rel_h` = the stock's return minus its **sector ETF's** return over the same window and entry. The ETF comes from the SIC code (e.g. 3674 semis → XLK, 2834 pharma → XLV, 6798 REITs → XLRE). SPY stands in when the code is unmapped or the ETF has no bar.
- `dir_xs_h` = 1 if `rel_h` beat the **median universe member that day**. That's exactly half the names every day, so the base rate never drifts and calibration holds across regimes.
- The direction models predict `dir_xs_h`, and the return quantiles (`mag_q10/50/90`, so SkewScore too) are of `rel_h`.
- **MFE and MAE stay absolute**, because those are what hit stops and targets.
- The IC gate screens against `dir_xs`, `rel`, MFE and MAE.

**Also computed, for context:** absolute return and sign, excess vs SPY, `rel_rank_h` (the day's
percentile), MFE, MAE, range, up/down hit flags at 0.5/1/2/3%, summed open-to-close return and its
sign, and the **hold-if-still-strong overlay** (absolute). At the exit close, the overlay holds *h*
more sessions if the close is above the 10-day SMA and the trade is green; it's evaluated
separately and never trained on.

`--target-mode absolute` restores raw-return training.

## Regime overlay (live)

Relative scores say "best of today's universe", not "this makes money". Live scores read the War
Room posture snapshot (`data/narrative_regime.parquet` from `narrative_regime_model.py`, path in
`Config.regime_path` / `--regime-path`). Each positive-skew name gets a `trade_view`:

| Posture | View |
|---|---|
| BUY_RIP, BUY_DIP | ACTIONABLE |
| NEUTRAL | SELECTIVE |
| AVOID, or entry mode CASH | BEST OF WEAK TAPE |
| no snapshot | NO REGIME DATA |

The overlay isn't backtested; the posture history is too short.

## IC screen

The IC is cross-sectional Spearman, **one IC per date**. Over a fold's training dates the screen
reports mean IC, std, IR, t, % positive and quarterly sign stability. Each is computed:
- unconditionally
- within low/mid/high HV20 regimes
- on post-climax rows

The daily IC series is computed once. Each fold aggregates **only its own fit dates**, so the gate
never sees test outcomes. A feature passes if, for any core target (direction, return, excess, MFE,
MAE) under any condition, all three hold: |mean IC| ≥ 0.015, |IR| ≥ 0.15, and sign stability ≥ 55%.
At most 30 features survive. A two-way search on the top survivors (centred-rank products) adds up
to 8 interactions that pass the same gate.

**If fewer than 3 features pass**, the fold falls back to the top 10 by |t| and logs `gate_relaxed`.

## Models (per horizon, per fold)

- **Direction:** standardized L2 logistic regression with the explicit interactions, random forest, LightGBM, and a peer-bin isotonic model on the best single feature.
  - Every model is **isotonic-calibrated on a purged slice at the end of the fold**.
  - The ensemble P(up) is the mean of the calibrated linear, RF and GBM outputs.
  - Disagreement is the std across all four.
- **Quantiles (primary for magnitude):** LightGBM quantile models for return q10/50/90, MFE q50/90 and MAE q10/50. They're rearranged per row so they can't cross.
- **Isolation Forest:** fit on the fold's features. Scores above the 95th percentile of training scores → confidence = low.

## Skew

```
SkewScore = (q90 − q50) / max(q50 − q10, floor),   floor = max(0.2%, 10% × (q90 − q10))
```

The quantiles are of the **forward return over the sector ETF** (the raw forward return in absolute mode). The path skew is q50(MFE) / max(|q50(MAE)|, 0.2%), on absolute MFE/MAE.

**Tails (kurtosis).** The return model now predicts five quantiles (q10/q25/q50/q75/q90, rearranged), which gives a quantile kurtosis per name:

```
tail_weight = [(q90 − q10) / (q75 − q25)] / 1.90      normal = 1.00 · fat ≥ 1.15 · thin ≤ 0.90
```

That is the outer 80% range relative to the inner 50% range. It's robust to single outliers and pairs with the quantile skew, but it's blind beyond the 10th/90th percentiles (a Student-t with 3 degrees of freedom scores only about 1.13). The extreme-tail side enters the model as a feature instead: `kurt60`, the 60-session excess kurtosis of daily returns (past-only, peer-ranked like the rest). Tail weight is reported, not used in the flag.

Positive-skew candidate = SkewScore > 1.2, calibrated P(up) ≥ 0.5, and confidence ≠ low.

## Explanation

- TreeSHAP on GBM and RF (handles any shap output shape) and exact linear SHAP.
- **SkewSHAP = SHAP(q90) + SHAP(q10) − 2·SHAP(q50)**, the additive attribution of (upper − lower) tail width. SHAP(q90) − SHAP(q10) would attribute spread, not asymmetry.
- GBM interaction values, dependence samples coloured by `hv_elevated` and `post_climax`, and the top 6 local drivers for every live row.

## Harness

```
[ fit ] purge [ calib ] purge [ test 63 ]     purge = max(21, h + 1) sessions
```

- Expanding window by default; rolling is an option.
- Every learned step happens inside the fold: the gate, interactions, models, calibrators, quantiles and anomaly threshold.
- Training rows are sampled to 400K per fold for speed.

**Out-of-sample metrics:**
- **Direction:** Brier score and Brier skill vs base rate, log loss, AUC, and **cross-sectional AUC** (per-date, so market up/down days don't dominate). Each is reported overall and by HV regime.
- **Calibration:** a calibration table.
- **Ranking:** per-date decile lift.
- **Quantiles:** pinball loss and skill vs an expanding unconditional quantile, plus coverage.
- **Skew:** realized MFE/|MAE| and Bowley skew by SkewScore bucket.
- **Flags:** flagged vs not flagged.
- **Overlay:** the hold-if-still-strong overlay vs fixed exits.
- **IC:** the predictions' own daily OOS IC.

## Dashboard (`dashboard.html`)

Built from the run's own files. Controls run along the top: horizon, sector (benchmark ETF), flagged or all names, and which reads to show.

Sections, and the chart used for each:

| Panel | Chart | Why this form |
|---|---|---|
| Shortlist | table with inline range bars and **Horizons** squares (3/5/7/10/21D flags); sortable, searchable; **Copy tickers** (comma list for TradingView/Barchart) and **Download CSV** of the current view | many names × a few fields; agreement across horizons is a sturdier signal than one flag |
| Upside vs downside map | scatter: downside width vs upside width, iso-skew lines at 1.0 and 1.2 | SkewScore is a ratio of two widths; the scatter shows both, and how far past the line a name sits |
| Sector strip | strip plot: one row per sector ETF, one dot per name, median tick | the spread inside each sector, before picking from the cards |
| Best in each sector | top-3 cards | the short list per sector |
| Drill-down | big range bar; **range by holding period** (fan of q10/q50/q90 plus typical best/worst move across horizons); **SkewScore and P(beat) by horizon** (two small charts); two **SHAP waterfalls** (skew, direction) | which holding period suits the name, and how it got from the typical value to its own |
| Decile lift | bars | ordered buckets, one measure |
| Edge over time | monthly top-minus-bottom spread (bars, warm/cool sign) + 6-month line | whether the edge is steady or came from one stretch |
| Calibration | reliability scatter with binomial 95% whiskers | separates real misses from small-sample noise |
| Tails (kurtosis) | scatter of predicted vs realized tail weight by quintile (diagonal = calibrated), plus big-move share by quintile | does predicted fat-tailedness show up afterwards? |
| Range honesty | dumbbell: target vs realized coverage per quantile | the gap is the point |
| Skew buckets | best÷worst bars starting at 1.0, and a **box plot** of realized outcomes per bucket | skew is a distribution's shape; a box shows it, a mean doesn't |
| Flagged vs not, score IC, direction scorecard | tables | exact figures to compare |
| Skew drivers | **SHAP beeswarm** (dots coloured by feature value) | direction, size, spread and monotonicity in one view |
| Direction drivers | ranked bars of mean \|SHAP\| | magnitude ranking |
| Dependence | scatter coloured by post-climax | the shape of the top driver's effect |
| IC gate | heatmap: features × folds | when inputs entered or dropped out, not just how often |

The map and strip dots open the drill-down when clicked.

Every panel has a **Trader** read and a **Data scientist** read. Several are generated from the run's numbers (decile spread, coverage, calibration gap, whether skew buckets rise), so they change with the data.

**Themes** (switch in the header, remembered per browser): Midnight (default), Graphite, Paper (light) and War Room. Each theme's data colours passed the colour-blind/contrast validator against its own background. Positive vs negative always uses the theme's warm/cool pair. Green/red are only for confidence status, always with a ✓/✕ label. The reads name the active theme's colours.

The shortlist columns sort when you click a header, and there's a ticker search box. The page works down to phone width.

## Output schema (`live_scores.csv`, `oos_predictions.parquet`)

| Columns | What |
|---|---|
| `date, ticker, horizon, sector, hv_regime, post_climax, scr_*` | identity, regime and the screens it passed |
| `p_up, p_linear, p_rf, p_gbm, p_bin_iso, disagreement` | calibrated direction probabilities |
| `ret_q10/q50/q90, mfe_q50/q90, mae_q10/q50` | quantiles |
| `skew_score, path_skew, bowley, positive_skew` | skew |
| `anomaly, confidence, still_strong_eligible` | confidence |
| `skew_drivers, dir_drivers` | top local SHAP drivers (live only) |
| `regime_posture, trade_view` | War Room overlay (live only) |
| `y_dir, y_mag` | the realized labels the models were trained on (OOS only) |
| `y_*` | every other realized label (OOS only) |

## Tests (`tests/test_pipeline.py`, 32)

- **Perturbation leakage tests:** changing day *t*'s bar moves nothing dated ≤ *t*, for both the universe and the features. A test also checks the change does show up the next day.
- **Universe:** the $9 floor is strict; small caps are dropped and unknown caps kept.
- **Features:** HV20 percentile is past-only; the AVWAP anchor is correct.
- **Targets:** hand-computed returns and MFE/MAE; incomplete windows give NaN labels.
- **Overlay:** the hold rule.
- **IC and folds:** the daily IC matches scipy's Spearman; every fold's purge holds.
- **End to end:** **no skill on a pure random walk**; a planted signal is recovered.
- **Skew and SHAP:** the skew floor and flag logic; RF and SkewSHAP return values.
- **Massive client** (offline mocks): parsing, 429 retry, caching, pagination.
- **Short plans:** the default config is Massive-only with an auto start (no yfinance call); out-of-plan days are skipped, not cached; the plan probe finds the earliest day in under 16 calls and caches it; the yfinance splice has no jump at the join; `load_market` backfills end to end on a fake 300-session plan; the short profile settings.
- **Kurtosis:** tail weight = 1 on normal data, > 1 on Student-t, < 1 on uniform; the five quantiles are ordered; `kurt60` is built.
- **Dashboard metrics:** the monthly edge series, bucket quantiles, and SHAP waterfall ends adding up (base + all SkewSHAP = raw q90 + q10 − 2·q50).
- **Relative targets:** the SIC→ETF map, a hand-computed `rel_h` (with the SPY fallback), `dir_xs` at exactly 50% per day, and the overlay rules.

## Known limits

- Market cap uses current share counts (see Universe).
- The SIC→ETF map is coarse in places (conglomerates, ADRs filed under odd codes); it's a table in `universe.py`.
- Relative scores don't say whether a trade makes money in absolute terms; that's the overlay's job, and the overlay isn't backtested.
- The IC t-stat overstates significance for h > 1 because the labels overlap; the gate relies on IR and sign stability instead.
- Runtime: a full 5-horizon walk-forward on ~5 years of the whole-market universe takes hours (LightGBM quantile fits dominate). `--test-sessions 126`, fewer `--horizons` or a lower `--max-train-rows` cut it.
