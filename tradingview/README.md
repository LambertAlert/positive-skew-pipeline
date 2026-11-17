# TradingView indicators (Pine Script v6)

| File | What it draws |
|---|---|
| `wr_avwap.pine` | Anchored VWAPs from the 52-week high, the 63-bar low and the last gap-on-volume event (War Room lab rules, after Brian Shannon); the pipeline's AVWAP (more recent N-bar closing extreme); session VWAP on intraday charts; Reclaim / Pullback / Event-pullback / Pinch markers with buy-stop, stop and risk in the tooltip; alerts; a state table |
| `psp_setup_read.pine` | A table of the pipeline's per-stock reads: the Luk screens ($9 floor), RS vs the sector ETF, momentum and its acceleration, HV20 percentile, 60-bar realized skew (Bowley and moment), tail weight and excess kurtosis, climax volume; climax markers; alerts |

**Install:**
1. In TradingView, open **Pine Editor** at the bottom of a chart.
2. Make a new indicator, select everything and paste one file.
3. **Save**, then **Add to chart**. Do it again for the other file.

**Use on daily charts.** In PSP Setup Read, set **Sector ETF** to the name's benchmark (semis/software XLK, pharma XLV, banks XLF …), the same mapping the pipeline uses.

**What Pine can't do:** the model's own outputs (P(beat), SkewScore, predicted ranges, the horizons) come from the Python pipeline. Paste the dashboard's *Copy tickers* list into a TradingView watchlist and step through it with these two indicators on the chart.

**Differences from the Python side:**
- **Bar price:** TradingView's daily bars carry no VWAP, so the lines use hlc3. The pipeline uses Massive's bar VWAP, so lines differ slightly.
- **Kurtosis:** the population formula, where pandas uses the sample formula. Values differ a little on 60 bars.
- **Luk screens:** they don't check market cap.
