# SPX / QQQ Options Research Sandbox

This repo grew out of a practical question: can we use daily market context to say something useful about same-day SPX behavior, and then turn that into better planning for structures like verticals, condors, or simple directional trades?

The short answer is yes, but not in the original form.

The project started as a daily close-direction model for SPX 0DTE planning. That turned out to be weaker than we wanted. The more useful signal came from reframing the problem as:

- how far is SPX likely to move away from the open?
- which upside or downside levels are likely to be touched?
- once a level is touched, how does the day usually finish?
- how do those answers change with VIX, overnight gap, and prior-day range?

That led to two research tracks in this repo:

- a daily SPX excursion model and local webapp
- an intraday QQQ/NDX condor-proxy research pipeline

## What This Repo Is Good For

At a high level, this repo helps answer:

- `P(high_from_open >= +x%)`
- `P(low_from_open <= -x%)`
- given a touch, where does the close tend to finish?
- what do those probabilities look like under specific regimes like `high_vix`, `large_gap`, or `low_prev_range`?
- for intraday condor-style planning, when does the remaining session tend to stay within a chosen width?

It is useful as a research and planning environment.

It is not yet a production options backtester with real chain history, fill modeling, or execution logic.

## Main Pieces

### 1. Daily SPX excursion model

Package: [`spx_0dte_planner`](/Users/erictao/trading_2/spx_0dte_planner)

This is the main daily-feature modeling layer.

It uses:

- prior-day SPX and VIX OHLC
- lagged returns, ranges, rolling stats, and indicators
- the current day's SPX open
- the current day's VIX daily session open
- event flags from a calendar

The model now focuses on excursion-from-open targets:

- `high_from_open = (high - open) / open`
- `low_from_open = (low - open) / open`

Instead of only predicting the close, it estimates:

- likely upside excursion
- likely downside excursion
- threshold touch probabilities
- conditional close behavior after a threshold is touched

The local webapp then turns that into a live planning surface for the current day.

### 2. Intraday QQQ/NDX condor research

Package: [`intraday_condor_research`](/Users/erictao/trading_2/intraday_condor_research)

This is a separate intraday study built around QQQ 5-minute candles as a proxy for NDX-style same-day condors.

It asks:

- if price has already moved `X` from the open by time `T`, how likely is the rest of the session to stay within `Y`?
- what premium-to-width ratio would be needed for that setup to be positive expected value?

That pipeline is useful for structural condor research and regime analysis, but it is still a proxy. It does not use real historical options chains.

## Current Project Conclusions

These are the broad findings that drove the current shape of the code:

- Raw daily close-direction prediction was weak and unstable.
- Predicting touch / excursion thresholds from the open is more useful than predicting the exact close.
- High predicted touch probability does not always mean clean continuation. At the extreme, it often marks an expansion regime rather than a simple trend day.
- The most useful daily questions are:
  - what gets touched?
  - once touched, how does the day finish?
- VIX, overnight gap size, and prior-day range are more useful as regime variables than as standalone trade signals.
- For the intraday condor proxy, wider structures and later entries are generally more viable than tight early-day structures.

## Repository Layout

- [`spx_0dte_planner/features.py`](/Users/erictao/trading_2/spx_0dte_planner/features.py)
  Builds the daily feature set.
- [`spx_0dte_planner/model.py`](/Users/erictao/trading_2/spx_0dte_planner/model.py)
  Training, backtesting, excursion thresholds, and conditional-close analysis.
- [`spx_0dte_planner/live.py`](/Users/erictao/trading_2/spx_0dte_planner/live.py)
  Live inference helpers for the webapp.
- [`spx_0dte_planner/webapp.py`](/Users/erictao/trading_2/spx_0dte_planner/webapp.py)
  Local browser UI for current-day touch and continuation analysis.
- [`intraday_condor_research/`](/Users/erictao/trading_2/intraday_condor_research)
  Intraday condor-proxy analysis.
- [`scripts/download_market_data.py`](/Users/erictao/trading_2/scripts/download_market_data.py)
  Refreshes SPX and VIX daily history.
- [`scripts/download_twelve_data_intraday.py`](/Users/erictao/trading_2/scripts/download_twelve_data_intraday.py)
  Pulls QQQ intraday proxy candles from Twelve Data.

## Data Files

Tracked inputs live in [`data/`](/Users/erictao/trading_2/data).

Main files:

- [`data/spx_daily.csv`](/Users/erictao/trading_2/data/spx_daily.csv)
- [`data/vix_daily.csv`](/Users/erictao/trading_2/data/vix_daily.csv)
- [`data/events.csv`](/Users/erictao/trading_2/data/events.csv)
- [`data/qqq_intraday_5min.csv`](/Users/erictao/trading_2/data/qqq_intraday_5min.csv)
- [`data/tradingview_option_quotes_template.csv`](/Users/erictao/trading_2/data/tradingview_option_quotes_template.csv)

Expected daily OHLC columns:

- `date`
- `open`
- `high`
- `low`
- `close`
- optional `volume`

Expected event calendar columns:

- `date`
- `event_type`

## Daily SPX Model

### Inputs

The daily model is deliberately constrained to what should be knowable at or near the open:

- today's SPX regular-session open
- today's VIX daily session open
- lagged SPX and VIX features from prior days
- same-day event flags

`VIX open` is defined consistently with the training data:

- it comes from the `OPEN` column of Cboe's daily `VIX_History.csv`
- it is aligned by trade date
- it is not a separately sampled `9:30:00` tick

### Outputs

The range / excursion path produces:

- predicted high from open
- predicted low from open
- touch probabilities for a threshold grid
- backtested AUC / Brier / base-rate tables by threshold
- conditional-close summaries after touches
- regime-conditioned continuation summaries

### Why AUC matters here

AUC is the area under the ROC curve. In this repo, it is mostly used as a ranking-quality score:

- `0.50` is random
- `> 0.50` means the model tends to rank true touch days above non-touch days
- `0.70+` is meaningful for a noisy market problem

This is more useful than plain hit rate because it tells us whether the model is actually sorting days in a useful order.

## Webapp

Run the local UI with:

```bash
cd /Users/erictao/trading_2
/opt/homebrew/bin/python3.11 -m spx_0dte_planner.webapp \
  --spx data/spx_daily.csv \
  --vix data/vix_daily.csv \
  --events data/events.csv \
  --underlying-label SPX \
  --train-end-date 2024-12-31
```

Then open:

- `http://127.0.0.1:8011`

What the UI shows:

- the suggested next trade date
- today's input form
- top upside and downside thresholds worth watching
- touch levels in:
  - percent
  - SPX points
  - absolute price level
- regime-aware historical hit rates
- continuation tables answering:
  - if this level is touched on a day like today, where did the close usually finish?

The continuation layer uses a practical fallback order:

1. exact `vix + prior-range + gap` combo
2. strongest matching single regime slice
3. overall history

That keeps the UI informative without pretending sparse historical buckets are more precise than they are.

## CLI

Example daily range run:

```bash
/opt/homebrew/bin/python3.11 -m spx_0dte_planner.cli \
  --underlying data/spx_daily.csv \
  --underlying-label SPX \
  --vix data/vix_daily.csv \
  --events data/events.csv \
  --train-end-date 2024-12-31 \
  --pca-variance-ratio 0.95 \
  --target-mode range \
  --excursion-thresholds 0.005,0.0075,0.01
```

In `range` mode, the CLI prints:

- high / low prediction error metrics
- threshold touch backtests
- probability backtests
- conditional-close summaries after touches
- regime-conditioned touch-to-close behavior

## Intraday Condor Research

Run the intraday pipeline with:

```bash
/opt/homebrew/bin/python3.11 -m intraday_condor_research.cli \
  --input data/qqq_intraday_5min.csv \
  --vix-daily data/vix_daily.csv \
  --symbol QQQ \
  --output-dir artifacts/intraday_condor
```

This pipeline produces:

- per-day and per-checkpoint datasets
- remaining-session excursion tables
- simplified expected value tables
- breakeven credit tables
- decision tables for chosen EV thresholds
- regime tables and plots

### Volatility regime formula

The intraday regime is designed to be live-safe and uses:

- daily `VIX`
- estimated `VXN = vxn_multiplier * VIX`
- the last 5 completed intraday OHLC bars

Forward-looking daily move proxy:

```text
sigma_fwd = ((VIX + VXN_est) / 2) / (100 * sqrt(252))
VXN_est = vxn_multiplier * VIX
```

Intraday realized-vol proxy from the last 5 completed bars uses a Garman-Klass estimate:

```text
v_i = 0.5 * ln(H_i / L_i)^2 - (2 * ln(2) - 1) * ln(C_i / O_i)^2
sigma_intraday = sqrt(bars_per_session * mean(v_i over last 5 completed bars))
```

Regime score:

```text
sigma_regime = max(sigma_fwd, sigma_intraday)
```

Current buckets:

- `low_vol` if `< 1.0%`
- `mid_vol` if `1.0% <= sigma_regime < 2.0%`
- `high_vol` if `>= 2.0%`

## Data Refresh

Refresh daily SPX/VIX history with:

```bash
/opt/homebrew/bin/python3.11 scripts/download_market_data.py \
  --start 2021-01-01 \
  --end 2026-12-31
```

The downloader now falls back to Yahoo for SPX if Stooq returns no rows, which helps avoid accidental empty SPX files.

## Tests

Run:

```bash
MPLCONFIGDIR=/Users/erictao/trading_2/.mplconfig /opt/homebrew/bin/python3.11 -m pytest -q
```

## Current Limitations

This repo is still a research sandbox, not a production trading system.

Important limits:

- daily SPX modeling is based on daily features, not intraday state
- touch-conditioned close behavior is path-agnostic within the day
- there is no historical options chain backtest yet
- fill assumptions are still simplified
- the TradingView quote workflow is manual and sparse
- intraday condor expected value uses structural proxies rather than real quote history

## Where to Go Next

The most natural next upgrades are:

- replace ridge baselines with quantile or distributional models
- add true intraday state for post-touch updates
- add real options-chain history and fill calibration
- turn the current webapp outputs into explicit strategy playbooks

For now, the repo is in a solid place for regime-aware SPX touch analysis and intraday condor proxy research.
