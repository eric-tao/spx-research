# SPX / QQQ Options Research Sandbox

This repo is a research sandbox for same-day options planning.

It started from a daily SPX close-direction idea, but the stronger signal turned out to be:

- how far SPX is likely to move from the open
- which upside or downside levels are likely to be touched
- once a level is touched, how the day tends to finish
- how those answers change with VIX, overnight gap, and prior-day range

That led to two main tracks:

- a daily SPX excursion model and local webapp
- an intraday QQQ/NDX condor-proxy research pipeline

## What This Repo Helps With

At a high level, the repo helps answer:

- `P(high_from_open >= +x%)`
- `P(low_from_open <= -x%)`
- given a touch, where does the close tend to finish?
- what do those probabilities look like under regimes like `high_vix`, `large_gap`, or `low_prev_range`?
- for intraday condor-style planning, when does the remaining session tend to stay within a chosen width?

It is useful for research and planning.

It is not yet a production options backtester with real chain history, fill modeling, or execution logic.

## Main Pieces

### Daily SPX excursion model

Package: [`spx_0dte_planner`](spx_0dte_planner)

This is the main daily-feature modeling layer.

It uses:

- prior-day SPX and VIX OHLC
- lagged returns, ranges, rolling stats, and indicators
- the current day's SPX open
- the current day's VIX daily session open
- event flags from a calendar

The model focuses on excursion-from-open targets:

- `high_from_open = (high - open) / open`
- `low_from_open = (low - open) / open`

Instead of only predicting the close, it estimates:

- likely upside excursion
- likely downside excursion
- threshold touch probabilities
- conditional close behavior after a threshold is touched

The local webapp turns that into a live planning surface for the current day.

### Intraday QQQ/NDX condor research

Package: [`intraday_condor_research`](intraday_condor_research)

This is a separate intraday study built around QQQ 5-minute candles as a proxy for NDX-style same-day condors.

It asks:

- if price has already moved `X` from the open by time `T`, how likely is the rest of the session to stay within `Y`?
- what premium-to-width ratio would be needed for that setup to be positive expected value?

That pipeline is useful for structural condor research and regime analysis, but it is still a proxy. It does not use real historical options chains.

## Current Project Conclusions

- Raw daily close-direction prediction was weak and unstable.
- Predicting touch / excursion thresholds from the open is more useful than predicting the exact close.
- High predicted touch probability does not always mean clean continuation. At the extreme, it often marks an expansion regime rather than a simple trend day.
- The most useful daily questions are:
  - what gets touched?
  - once touched, how does the day finish?
- VIX, overnight gap size, and prior-day range are more useful as regime variables than as standalone trade signals.
- For the intraday condor proxy, wider structures and later entries are generally more viable than tight early-day structures.

## Repository Layout

- [`spx_0dte_planner/features.py`](spx_0dte_planner/features.py)
  Daily feature set and targets.
- [`spx_0dte_planner/model.py`](spx_0dte_planner/model.py)
  Training, backtesting, excursion thresholds, and conditional-close analysis.
- [`spx_0dte_planner/live.py`](spx_0dte_planner/live.py)
  Live inference helpers for the webapp.
- [`spx_0dte_planner/webapp.py`](spx_0dte_planner/webapp.py)
  Local browser UI for current-day touch and continuation analysis.
- [`intraday_condor_research/`](intraday_condor_research)
  Intraday condor-proxy analysis.
- [`scripts/download_market_data.py`](scripts/download_market_data.py)
  Refreshes SPX and VIX daily history.
- [`scripts/download_twelve_data_intraday.py`](scripts/download_twelve_data_intraday.py)
  Pulls QQQ intraday proxy candles from Twelve Data.

## Data Files

Tracked inputs live in [`data/`](data).

Main files:

- [`data/spx_daily.csv`](data/spx_daily.csv)
- [`data/vix_daily.csv`](data/vix_daily.csv)
- [`data/events.csv`](data/events.csv)
- [`data/qqq_intraday_5min.csv`](data/qqq_intraday_5min.csv)
- [`data/tradingview_option_quotes_template.csv`](data/tradingview_option_quotes_template.csv)

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
python3.11 -m spx_0dte_planner.webapp \
  --spx data/spx_daily.csv \
  --vix data/vix_daily.csv \
  --events data/events.csv \
  --underlying-label SPX \
  --train-end-date 2024-12-31
```

Then open:

- `http://localhost:8011`

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

## Frontend / Backend Split

If we move the current local UI to a React frontend while keeping analytics in Python, the planned architecture and API contract are documented in:

- [`docs/react_python_architecture.md`](docs/react_python_architecture.md)

That document defines:

- the React/Node vs Python responsibility split
- the JSON API contract
- regression guardrails for touch confirmation, cache safety, and vertical pricing provenance
- request and response schemas
- the recommended implementation order

There is now a first-pass scaffold for that split:

- Python API: [`python_api/app.py`](python_api/app.py)
- React frontend shell: [`frontend/`](frontend)

The existing Python webapp in [`spx_0dte_planner/webapp.py`](spx_0dte_planner/webapp.py) remains in place for comparison.

Run the Python API:

```bash
python3.11 -m uvicorn python_api.app:app --reload
```

Then open the API docs:

- `http://127.0.0.1:8000/docs`

Run the React frontend shell:

```bash
cd frontend
npm install
npm run dev
```

Then open:

- `http://127.0.0.1:5173`

The Vite dev server is configured to proxy `/api` to `http://127.0.0.1:8000`.

## CLI

Example daily range run:

```bash
python3.11 -m spx_0dte_planner.cli \
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
python3.11 -m intraday_condor_research.cli \
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
python3.11 scripts/download_market_data.py \
  --start 2021-01-01 \
  --end 2026-12-31
```

The downloader falls back to Yahoo for SPX if Stooq returns no rows, which helps avoid accidental empty SPX files.

## TradingView Trade-Based Strategy Cost Estimation

If you have recent SPX 0DTE option trade prints with timestamps from TradingView, you can use them as executable price proxies for individual legs and then aggregate those into estimated strategy entry costs.

This is a different workflow from the manual bid/ask snapshot calibration:

- trade prints estimate likely fill price
- bid/ask snapshots estimate quoted spread

Templates:

- [`data/tradingview_option_trade_prints_template.csv`](data/tradingview_option_trade_prints_template.csv)
- [`data/tradingview_timed_strategy_legs_template.csv`](data/tradingview_timed_strategy_legs_template.csv)

Run:

```bash
python3.11 scripts/estimate_strategy_costs_from_trades.py \
  --trades data/tradingview_option_trade_prints_template.csv \
  --legs data/tradingview_timed_strategy_legs_template.csv \
  --output-dir artifacts/tradingview_trade_costs
```

What it does:

- matches each strategy leg to the nearest trade print for the same option
- enforces a per-leg time window
- treats the matched trade price as the fill proxy for that leg
- sums the matched leg prices into a strategy-level debit or credit estimate

Important limitation:

- trade prints alone do not reconstruct the full bid/ask spread
- they estimate executable cost, not quoted market width

## TradingView 0DTE Chain Snapshot Extraction

If you copy a same-expiry SPX options chain from TradingView in a wide format, you can normalize it into the single-option quote snapshot files the rest of the repo already expects.

Template:

- [`data/tradingview_0dte_chain_snapshot_template.csv`](data/tradingview_0dte_chain_snapshot_template.csv)

The wide snapshot format is one row per strike and includes:

- snapshot id
- timestamp
- SPX spot
- optional VIX level at that snapshot
- expiry date
- strike
- call bid/ask
- put bid/ask
- optional call/put last or fill prices

Run:

```bash
python3.11 scripts/extract_tradingview_0dte_chain_snapshot.py \
  --input data/tradingview_0dte_chain_snapshot_template.csv \
  --quotes-output data/tradingview_single_option_quotes_template.csv \
  --vix-output data/tradingview_vix_snapshots_template.csv
```

What this does:

- turns each wide chain row into two normalized single-option quote rows, one call and one put
- preserves the snapshot id, timestamp, spot, expiry, and note
- optionally emits one aligned VIX snapshot row per snapshot id

This is the easiest way to get a real 0DTE chain snapshot into:

- vertical fill estimation
- trade-cost estimation
- the webapp’s vertical proxy layer

## VIX / Time-of-Day Fill Sampling

If you want to build a simple empirical model of what 0DTE fills tend to look like at different VIX levels over the course of the day, use sampled single-option observations that include:

- timestamp
- SPX price
- VIX price
- bid
- ask
- observed fill
- option type and strike
- buy vs sell side

Template:

- [`data/tradingview_fill_samples_template.csv`](data/tradingview_fill_samples_template.csv)

Run:

```bash
python3.11 scripts/model_fill_sampling.py \
  --input data/tradingview_fill_samples_template.csv \
  --output-dir artifacts/fill_sampling_model
```

What this produces:

- enriched fill rows with:
  - `mid`
  - `spread`
  - `minutes_from_open`
  - moneyness buckets
  - adverse fill from midpoint
- a bucketed summary by:
  - VIX bucket
  - time-of-day bucket
  - option type
  - moneyness
  - side
- a simple linear model of adverse fill versus:
  - VIX
  - time from open
  - spread width
  - option mid
  - moneyness
  - call/put
  - buy/sell side

Interpretation:

- this is a practical sampling model, not a market microstructure model
- it is meant to help estimate what fills tend to look like as VIX and time of day change
- it gets stronger as you add more sampled 0DTE observations across quiet and stressed sessions

## 5-Point SPX Vertical Fill Estimation

Once you have:

- sampled single-option fills with bid/ask and VIX
- a current option quote snapshot
- the VIX level aligned to that snapshot

you can estimate `5-point` SPX vertical fills by:

1. fitting the single-leg fill model
2. estimating each leg's likely fill from its quote mid
3. summing those leg estimates into a vertical debit or credit

Templates:

- [`data/tradingview_fill_samples_template.csv`](data/tradingview_fill_samples_template.csv)
- [`data/tradingview_single_option_quotes_template.csv`](data/tradingview_single_option_quotes_template.csv)
- [`data/tradingview_vix_snapshots_template.csv`](data/tradingview_vix_snapshots_template.csv)

Run:

```bash
python3.11 scripts/estimate_vertical_fills.py \
  --fill-samples data/tradingview_fill_samples_template.csv \
  --quotes data/tradingview_single_option_quotes_template.csv \
  --vix-snapshots data/tradingview_vix_snapshots_template.csv \
  --width-points 5 \
  --output-dir artifacts/vertical_fill_estimates
```

What this outputs:

- estimated fill prices for all available `5-point` call and put verticals in the snapshot
- both debit and credit structures
- estimated net price
- quoted mid net price
- natural net price
- distance from spot for the short leg

Important limitation:

- these are estimated vertical fills built from single-leg fill behavior
- they are useful as a first approximation when real vertical trade history is unavailable

## First-Pass Long / Short Vertical Opportunity Screen

You can combine:

- the daily SPX excursion model
- the touch-conditioned continuation tables
- the estimated `5-point` vertical fills

to produce a rough same-day screen for both long and short verticals held to end of day.

Run:

```bash
python3.11 scripts/screen_vertical_opportunities.py \
  --spx data/spx_daily.csv \
  --vix data/vix_daily.csv \
  --events data/events.csv \
  --prediction-date 2026-04-06 \
  --spx-open 5000 \
  --vix-open 22 \
  --fill-samples data/tradingview_fill_samples_template.csv \
  --quotes data/tradingview_single_option_quotes_template.csv \
  --vix-snapshots data/tradingview_vix_snapshots_template.csv \
  --width-points 5 \
  --output artifacts/vertical_opportunity_screen.csv
```

What this does:

- fits the daily excursion model
- estimates current touch probabilities from the open
- converts touch probabilities into rough close-beyond-strike probabilities using the continuation layer
- estimates entry prices for `5-point` debit and credit verticals from sampled single-leg fill behavior
- computes a rough expected value and `EV / risk` ranking

Current strategy types:

- `bull_call_debit`
- `bear_call_credit`
- `bear_put_debit`
- `bull_put_credit`

Important limitation:

- this is a first-pass ranking model, not a production pricing engine
- the expected value uses a simple piecewise approximation for the terminal spread payoff
- continuation probabilities are mapped from the nearest available touch threshold, not a fully continuous close distribution

## Using SPY Intraday As A First-Pass SPX Proxy

If you do not yet have direct SPX intraday candles, the repo now supports a simple `SPY -> SPX` translation for the intraday bridge backtest.

Rule:

- keep the intraday percentage path from the SPY session
- re-anchor that path to the actual SPX session open for the same date

In practice:

```text
proxy_return_t = SPY_t / SPY_open - 1
SPX_proxy_t = SPX_open * (1 + proxy_return_t)
```

This preserves the session shape while putting the path onto SPX levels.

That is still an approximation, but it is a cleaner first pass than using QQQ for SPX path work.

The intraday bridge backtest script supports this with:

```bash
python3.11 scripts/backtest_intraday_excursion_bridge.py \
  --intraday path/to/spy_intraday.csv \
  --translate-proxy-to-spx \
  --spx data/spx_daily.csv \
  --vix data/vix_daily.csv \
  --events data/events.csv
```

## Tests

Run:

```bash
MPLCONFIGDIR=.mplconfig python3.11 -m pytest -q
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
