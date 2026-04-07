# AGENTS.md

## Project Summary

This repo is a research sandbox for same-day SPX / QQQ options planning.

There are two main tracks:

- `spx_0dte_planner/`
  Daily-feature SPX modeling focused on excursion-from-open, touch probabilities, and conditional-close behavior.
- `intraday_condor_research/`
  Intraday QQQ/NDX condor-proxy research using 5-minute candles, expected value tables, and regime analysis.

The stronger signal in this repo is not raw close direction. It is:

- `P(high_from_open >= +x%)`
- `P(low_from_open <= -x%)`
- conditional-close behavior after a touch
- regime splits using VIX, gap, and prior-day range

## Current Product Surface

### Daily SPX webapp

Entry point:

- [`spx_0dte_planner/webapp.py`](spx_0dte_planner/webapp.py)

What it does:

- takes today's SPX open, VIX daily session open, and event flags
- uses lagged daily context from local CSVs
- shows touch levels in percent, points, and price
- shows regime-aware continuation tables if a level is touched

### Daily SPX CLI

Entry point:

- [`spx_0dte_planner/cli.py`](spx_0dte_planner/cli.py)

Important mode:

- `--target-mode range`

This prints:

- high / low model metrics
- excursion threshold backtests
- probability backtests
- conditional-close tables

### Intraday condor pipeline

Entry point:

- [`intraday_condor_research/cli.py`](intraday_condor_research/cli.py)

This is still a structural proxy, not a real options backtest.

## Important Data Assumptions

- SPX data is daily OHLC in [`data/spx_daily.csv`](data/spx_daily.csv)
- VIX data is daily OHLC in [`data/vix_daily.csv`](data/vix_daily.csv)
- events live in [`data/events.csv`](data/events.csv)
- `VIX open` means the daily `OPEN` field from the VIX history source, aligned by trade date

The daily SPX model is intended to be open-time sane:

- current-day SPX open is allowed
- current-day VIX daily session open is allowed
- current-day SPX high/low/close are not used as inputs

## Key Files

- [`spx_0dte_planner/features.py`](spx_0dte_planner/features.py)
  Daily features and targets.
- [`spx_0dte_planner/model.py`](spx_0dte_planner/model.py)
  Training, backtests, excursion probabilities, conditional-close summaries.
- [`spx_0dte_planner/live.py`](spx_0dte_planner/live.py)
  Live inference, regime classification, touch/continuation lookups.
- [`spx_0dte_planner/webapp.py`](spx_0dte_planner/webapp.py)
  Local browser UI.
- [`intraday_condor_research/stats.py`](intraday_condor_research/stats.py)
  Intraday probability / EV tables.
- [`scripts/download_market_data.py`](scripts/download_market_data.py)
  Daily data refresh; includes SPX fallback to Yahoo if Stooq is empty.

## Commands That Work

Run tests:

```bash
MPLCONFIGDIR=.mplconfig python3.11 -m pytest -q
```

Run daily SPX webapp:

```bash
python3.11 -m spx_0dte_planner.webapp \
  --spx data/spx_daily.csv \
  --vix data/vix_daily.csv \
  --events data/events.csv \
  --underlying-label SPX \
  --train-end-date 2024-12-31
```

Run daily range CLI:

```bash
python3.11 -m spx_0dte_planner.cli \
  --underlying data/spx_daily.csv \
  --underlying-label SPX \
  --vix data/vix_daily.csv \
  --events data/events.csv \
  --train-end-date 2024-12-31 \
  --pca-variance-ratio 0.95 \
  --target-mode range
```

Refresh daily history:

```bash
python3.11 scripts/download_market_data.py \
  --start 2021-01-01 \
  --end 2026-12-31
```

## Current Modeling Takeaways

- Close-direction signal is weaker than excursion / touch signal.
- The most useful daily outputs are touch probabilities and touch-conditioned close behavior.
- VIX, gap size, and prior-day range matter more as regime variables than as standalone predictors.
- Very high touch probability often marks an expansion regime, not a clean continuation regime.
- Intraday condor work is informative for width / timing / EV studies, but still lacks real chain data.

## Working Conventions

- Keep changes open-time realistic for the daily SPX model.
- Preserve the distinction between:
  - pre-open touch probabilities
  - post-touch conditional-close behavior
- If you add new UI outputs, make them interpretable in:
  - percent
  - points
  - absolute price level
- Prefer regime-aware summaries over overfit single-number signals.
- Use `data/` for tracked inputs and templates.
- Generated outputs belong under `artifacts/`.

## Testing Expectations

If you change:

- daily modeling logic: run [`tests/test_pipeline.py`](tests/test_pipeline.py)
- webapp logic: run [`tests/test_webapp.py`](tests/test_webapp.py)
- intraday research: run [`tests/test_intraday_pipeline.py`](tests/test_intraday_pipeline.py)

In practice, run the full test suite unless the user asks otherwise.

## Known Limits

- no real historical options-chain backtest
- no robust fill model yet
- manual TradingView quote calibration is sparse and auxiliary
- daily touch-conditioned close analysis does not know intraday path timing
- intraday condor EV uses structural approximations, not actual chain quotes
