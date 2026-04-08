from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from intraday_condor_research.vertical_fills import estimate_vertical_fills_from_quotes, load_vertical_fill_inputs
from spx_0dte_planner.data import load_daily_bars
from spx_0dte_planner.events import load_event_calendar
from spx_0dte_planner.features import align_and_build
from spx_0dte_planner.live import (
    PriorDayOverrides,
    build_live_feature_row,
    build_touch_target_lookup,
    compute_regime_cutoffs,
    predict_live_excursions,
)
from spx_0dte_planner.model import fit_train_backtest_range_model, predict_backtest_range
from spx_0dte_planner.opportunity_screen import score_vertical_opportunities_after_touch


def _parse_thresholds(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen SPX long and short 5-point verticals after a touch has already occurred"
    )
    parser.add_argument("--spx", default="data/spx_daily.csv")
    parser.add_argument("--vix", default="data/vix_daily.csv")
    parser.add_argument("--events", default="data/events.csv")
    parser.add_argument("--prediction-date", required=True, help="Trade date in YYYY-MM-DD")
    parser.add_argument("--spx-open", required=True, type=float, help="Regular-session SPX open")
    parser.add_argument("--vix-open", required=True, type=float, help="VIX daily session open")
    parser.add_argument("--touched-side", required=True, choices=["upside_touch", "downside_touch"])
    parser.add_argument("--touch-threshold", required=True, type=float, help="Touched threshold as decimal return from open, e.g. 0.005 for 0.50%%")
    parser.add_argument("--fill-samples", required=True, help="CSV of sampled single-option fills with bid/ask and VIX")
    parser.add_argument("--quotes", required=True, help="CSV of option quote snapshots")
    parser.add_argument("--vix-snapshots", required=True, help="CSV of VIX levels aligned to quote snapshots")
    parser.add_argument("--train-end-date", default="2024-12-31")
    parser.add_argument("--touch-thresholds", default="0.0025,0.005,0.0075,0.01")
    parser.add_argument("--close-thresholds", default="0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01")
    parser.add_argument("--width-points", type=float, default=5.0)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--pca-variance-ratio", type=float, default=0.95)
    parser.add_argument("--output", default="artifacts/vertical_opportunity_screen_after_touch.csv")
    args = parser.parse_args()

    spx_bars = load_daily_bars(args.spx)
    vix_bars = load_daily_bars(args.vix)
    events = load_event_calendar(args.events)
    rows = align_and_build(spx_bars, vix_bars, events, max_lag=args.max_lag)
    range_fit = fit_train_backtest_range_model(
        rows,
        train_end_date=args.train_end_date,
        pca_variance_ratio=args.pca_variance_ratio,
    )
    backtest_predictions = predict_backtest_range(range_fit)
    touch_target_lookup = build_touch_target_lookup(
        range_fit,
        backtest_predictions,
        touch_thresholds=_parse_thresholds(args.touch_thresholds),
        close_thresholds=_parse_thresholds(args.close_thresholds),
    )

    live_row = build_live_feature_row(
        spx_bars,
        vix_bars,
        events,
        prediction_date=pd.Timestamp(args.prediction_date).date(),
        current_spx_open=args.spx_open,
        current_vix_open=args.vix_open,
        max_lag=args.max_lag,
        selected_events=[],
        prior_day_overrides=PriorDayOverrides(),
    )
    prediction = predict_live_excursions(range_fit, live_row, compute_regime_cutoffs(range_fit.train_rows))

    fill_samples, quote_snapshots, vix_by_snapshot = load_vertical_fill_inputs(
        args.fill_samples,
        args.quotes,
        args.vix_snapshots,
    )
    vertical_estimates = estimate_vertical_fills_from_quotes(
        fill_samples=fill_samples,
        quote_snapshots=quote_snapshots,
        vix_by_snapshot=vix_by_snapshot,
        width_points=args.width_points,
    )
    opportunities = score_vertical_opportunities_after_touch(
        prediction=prediction,
        touch_target_lookup=touch_target_lookup,
        vertical_estimates=vertical_estimates,
        touched_side=args.touched_side,
        touch_threshold_return=args.touch_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opportunities.to_csv(output_path, index=False)

    print("Vertical Opportunity Screen After Touch")
    print(f"conditioning_touch_side={args.touched_side}")
    print(f"conditioning_touch_threshold_pct={args.touch_threshold * 100.0:.2f}")
    print(f"vertical_candidates={len(vertical_estimates)}")
    print(f"opportunities={len(opportunities)}")
    print(f"output={output_path}")
    if not opportunities.empty:
        preview = opportunities.head(10)[
            [
                "strategy",
                "outlook",
                "entry_price",
                "expected_value",
                "ev_to_risk",
                "predicted_terminal_value_proxy",
                "profit_to_cost_ratio_proxy",
                "near_close_probability_given_touch",
                "far_close_probability_given_touch",
            ]
        ]
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
