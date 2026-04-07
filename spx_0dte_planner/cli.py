from __future__ import annotations

import argparse
from typing import Dict

from .backtest import DebitSpreadConfig, backtest_strategy, strategy_metrics
from .data import load_daily_bars
from .events import collect_event_names, load_event_calendar
from .features import align_and_build
from .model import (
    conditional_close_given_touch_by_regime,
    conditional_close_given_touch,
    directional_metrics,
    excursion_probability_backtest,
    excursion_threshold_metrics,
    fit_direction_classifier,
    fit_train_backtest_range_model,
    fit_train_backtest_model,
    metrics_by_flag,
    predict_backtest,
    predict_backtest_classifier,
    predict_backtest_range,
    range_metrics,
    regression_metrics,
)


def _format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "n/a"
    parts = []
    for key, value in metrics.items():
        if key == "samples" or key == "trades":
            parts.append(f"{key}={int(value)}")
        elif value == float("inf"):
            parts.append(f"{key}=inf")
        else:
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


def _parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="SPX 0DTE vertical planning model")
    parser.add_argument("--underlying", help="Path to underlying daily CSV (e.g. SPY or QQQ)")
    parser.add_argument("--spx", help="Deprecated alias for --underlying")
    parser.add_argument("--underlying-label", default="UNDERLYING", help="Label used in range-mode output, e.g. SPY or QQQ")
    parser.add_argument("--vix", required=True, help="Path to VIX daily CSV")
    parser.add_argument("--events", default="data/events.csv", help="Path to event calendar CSV")
    parser.add_argument("--target-mode", choices=["range", "close"], default="range", help="Model target: daily high/low range or close")
    parser.add_argument("--excursion-thresholds", default="0.0025,0.005,0.0075,0.01", help="Comma-separated return thresholds for upside/downside excursion backtests in range mode")
    parser.add_argument("--max-lag", type=int, default=5, help="Maximum lag depth per feature")
    parser.add_argument("--train-end-date", default="2024-12-31", help="Inclusive end date for training set")
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Fallback chronological split ratio")
    parser.add_argument("--ridge", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--pca-components", type=int, help="Fixed PCA component count")
    parser.add_argument("--pca-variance-ratio", type=float, default=0.95, help="Target retained variance for PCA; ignored if --pca-components is set")
    parser.add_argument("--event-vol-multiplier", type=float, default=1.5, help="Sigma widening on flagged event days")
    parser.add_argument("--confidence-threshold", type=float, default=0.70, help="Directional probability threshold to enter a trade")
    parser.add_argument("--spread-width", type=float, default=5.0, help="Width of the ITM debit spread")
    parser.add_argument("--spread-premium", type=float, default=3.0, help="Entry debit paid for the spread")
    parser.add_argument("--preview", type=int, default=5, help="Rows of latest predictions to print")
    args = parser.parse_args()

    underlying_path = args.underlying or args.spx
    if not underlying_path:
        raise SystemExit("Provide --underlying (or legacy --spx)")
    spx_bars = load_daily_bars(underlying_path)
    vix_bars = load_daily_bars(args.vix)
    event_calendar = load_event_calendar(args.events)
    feature_rows = align_and_build(spx_bars, vix_bars, event_calendar, max_lag=args.max_lag)
    if args.target_mode == "range":
        thresholds = _parse_thresholds(args.excursion_thresholds)
        range_fit = fit_train_backtest_range_model(
            feature_rows,
            ridge_lambda=args.ridge,
            train_end_date=args.train_end_date,
            train_ratio=args.train_ratio,
            pca_components=args.pca_components,
            pca_variance_ratio=args.pca_variance_ratio,
        )
        range_predictions = predict_backtest_range(range_fit)

        print("Data split")
        print(
            f"train_rows={len(range_fit.train_rows)}, backtest_rows={len(range_fit.backtest_rows)}, "
            f"train_start={range_fit.train_rows[0].date.isoformat()}, train_end={range_fit.train_rows[-1].date.isoformat()}, "
            f"backtest_start={range_fit.backtest_rows[0].date.isoformat()}, backtest_end={range_fit.backtest_rows[-1].date.isoformat()}"
        )
        print()
        print("Feature reduction")
        print(
            f"input_features={range_fit.input_feature_count}, "
            f"high_reduced_features={range_fit.high_reduced_feature_count}, "
            f"high_explained_variance_ratio={range_fit.high_explained_variance_ratio:.4f}, "
            f"low_reduced_features={range_fit.low_reduced_feature_count}, "
            f"low_explained_variance_ratio={range_fit.low_explained_variance_ratio:.4f}"
        )
        print()
        print(f"{args.underlying_label} High/Low Range Metrics")
        print(_format_metrics(range_metrics(range_predictions)))
        print()
        print(f"{args.underlying_label} Excursion Threshold Backtest")
        excursion_metrics = excursion_threshold_metrics(range_predictions, thresholds=thresholds)
        for side in ["upside", "downside"]:
            print(f"{side}:")
            for row in excursion_metrics[side]:
                print(
                    "  "
                    f"threshold={row['threshold_pct']:.2f}% | coverage={row['coverage']:.4f} | "
                    f"actual_rate={row['actual_rate']:.4f} | precision={row['precision']:.4f} | "
                    f"recall={row['recall']:.4f}"
                )
        print()
        print(f"{args.underlying_label} Excursion Probability Backtest")
        probability_metrics = excursion_probability_backtest(range_predictions, thresholds=thresholds)
        for side in ["upside", "downside"]:
            print(f"{side}:")
            for row in probability_metrics[side]:
                print(
                    "  "
                    f"threshold={row['threshold_pct']:.2f}% | avg_p={row['avg_predicted_probability']:.4f} | "
                    f"actual_rate={row['actual_rate']:.4f} | auc={row['auc']:.4f} | "
                    f"brier={row['brier']:.4f} | coverage@50={row['coverage_at_50']:.4f} | "
                    f"precision@50={row['precision_at_50']:.4f} | recall@50={row['recall_at_50']:.4f}"
                )
        print()
        print(f"{args.underlying_label} Conditional Close After Touch")
        conditional_metrics = conditional_close_given_touch(range_predictions, thresholds=thresholds)
        for side in ["upside", "downside"]:
            print(f"{side}:")
            for row in conditional_metrics[side]:
                if side == "upside":
                    print(
                        "  "
                        f"threshold={row['threshold_pct']:.2f}% | touch_rate={row['touch_rate']:.4f} | "
                        f"samples={int(row['samples'])} | avg_close_from_open={row['avg_close_return'] * 100.0:.2f}% | "
                        f"avg_finish_vs_touch={row['avg_finish_vs_touch'] * 100.0:.2f}% | "
                        f"close_above_open={row['close_above_open_rate']:.4f} | "
                        f"close_above_touch={row['close_above_touch_rate']:.4f}"
                    )
                else:
                    print(
                        "  "
                        f"threshold={row['threshold_pct']:.2f}% | touch_rate={row['touch_rate']:.4f} | "
                        f"samples={int(row['samples'])} | avg_close_from_open={row['avg_close_return'] * 100.0:.2f}% | "
                        f"avg_finish_vs_touch={row['avg_finish_vs_touch'] * 100.0:.2f}% | "
                        f"close_below_open={row['close_below_open_rate']:.4f} | "
                        f"close_below_touch={row['close_below_touch_rate']:.4f}"
                    )
        print()
        print(f"{args.underlying_label} Conditional Close After Touch By Regime")
        regime_metrics = conditional_close_given_touch_by_regime(
            range_fit.train_rows,
            range_fit.backtest_rows,
            range_predictions,
            thresholds=thresholds,
        )
        for side in ["upside", "downside"]:
            print(f"{side}:")
            for regime_family in ["vix_regime", "range_regime", "gap_regime"]:
                print(f"  {regime_family}:")
                grouped_rows = [row for row in regime_metrics[side][regime_family] if row["samples"] >= 15.0]
                grouped_rows.sort(key=lambda row: (row["threshold_pct"], row["regime_value"]))
                for row in grouped_rows:
                    if side == "upside":
                        print(
                            "    "
                            f"{row['regime_value']} | threshold={row['threshold_pct']:.2f}% | "
                            f"touched={int(row['samples'])}/{int(row['regime_days'])} | "
                            f"avg_close_from_open={row['avg_close_return'] * 100.0:.2f}% | "
                            f"close_above_open={row['close_above_open_rate']:.4f} | "
                            f"close_above_touch={row['close_above_touch_rate']:.4f}"
                        )
                    else:
                        print(
                            "    "
                            f"{row['regime_value']} | threshold={row['threshold_pct']:.2f}% | "
                            f"touched={int(row['samples'])}/{int(row['regime_days'])} | "
                            f"avg_close_from_open={row['avg_close_return'] * 100.0:.2f}% | "
                            f"close_below_open={row['close_below_open_rate']:.4f} | "
                            f"close_below_touch={row['close_below_touch_rate']:.4f}"
                        )
        print()
        print("Latest Excursion Predictions")
        for prediction in range_predictions[-args.preview:]:
            event_text = "event_risk" if prediction.event_risk_flag else "normal"
            print(
                f"{prediction.date} | {event_text} | open={prediction.open_price:.2f} | "
                f"pred_high_from_open={prediction.predicted_high_return * 100.0:.2f}% ({prediction.predicted_high:.2f}) | "
                f"actual_high_from_open={prediction.actual_high_return * 100.0:.2f}% ({prediction.actual_high:.2f}) | "
                f"pred_low_from_open={prediction.predicted_low_return * 100.0:.2f}% ({prediction.predicted_low:.2f}) | "
                f"actual_low_from_open={prediction.actual_low_return * 100.0:.2f}% ({prediction.actual_low:.2f})"
            )
        return

    fit_result = fit_train_backtest_model(
        feature_rows,
        ridge_lambda=args.ridge,
        train_end_date=args.train_end_date,
        train_ratio=args.train_ratio,
        pca_components=args.pca_components,
        pca_variance_ratio=args.pca_variance_ratio,
    )
    classifier_fit = fit_direction_classifier(
        feature_rows,
        train_end_date=args.train_end_date,
        train_ratio=args.train_ratio,
        pca_components=args.pca_components,
        pca_variance_ratio=args.pca_variance_ratio,
    )
    regression_predictions = predict_backtest(fit_result, event_vol_multiplier=args.event_vol_multiplier)
    classifier_predictions = predict_backtest_classifier(
        classifier_fit,
        sigma_return=fit_result.sigma_return,
        event_vol_multiplier=args.event_vol_multiplier,
    )
    config = DebitSpreadConfig(
        width=args.spread_width,
        premium=args.spread_premium,
        confidence_threshold=args.confidence_threshold,
    )
    regression_trades = backtest_strategy(regression_predictions, config)
    classifier_trades = backtest_strategy(classifier_predictions, config)

    print("Data split")
    print(
        f"train_rows={len(fit_result.train_rows)}, backtest_rows={len(fit_result.backtest_rows)}, "
        f"train_start={fit_result.train_rows[0].date.isoformat()}, train_end={fit_result.train_rows[-1].date.isoformat()}, "
        f"backtest_start={fit_result.backtest_rows[0].date.isoformat()}, backtest_end={fit_result.backtest_rows[-1].date.isoformat()}"
    )
    print()
    print("Feature reduction")
    print(
        f"input_features={fit_result.input_feature_count}, reduced_features={fit_result.reduced_feature_count}, "
        f"explained_variance_ratio={fit_result.explained_variance_ratio:.4f}"
    )
    print()
    print("Return Model Metrics")
    print(_format_metrics(regression_metrics(regression_predictions)))
    print(_format_metrics(directional_metrics(regression_predictions, confidence_threshold=args.confidence_threshold)))
    print()
    print("Classifier Metrics")
    print(_format_metrics(directional_metrics(classifier_predictions, confidence_threshold=args.confidence_threshold)))
    print()

    for event_name in collect_event_names(event_calendar):
        flag = f"event_{event_name.lower()}"
        split_metrics = metrics_by_flag(regression_predictions, flag)
        print(f"{event_name} days (return model)")
        print(f"  flagged:   {_format_metrics(split_metrics['flagged'])}")
        print(f"  unflagged: {_format_metrics(split_metrics['unflagged'])}")
        print()

    print("Strategy Comparison")
    print(f"return_model: {_format_metrics(strategy_metrics(regression_trades, total_backtest_days=len(regression_predictions)))}")
    print(f"classifier:   {_format_metrics(strategy_metrics(classifier_trades, total_backtest_days=len(classifier_predictions)))}")
    print()

    print("Latest Plans")
    recent_regression = regression_predictions[-args.preview:]
    recent_classifier = classifier_predictions[-args.preview:]
    regression_trade_by_date = {trade.date: trade for trade in backtest_strategy(recent_regression, config)}
    classifier_trade_by_date = {trade.date: trade for trade in backtest_strategy(recent_classifier, config)}
    for regression_prediction, classifier_prediction in zip(recent_regression, recent_classifier):
        event_text = "event_risk" if regression_prediction.event_risk_flag else "normal"
        print(
            f"{regression_prediction.date} | {event_text} | "
            f"return_model_p_up={regression_prediction.probability_up:.4f} | "
            f"classifier_p_up={classifier_prediction.probability_up:.4f}"
        )
        for band in regression_prediction.probability_bands:
            lower = f"{band.lower_price:.2f}" if band.lower_price is not None else "-inf"
            upper = f"{band.upper_price:.2f}" if band.upper_price is not None else "inf"
            print(f"  band={band.label} | range=[{lower}, {upper}] | prob={band.probability:.4f}")
        return_trade = regression_trade_by_date.get(regression_prediction.date)
        classifier_trade = classifier_trade_by_date.get(classifier_prediction.date)
        print(
            f"  return_model_trade={return_trade.direction if return_trade else 'no_trade'} | "
            f"classifier_trade={classifier_trade.direction if classifier_trade else 'no_trade'}"
        )


if __name__ == "__main__":
    main()
