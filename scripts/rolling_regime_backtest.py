from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import sys
from typing import Dict, List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spx_0dte_planner.backtest import DebitSpreadConfig, backtest_strategy, strategy_metrics
from spx_0dte_planner.data import load_daily_bars
from spx_0dte_planner.events import load_event_calendar
from spx_0dte_planner.features import FeatureRow, align_and_build
from spx_0dte_planner.model import DirectionalPrediction, directional_metrics


@dataclass
class RollingResult:
    predictions: List[DirectionalPrediction]
    fallback_count: int
    regime_counts: Dict[str, int]


def _make_pipeline(pca_variance_ratio: float, c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance_ratio)),
            ("classifier", LogisticRegression(max_iter=5000, C=c_value)),
        ]
    )


def regime_key(row: FeatureRow) -> str:
    gap = row.values[row.feature_names.index("current_spx_overnight_gap")]
    rsi = row.values[row.feature_names.index("spx_rsi_14_lag_1")]
    vix = row.values[row.feature_names.index("current_vix_open")]
    gap_label = "gap_up" if gap > 0 else "gap_down"
    rsi_label = "high_rsi" if rsi >= 55.0 else "low_rsi"
    vix_label = "high_vix" if vix >= 18.0 else "low_vix"
    return f"{gap_label}|{rsi_label}|{vix_label}"


def regime_parts(row: FeatureRow) -> Dict[str, str]:
    gap = row.values[row.feature_names.index("current_spx_overnight_gap")]
    rsi = row.values[row.feature_names.index("spx_rsi_14_lag_1")]
    vix = row.values[row.feature_names.index("current_vix_open")]
    return {
        "gap": "gap_up" if gap > 0 else "gap_down",
        "rsi": "high_rsi" if rsi >= 55.0 else "low_rsi",
        "vix": "high_vix" if vix >= 18.0 else "low_vix",
    }


def make_regime_key(row: FeatureRow, dimensions: Sequence[str]) -> str:
    parts = regime_parts(row)
    return "|".join(parts[dimension] for dimension in dimensions)


def make_prediction(
    row: FeatureRow,
    probability_up: float,
    sigma_return: float,
) -> DirectionalPrediction:
    predicted_return = (probability_up - 0.5) * 2.0 * sigma_return
    predicted_close = row.open_price * (1.0 + predicted_return)
    error = predicted_close - row.actual_close
    return DirectionalPrediction(
        date=row.date.isoformat(),
        predicted_return=predicted_return,
        predicted_close=predicted_close,
        actual_close=row.actual_close,
        open_price=row.open_price,
        sigma_return=sigma_return,
        probability_up=probability_up,
        probability_down=1.0 - probability_up,
        event_risk_flag=bool(row.metadata.get("event_any", 0.0)),
        probability_bands=[],
        abs_error=abs(error),
        signed_error=error,
        metadata=row.metadata,
    )


def rolling_regime_predictions(
    rows: Sequence[FeatureRow],
    start_date: str,
    lookback: int = 252,
    min_regime_samples: int = 80,
    pca_variance_ratio: float = 0.95,
    c_value: float = 0.2,
    regime_dimensions: Sequence[str] = ("gap", "rsi", "vix"),
) -> RollingResult:
    predictions: List[DirectionalPrediction] = []
    fallback_count = 0
    regime_counter: Counter[str] = Counter()
    start_idx = next((idx for idx, row in enumerate(rows) if row.date.isoformat() >= start_date), len(rows))

    for idx in range(max(start_idx, lookback), len(rows)):
        train_rows = list(rows[idx - lookback : idx])
        test_row = rows[idx]
        current_regime = make_regime_key(test_row, regime_dimensions)
        regime_counter[current_regime] += 1
        regime_rows = [row for row in train_rows if make_regime_key(row, regime_dimensions) == current_regime]
        fit_rows = regime_rows
        if len(regime_rows) < min_regime_samples:
            fit_rows = train_rows
            fallback_count += 1

        x_train = np.asarray([row.values for row in fit_rows], dtype=float)
        y_train = np.asarray([1 if row.actual_close >= row.open_price else 0 for row in fit_rows], dtype=int)
        move_train = np.asarray([(row.actual_close - row.open_price) / row.open_price for row in fit_rows], dtype=float)
        if y_train.min() == y_train.max():
            continue
        model = _make_pipeline(pca_variance_ratio=pca_variance_ratio, c_value=c_value)
        model.fit(x_train, y_train)
        probability_up = float(model.predict_proba(np.asarray([test_row.values], dtype=float))[0][1])
        sigma_return = float(np.std(move_train, ddof=1)) if len(move_train) > 1 else 1e-4
        sigma_return = max(sigma_return, 1e-4)
        predictions.append(make_prediction(test_row, probability_up, sigma_return))

    return RollingResult(predictions=predictions, fallback_count=fallback_count, regime_counts=dict(regime_counter))


def summarize_by_regime(
    rows: Sequence[FeatureRow],
    predictions: Sequence[DirectionalPrediction],
    config: DebitSpreadConfig,
    regime_dimensions: Sequence[str],
) -> List[str]:
    row_by_date = {row.date.isoformat(): row for row in rows}
    grouped: Dict[str, List[DirectionalPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[make_regime_key(row_by_date[prediction.date], regime_dimensions)].append(prediction)
    lines: List[str] = []
    for key, group in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        trades = backtest_strategy(group, config)
        metrics = strategy_metrics(trades, total_backtest_days=len(group))
        lines.append(
            f"{key} | days={len(group)} | trades={int(metrics['trades'])} | "
            f"win_rate={metrics['win_rate']:.4f} | total_pnl={metrics['total_pnl']:.2f} | "
            f"avg_pnl={metrics['avg_pnl']:.4f}"
        )
    return lines


def evaluate_regime_dimensions(
    rows: Sequence[FeatureRow],
    start_date: str,
    lookback: int,
    min_regime_samples: int,
    pca_variance_ratio: float,
    c_value: float,
    config: DebitSpreadConfig,
    regime_dimensions: Sequence[str],
) -> Dict[str, object]:
    result = rolling_regime_predictions(
        rows,
        start_date=start_date,
        lookback=lookback,
        min_regime_samples=min_regime_samples,
        pca_variance_ratio=pca_variance_ratio,
        c_value=c_value,
        regime_dimensions=regime_dimensions,
    )
    trades = backtest_strategy(result.predictions, config)
    strategy = strategy_metrics(trades, total_backtest_days=len(result.predictions))
    direction = directional_metrics(result.predictions, confidence_threshold=config.confidence_threshold)
    return {
        "dimensions": tuple(regime_dimensions),
        "predictions": result.predictions,
        "fallback_count": result.fallback_count,
        "regime_counts": result.regime_counts,
        "strategy": strategy,
        "direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling regime-aware directional backtest")
    parser.add_argument("--spx", default="data/spx_daily.csv")
    parser.add_argument("--vix", default="data/vix_daily.csv")
    parser.add_argument("--events", default="data/events.csv")
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--min-regime-samples", type=int, default=80)
    parser.add_argument("--pca-variance-ratio", type=float, default=0.95)
    parser.add_argument("--logit-c", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--spread-width", type=float, default=5.0)
    parser.add_argument("--spread-premium", type=float, default=3.0)
    parser.add_argument("--regime-dimensions", default="all", help="Comma-separated subset of gap,rsi,vix or 'all'")
    parser.add_argument("--preview", type=int, default=8)
    args = parser.parse_args()

    spx_bars = load_daily_bars(args.spx)
    vix_bars = load_daily_bars(args.vix)
    event_calendar = load_event_calendar(args.events)
    rows = align_and_build(spx_bars, vix_bars, event_calendar, max_lag=args.max_lag)

    config = DebitSpreadConfig(
        width=args.spread_width,
        premium=args.spread_premium,
        confidence_threshold=args.confidence_threshold,
    )

    all_dimensions = ("gap", "rsi", "vix")
    if args.regime_dimensions == "all":
        dimension_sets = [combo for r in range(1, len(all_dimensions) + 1) for combo in combinations(all_dimensions, r)]
    else:
        dimension_sets = [tuple(part.strip() for part in args.regime_dimensions.split(",") if part.strip())]

    evaluations = [
        evaluate_regime_dimensions(
            rows,
            start_date=args.start_date,
            lookback=args.lookback,
            min_regime_samples=args.min_regime_samples,
            pca_variance_ratio=args.pca_variance_ratio,
            c_value=args.logit_c,
            config=config,
            regime_dimensions=dimensions,
        )
        for dimensions in dimension_sets
    ]
    evaluations.sort(key=lambda item: (item["strategy"]["total_pnl"], item["strategy"]["avg_pnl"]), reverse=True)

    print("Regime Combination Summary")
    for evaluation in evaluations:
        dimensions = ",".join(evaluation["dimensions"])
        strategy = evaluation["strategy"]
        direction = evaluation["direction"]
        predictions = evaluation["predictions"]
        print(
            f"{dimensions} | predictions={len(predictions)} | fallbacks={evaluation['fallback_count']} | "
            f"acc={direction['accuracy']:.4f} | auc={direction['auc']:.4f} | coverage={direction['confidence_coverage']:.4f} | "
            f"trades={int(strategy['trades'])} | win_rate={strategy['win_rate']:.4f} | "
            f"total_pnl={strategy['total_pnl']:.2f} | avg_pnl={strategy['avg_pnl']:.4f}"
        )
    print()

    best = evaluations[0]
    result = {
        "predictions": best["predictions"],
        "fallback_count": best["fallback_count"],
        "regime_counts": best["regime_counts"],
        "dimensions": best["dimensions"],
        "strategy": best["strategy"],
        "direction": best["direction"],
    }
    trades = backtest_strategy(result["predictions"], config)
    metrics = result["strategy"]
    direction = result["direction"]

    print("Rolling Summary")
    print(
        f"best_dimensions={','.join(result['dimensions'])}, predictions={len(result['predictions'])}, "
        f"lookback={args.lookback}, fallbacks={result['fallback_count']}, confidence_threshold={args.confidence_threshold:.2f}"
    )
    print()
    print("Directional Metrics")
    print(
        f"accuracy={direction['accuracy']:.4f}, auc={direction['auc']:.4f}, "
        f"brier={direction['brier']:.4f}, coverage={direction['confidence_coverage']:.4f}, "
        f"coverage_accuracy={direction['confidence_accuracy']:.4f}"
    )
    print()
    print("Strategy Metrics")
    print(
        f"trades={int(metrics['trades'])}, trade_rate={metrics['trade_rate']:.4f}, "
        f"win_rate={metrics['win_rate']:.4f}, total_pnl={metrics['total_pnl']:.2f}, "
        f"avg_pnl={metrics['avg_pnl']:.4f}, profit_factor={metrics['profit_factor']:.4f}"
    )
    print()
    print("Regime Breakdown")
    for line in summarize_by_regime(rows, result["predictions"], config, result["dimensions"]):
        print(line)
    print()
    print("Latest Predictions")
    recent = result["predictions"][-args.preview :]
    trade_by_date = {trade.date: trade for trade in backtest_strategy(recent, config)}
    row_by_date = {row.date.isoformat(): row for row in rows}
    for prediction in recent:
        trade = trade_by_date.get(prediction.date)
        print(
            f"{prediction.date} | {make_regime_key(row_by_date[prediction.date], result['dimensions'])} | "
            f"p_up={prediction.probability_up:.4f} | trade={trade.direction if trade else 'no_trade'} | "
            f"realized_close={prediction.actual_close:.2f}"
        )


if __name__ == "__main__":
    main()
