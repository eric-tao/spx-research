from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from .data import DailyBar
from .events import collect_event_names
from .features import FeatureRow, align_and_build
from .model import (
    RangeFitResult,
    RangePrediction,
    conditional_close_given_touch,
    conditional_close_given_touch_by_regime,
)


@dataclass
class PriorDayOverrides:
    spx_open: float | None = None
    spx_high: float | None = None
    spx_low: float | None = None
    spx_close: float | None = None
    vix_open: float | None = None
    vix_high: float | None = None
    vix_low: float | None = None
    vix_close: float | None = None


@dataclass
class RegimeCutoffs:
    vix_low_upper: float
    vix_mid_upper: float
    prev_range_low_upper: float
    prev_range_mid_upper: float
    gap_flat_upper: float
    gap_small_upper: float
    gap_medium_upper: float


@dataclass
class RegimeContext:
    weekday: str
    current_vix_open: float
    prev_range_pct: float
    overnight_gap: float
    vix_regime: str
    range_regime: str
    gap_regime: str


@dataclass
class LiveExcursionPrediction:
    prediction_date: date
    open_price: float
    prev_close: float
    predicted_high_return: float
    predicted_low_return: float
    predicted_high_price: float
    predicted_low_price: float
    sigma_high_return: float
    sigma_low_return: float
    regime_context: RegimeContext


def _merge_event_flags(
    prediction_date: date,
    event_calendar: Mapping[date, set[str]],
    known_events: Sequence[str],
    selected_events: Iterable[str] | None,
) -> Dict[str, float]:
    todays_events = set(event_calendar.get(prediction_date, set()))
    todays_events.update(selected_events or [])
    flags = {f"event_{name.lower()}": 1.0 if name in todays_events else 0.0 for name in known_events}
    flags["event_any"] = 1.0 if todays_events else 0.0
    return flags


def _override_bar(bar: DailyBar, open_value: float | None, high_value: float | None, low_value: float | None, close_value: float | None) -> DailyBar:
    return DailyBar(
        date=bar.date,
        open=bar.open if open_value is None else open_value,
        high=bar.high if high_value is None else high_value,
        low=bar.low if low_value is None else low_value,
        close=bar.close if close_value is None else close_value,
        volume=bar.volume,
    )


def _apply_latest_overrides(
    spx_bars: Sequence[DailyBar],
    vix_bars: Sequence[DailyBar],
    overrides: PriorDayOverrides | None,
) -> tuple[List[DailyBar], List[DailyBar]]:
    if overrides is None:
        return list(spx_bars), list(vix_bars)
    adjusted_spx = list(spx_bars)
    adjusted_vix = list(vix_bars)
    if adjusted_spx:
        adjusted_spx[-1] = _override_bar(
            adjusted_spx[-1],
            overrides.spx_open,
            overrides.spx_high,
            overrides.spx_low,
            overrides.spx_close,
        )
    if adjusted_vix:
        adjusted_vix[-1] = _override_bar(
            adjusted_vix[-1],
            overrides.vix_open,
            overrides.vix_high,
            overrides.vix_low,
            overrides.vix_close,
        )
    return adjusted_spx, adjusted_vix


def next_weekday(on_or_after: date) -> date:
    current = on_or_after
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def previous_weekday(before_or_on: date) -> date:
    current = before_or_on
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def next_trade_date(now: datetime | None = None, timezone_name: str = "America/New_York") -> date:
    current_dt = now or datetime.now(ZoneInfo(timezone_name))
    current_date = current_dt.date()
    if current_date.weekday() >= 5:
        return next_weekday(current_date)
    market_close = time(hour=16, minute=0)
    if current_dt.timetz().replace(tzinfo=None) >= market_close:
        return next_weekday(current_date + timedelta(days=1))
    return current_date


def required_history_date(now: datetime | None = None, timezone_name: str = "America/New_York") -> date:
    target_date = next_trade_date(now=now, timezone_name=timezone_name)
    return previous_weekday(target_date - timedelta(days=1))


def default_prediction_date(spx_bars: Sequence[DailyBar], vix_bars: Sequence[DailyBar]) -> date:
    return next_trade_date()


def compute_regime_cutoffs(train_rows: Sequence[FeatureRow]) -> RegimeCutoffs:
    if not train_rows:
        raise ValueError("Need training rows to compute regime cutoffs")
    name_index = {name: idx for idx, name in enumerate(train_rows[0].feature_names)}

    def feature(row: FeatureRow, name: str) -> float:
        return row.values[name_index[name]]

    train_vix = np.asarray([feature(row, "current_vix_open") for row in train_rows], dtype=float)
    train_prev_range = np.asarray([feature(row, "spx_range_pct_lag_1") for row in train_rows], dtype=float)
    train_gap = np.asarray([feature(row, "current_spx_overnight_gap") for row in train_rows], dtype=float)
    train_abs_gap = np.abs(train_gap)

    vix_low_upper, vix_mid_upper = np.quantile(train_vix, [1.0 / 3.0, 2.0 / 3.0])
    prev_range_low_upper, prev_range_mid_upper = np.quantile(train_prev_range, [1.0 / 3.0, 2.0 / 3.0])
    gap_flat_upper = float(np.quantile(train_abs_gap, 0.25))
    gap_small_upper = float(np.quantile(train_abs_gap, 0.60))
    gap_medium_upper = float(np.quantile(train_abs_gap, 0.85))
    return RegimeCutoffs(
        vix_low_upper=float(vix_low_upper),
        vix_mid_upper=float(vix_mid_upper),
        prev_range_low_upper=float(prev_range_low_upper),
        prev_range_mid_upper=float(prev_range_mid_upper),
        gap_flat_upper=gap_flat_upper,
        gap_small_upper=gap_small_upper,
        gap_medium_upper=gap_medium_upper,
    )


def classify_regime_context(
    prediction_date: date,
    current_vix_open: float,
    prev_range_pct: float,
    overnight_gap: float,
    cutoffs: RegimeCutoffs,
) -> RegimeContext:
    abs_gap = abs(overnight_gap)
    if current_vix_open <= cutoffs.vix_low_upper:
        vix_regime = "low_vix"
    elif current_vix_open <= cutoffs.vix_mid_upper:
        vix_regime = "mid_vix"
    else:
        vix_regime = "high_vix"

    if prev_range_pct <= cutoffs.prev_range_low_upper:
        range_regime = "low_prev_range"
    elif prev_range_pct <= cutoffs.prev_range_mid_upper:
        range_regime = "mid_prev_range"
    else:
        range_regime = "high_prev_range"

    if abs_gap <= cutoffs.gap_flat_upper:
        gap_regime = "flat_gap"
    elif abs_gap <= cutoffs.gap_small_upper:
        gap_regime = "small_gap"
    elif abs_gap <= cutoffs.gap_medium_upper:
        gap_regime = "medium_gap"
    else:
        gap_regime = "large_gap"

    return RegimeContext(
        weekday=prediction_date.strftime("%A"),
        current_vix_open=current_vix_open,
        prev_range_pct=prev_range_pct,
        overnight_gap=overnight_gap,
        vix_regime=vix_regime,
        range_regime=range_regime,
        gap_regime=gap_regime,
    )


def build_live_feature_row(
    spx_bars: Sequence[DailyBar],
    vix_bars: Sequence[DailyBar],
    event_calendar: Mapping[date, set[str]],
    prediction_date: date,
    current_spx_open: float,
    current_vix_open: float,
    max_lag: int = 5,
    selected_events: Iterable[str] | None = None,
    prior_day_overrides: PriorDayOverrides | None = None,
) -> FeatureRow:
    spx_history = [bar for bar in spx_bars if bar.date < prediction_date]
    vix_history = [bar for bar in vix_bars if bar.date < prediction_date]
    if not spx_history or not vix_history:
        raise ValueError("Need historical SPX and VIX rows before the prediction date")

    common_dates = {bar.date for bar in spx_history}.intersection(bar.date for bar in vix_history)
    aligned_spx = [bar for bar in spx_history if bar.date in common_dates]
    aligned_vix = [bar for bar in vix_history if bar.date in common_dates]
    aligned_spx.sort(key=lambda bar: bar.date)
    aligned_vix.sort(key=lambda bar: bar.date)
    aligned_spx, aligned_vix = _apply_latest_overrides(aligned_spx, aligned_vix, prior_day_overrides)
    historical_rows = align_and_build(aligned_spx, aligned_vix, dict(event_calendar), max_lag=max_lag)
    if len(historical_rows) < max_lag:
        raise ValueError("Not enough historical rows to build lagged live features")

    known_events = collect_event_names(event_calendar)
    latest_spx = aligned_spx[-1]
    latest_vix = aligned_vix[-1]
    merged_event_flags = _merge_event_flags(prediction_date, event_calendar, known_events, selected_events)

    open_time_features: Dict[str, float] = {
        "current_spx_open": current_spx_open,
        "current_spx_prev_close": latest_spx.close,
        "current_spx_overnight_gap": (current_spx_open - latest_spx.close) / latest_spx.close if latest_spx.close else 0.0,
        "current_vix_open": current_vix_open,
        "current_vix_prev_close": latest_vix.close,
    }
    open_time_features.update(merged_event_flags)

    realized_feature_names = list(historical_rows[-1].metadata.keys())
    feature_values: Dict[str, float] = {}
    feature_values.update(open_time_features)
    for name in realized_feature_names:
        for lag in range(1, max_lag + 1):
            feature_values[f"{name}_lag_{lag}"] = historical_rows[-lag].metadata.get(name, 0.0)

    feature_names = list(historical_rows[-1].feature_names)
    values = [feature_values.get(name, 0.0) for name in feature_names]
    metadata = dict(historical_rows[-1].metadata)
    metadata["current_vix_open"] = current_vix_open
    metadata["current_spx_overnight_gap"] = open_time_features["current_spx_overnight_gap"]
    return FeatureRow(
        date=prediction_date,
        feature_names=feature_names,
        values=values,
        target_close=current_spx_open,
        target_return=0.0,
        target_high_return=0.0,
        target_low_return=0.0,
        open_price=current_spx_open,
        high_price=current_spx_open,
        low_price=current_spx_open,
        actual_close=current_spx_open,
        metadata=metadata,
    )


def predict_live_excursions(
    fit_result: RangeFitResult,
    live_row: FeatureRow,
    cutoffs: RegimeCutoffs,
) -> LiveExcursionPrediction:
    features = np.asarray([live_row.values], dtype=float)
    predicted_high_return = max(float(fit_result.high_model.predict(features)[0]), 0.0)
    predicted_low_return = min(float(fit_result.low_model.predict(features)[0]), 0.0)
    predicted_high_price = live_row.open_price * (1.0 + predicted_high_return)
    predicted_low_price = live_row.open_price * (1.0 + predicted_low_return)
    prev_range_pct = live_row.metadata.get("spx_range_pct", 0.0)
    regime_context = classify_regime_context(
        prediction_date=live_row.date,
        current_vix_open=live_row.values[live_row.feature_names.index("current_vix_open")],
        prev_range_pct=prev_range_pct,
        overnight_gap=live_row.values[live_row.feature_names.index("current_spx_overnight_gap")],
        cutoffs=cutoffs,
    )
    return LiveExcursionPrediction(
        prediction_date=live_row.date,
        open_price=live_row.open_price,
        prev_close=live_row.values[live_row.feature_names.index("current_spx_prev_close")],
        predicted_high_return=predicted_high_return,
        predicted_low_return=predicted_low_return,
        predicted_high_price=predicted_high_price,
        predicted_low_price=predicted_low_price,
        sigma_high_return=fit_result.sigma_high_return,
        sigma_low_return=fit_result.sigma_low_return,
        regime_context=regime_context,
    )


def threshold_probabilities(
    prediction: LiveExcursionPrediction,
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    upside_dist = NormalDist(mu=prediction.predicted_high_return, sigma=prediction.sigma_high_return)
    downside_dist = NormalDist(mu=prediction.predicted_low_return, sigma=prediction.sigma_low_return)
    for threshold in thresholds:
        rows.append(
            {
                "threshold_return": float(threshold),
                "threshold_pct": float(threshold * 100.0),
                "upside_probability": float(1.0 - upside_dist.cdf(threshold)),
                "downside_probability": float(downside_dist.cdf(-threshold)),
            }
        )
    return rows


def build_regime_probability_lookup(
    fit_result: RangeFitResult,
    backtest_predictions: Sequence,
    cutoffs: RegimeCutoffs,
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float]]]]]:
    name_index = {name: idx for idx, name in enumerate(fit_result.backtest_rows[0].feature_names)}

    def feature(row: FeatureRow, name: str) -> float:
        return row.values[name_index[name]]

    enriched_rows: List[Dict[str, object]] = []
    for feature_row, prediction in zip(fit_result.backtest_rows, backtest_predictions):
        regime_context = classify_regime_context(
            prediction_date=feature_row.date,
            current_vix_open=feature(feature_row, "current_vix_open"),
            prev_range_pct=feature(feature_row, "spx_range_pct_lag_1"),
            overnight_gap=feature(feature_row, "current_spx_overnight_gap"),
            cutoffs=cutoffs,
        )
        enriched_rows.append(
            {
                "weekday": regime_context.weekday,
                "vix_regime": regime_context.vix_regime,
                "range_regime": regime_context.range_regime,
                "gap_regime": regime_context.gap_regime,
                "predicted_high_return": prediction.predicted_high_return,
                "predicted_low_return": prediction.predicted_low_return,
                "actual_high_return": prediction.actual_high_return,
                "actual_low_return": prediction.actual_low_return,
                "sigma_high_return": prediction.sigma_high_return,
                "sigma_low_return": prediction.sigma_low_return,
            }
        )

    dimensions = {
        "overall": ["overall"],
        "vix_regime": ["low_vix", "mid_vix", "high_vix"],
        "range_regime": ["low_prev_range", "mid_prev_range", "high_prev_range"],
        "gap_regime": ["flat_gap", "small_gap", "medium_gap", "large_gap"],
        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    }
    lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float]]]]] = {
        "upside": {dimension: {} for dimension in dimensions},
        "downside": {dimension: {} for dimension in dimensions},
    }

    def _auc(probabilities: np.ndarray, actuals: np.ndarray) -> float:
        pos = probabilities[actuals == 1]
        neg = probabilities[actuals == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    for side in ["upside", "downside"]:
        for dimension, buckets in dimensions.items():
            for bucket in buckets:
                if dimension == "overall":
                    subset = enriched_rows
                else:
                    subset = [row for row in enriched_rows if row[dimension] == bucket]
                threshold_rows: Dict[float, Dict[str, float]] = {}
                for threshold in thresholds:
                    if side == "upside":
                        probabilities = np.asarray(
                            [
                                1.0 - NormalDist(mu=float(row["predicted_high_return"]), sigma=float(row["sigma_high_return"])).cdf(threshold)
                                for row in subset
                            ],
                            dtype=float,
                        )
                        actuals = np.asarray([1 if row["actual_high_return"] >= threshold else 0 for row in subset], dtype=int)
                    else:
                        probabilities = np.asarray(
                            [
                                NormalDist(mu=float(row["predicted_low_return"]), sigma=float(row["sigma_low_return"])).cdf(-threshold)
                                for row in subset
                            ],
                            dtype=float,
                        )
                        actuals = np.asarray([1 if row["actual_low_return"] <= -threshold else 0 for row in subset], dtype=int)
                    threshold_rows[float(threshold)] = {
                        "avg_probability": float(np.mean(probabilities)) if len(probabilities) else 0.0,
                        "actual_rate": float(np.mean(actuals)) if len(actuals) else 0.0,
                        "samples": float(len(actuals)),
                        "auc": _auc(probabilities, actuals),
                    }
                lookup[side][dimension][bucket] = threshold_rows
    return lookup


def build_continuation_lookup(
    fit_result: RangeFitResult,
    backtest_predictions: Sequence[RangePrediction],
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]]:
    overall_rows = conditional_close_given_touch(backtest_predictions, thresholds)
    regime_rows = conditional_close_given_touch_by_regime(
        fit_result.train_rows,
        fit_result.backtest_rows,
        backtest_predictions,
        thresholds,
    )
    lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]] = {
        "upside": {"overall": {"overall": {}}},
        "downside": {"overall": {"overall": {}}},
    }
    for side in ["upside", "downside"]:
        for row in overall_rows[side]:
            lookup[side]["overall"]["overall"][float(row["threshold_return"])] = row
        for family, rows in regime_rows[side].items():
            family_lookup: Dict[str, Dict[float, Dict[str, float | str]]] = {}
            for row in rows:
                regime_value = str(row["regime_value"])
                family_lookup.setdefault(regime_value, {})
                family_lookup[regime_value][float(row["threshold_return"])] = row
            lookup[side][family] = family_lookup
    return lookup


def select_continuation_stats(
    continuation_lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]],
    side: str,
    threshold: float,
    regime_context: RegimeContext,
    min_combo_samples: int = 8,
    min_family_samples: int = 12,
) -> Dict[str, float | str]:
    combo_key = f"{regime_context.vix_regime}|{regime_context.range_regime}|{regime_context.gap_regime}"
    combo_row = continuation_lookup.get(side, {}).get("combo_regime", {}).get(combo_key, {}).get(threshold)
    if combo_row is not None and float(combo_row.get("samples", 0.0)) >= min_combo_samples:
        return dict(combo_row, basis_label=f"combo: {combo_key}", basis_kind="combo")

    candidates = [
        ("vix", regime_context.vix_regime),
        ("gap", regime_context.gap_regime),
        ("prior range", regime_context.range_regime),
    ]
    best_row: Dict[str, float | str] | None = None
    best_label = ""
    best_samples = -1.0
    for basis_kind, regime_value in candidates:
        family_key = {
            "vix": "vix_regime",
            "gap": "gap_regime",
            "prior range": "range_regime",
        }[basis_kind]
        row = continuation_lookup.get(side, {}).get(family_key, {}).get(regime_value, {}).get(threshold)
        if row is None:
            continue
        samples = float(row.get("samples", 0.0))
        if samples >= min_family_samples and samples > best_samples:
            best_row = row
            best_label = f"{basis_kind}: {regime_value}"
            best_samples = samples
    if best_row is not None:
        return dict(best_row, basis_label=best_label, basis_kind="family")

    for basis_kind, regime_value in candidates:
        family_key = {
            "vix": "vix_regime",
            "gap": "gap_regime",
            "prior range": "range_regime",
        }[basis_kind]
        row = continuation_lookup.get(side, {}).get(family_key, {}).get(regime_value, {}).get(threshold)
        if row is None:
            continue
        samples = float(row.get("samples", 0.0))
        if samples > best_samples:
            best_row = row
            best_label = f"{basis_kind}: {regime_value}"
            best_samples = samples
    if best_row is not None:
        return dict(best_row, basis_label=best_label, basis_kind="family")

    overall_row = continuation_lookup.get(side, {}).get("overall", {}).get("overall", {}).get(threshold, {})
    return dict(overall_row, basis_label="overall", basis_kind="overall")
