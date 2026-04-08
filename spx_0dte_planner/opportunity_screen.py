from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict

import pandas as pd

from .live import LiveExcursionPrediction, select_continuation_stats, select_touch_target_stats


@dataclass
class ThresholdEstimate:
    threshold_return: float
    touch_probability: float
    close_beyond_probability: float
    threshold_used_for_continuation: float
    continuation_basis: str
    continuation_rate: float


@dataclass
class ConditionalCloseEstimate:
    touched_side: str
    touch_threshold_return: float
    close_side: str
    close_threshold_return: float
    close_probability: float
    touch_threshold_used: float
    close_threshold_used: float
    continuation_basis: str
    avg_close_return: float


def _nearest_threshold(
    continuation_lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]],
    side: str,
    threshold_return: float,
) -> float:
    available = sorted(float(value) for value in continuation_lookup[side]["overall"]["overall"].keys())
    return min(available, key=lambda value: abs(value - threshold_return))


def estimate_close_beyond_threshold(
    *,
    prediction: LiveExcursionPrediction,
    continuation_lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]],
    side: str,
    threshold_return: float,
) -> ThresholdEstimate:
    if side == "upside":
        dist = NormalDist(mu=prediction.predicted_high_return, sigma=max(prediction.sigma_high_return, 1e-6))
        touch_probability = float(1.0 - dist.cdf(threshold_return))
        close_past_key = "close_above_touch_rate"
    else:
        dist = NormalDist(mu=prediction.predicted_low_return, sigma=max(prediction.sigma_low_return, 1e-6))
        touch_probability = float(dist.cdf(-threshold_return))
        close_past_key = "close_below_touch_rate"

    nearest_threshold = _nearest_threshold(continuation_lookup, side, threshold_return)
    continuation = select_continuation_stats(
        continuation_lookup,
        side,
        nearest_threshold,
        prediction.regime_context,
    )
    continuation_rate = float(continuation.get(close_past_key, 0.0))
    close_beyond_probability = touch_probability * continuation_rate
    return ThresholdEstimate(
        threshold_return=threshold_return,
        touch_probability=touch_probability,
        close_beyond_probability=close_beyond_probability,
        threshold_used_for_continuation=nearest_threshold,
        continuation_basis=str(continuation.get("basis_label", "overall")),
        continuation_rate=continuation_rate,
    )


def _nearest_touch_threshold(
    touch_target_lookup: Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, Dict[float, Dict[str, float | str]]]]]]],
    touched_side: str,
    close_side: str,
    touch_threshold_return: float,
) -> float:
    family_lookup = touch_target_lookup.get(touched_side, {}).get(close_side, {})
    available: set[float] = set()
    for family in family_lookup.values():
        for regime_thresholds in family.values():
            available.update(float(value) for value in regime_thresholds.keys())
    if not available:
        raise ValueError("No touch thresholds available in touch-target lookup")
    return min(sorted(available), key=lambda value: abs(value - touch_threshold_return))


def _nearest_close_threshold(
    touch_target_lookup: Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, Dict[float, Dict[str, float | str]]]]]]],
    touched_side: str,
    close_side: str,
    touch_threshold_return: float,
    close_threshold_return: float,
) -> float:
    family_lookup = touch_target_lookup.get(touched_side, {}).get(close_side, {})
    available: set[float] = set()
    for family in family_lookup.values():
        for regime_thresholds in family.values():
            if touch_threshold_return in regime_thresholds:
                available.update(float(value) for value in regime_thresholds[touch_threshold_return].keys())
    if not available:
        raise ValueError("No close thresholds available in touch-target lookup")
    return min(sorted(available), key=lambda value: abs(value - close_threshold_return))


def estimate_close_beyond_threshold_given_touch(
    *,
    prediction: LiveExcursionPrediction,
    touch_target_lookup: Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, Dict[float, Dict[str, float | str]]]]]]],
    touched_side: str,
    touch_threshold_return: float,
    close_side: str,
    close_threshold_return: float,
) -> ConditionalCloseEstimate:
    nearest_touch_threshold = _nearest_touch_threshold(
        touch_target_lookup,
        touched_side,
        close_side,
        touch_threshold_return,
    )
    nearest_close_threshold = _nearest_close_threshold(
        touch_target_lookup,
        touched_side,
        close_side,
        nearest_touch_threshold,
        close_threshold_return,
    )
    continuation = select_touch_target_stats(
        touch_target_lookup,
        touched_side=touched_side,
        close_side=close_side,
        touch_threshold=nearest_touch_threshold,
        close_threshold=nearest_close_threshold,
        regime_context=prediction.regime_context,
    )
    return ConditionalCloseEstimate(
        touched_side=touched_side,
        touch_threshold_return=touch_threshold_return,
        close_side=close_side,
        close_threshold_return=close_threshold_return,
        close_probability=float(continuation.get("close_rate", 0.0)),
        touch_threshold_used=nearest_touch_threshold,
        close_threshold_used=nearest_close_threshold,
        continuation_basis=str(continuation.get("basis_label", "overall")),
        avg_close_return=float(continuation.get("avg_close_return", 0.0)),
    )


def score_vertical_opportunities(
    *,
    prediction: LiveExcursionPrediction,
    continuation_lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]],
    vertical_estimates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    open_price = float(prediction.open_price)
    for _, vertical in vertical_estimates.iterrows():
        strategy = str(vertical["strategy"])
        lower_strike = float(vertical["lower_strike"])
        upper_strike = float(vertical["upper_strike"])
        width = float(vertical["width_points"])
        estimated_net = float(vertical["estimated_net_price"])
        underlying_price = float(vertical["underlying_price"])

        if strategy in {"bull_call_debit", "bear_call_credit"}:
            if lower_strike < open_price or upper_strike < open_price:
                continue
            near_threshold = (lower_strike - open_price) / open_price
            far_threshold = (upper_strike - open_price) / open_price
            near = estimate_close_beyond_threshold(
                prediction=prediction,
                continuation_lookup=continuation_lookup,
                side="upside",
                threshold_return=near_threshold,
            )
            far = estimate_close_beyond_threshold(
                prediction=prediction,
                continuation_lookup=continuation_lookup,
                side="upside",
                threshold_return=far_threshold,
            )
            expected_intrinsic = width * far.close_beyond_probability + 0.5 * width * max(
                near.close_beyond_probability - far.close_beyond_probability,
                0.0,
            )
            if strategy == "bull_call_debit":
                entry_price = estimated_net
                if entry_price <= 0:
                    continue
                expected_value = expected_intrinsic - entry_price
                max_risk = entry_price
                max_gain = max(width - entry_price, 0.0)
                outlook = "bullish debit"
            else:
                credit = -estimated_net
                if credit <= 0:
                    continue
                expected_value = credit - expected_intrinsic
                max_risk = max(width - credit, 0.0)
                max_gain = credit
                entry_price = credit
                outlook = "bearish credit"
        elif strategy in {"bear_put_debit", "bull_put_credit"}:
            if lower_strike > open_price or upper_strike > open_price:
                continue
            near_threshold = (open_price - upper_strike) / open_price
            far_threshold = (open_price - lower_strike) / open_price
            near = estimate_close_beyond_threshold(
                prediction=prediction,
                continuation_lookup=continuation_lookup,
                side="downside",
                threshold_return=near_threshold,
            )
            far = estimate_close_beyond_threshold(
                prediction=prediction,
                continuation_lookup=continuation_lookup,
                side="downside",
                threshold_return=far_threshold,
            )
            expected_intrinsic = width * far.close_beyond_probability + 0.5 * width * max(
                near.close_beyond_probability - far.close_beyond_probability,
                0.0,
            )
            if strategy == "bear_put_debit":
                entry_price = estimated_net
                if entry_price <= 0:
                    continue
                expected_value = expected_intrinsic - entry_price
                max_risk = entry_price
                max_gain = max(width - entry_price, 0.0)
                outlook = "bearish debit"
            else:
                credit = -estimated_net
                if credit <= 0:
                    continue
                expected_value = credit - expected_intrinsic
                max_risk = max(width - credit, 0.0)
                max_gain = credit
                entry_price = credit
                outlook = "bullish credit"
        else:
            continue

        if max_risk <= 0:
            continue

        predicted_terminal_value_proxy = expected_intrinsic
        predicted_profit_proxy = expected_value
        profit_to_cost_ratio_proxy = predicted_profit_proxy / max_risk if max_risk else 0.0
        cost_to_profit_ratio_proxy = (
            max_risk / predicted_profit_proxy if predicted_profit_proxy > 0 else float("inf")
        )

        rows.append(
            {
                "snapshot_id": str(vertical["snapshot_id"]),
                "timestamp": vertical["timestamp"],
                "strategy": strategy,
                "outlook": outlook,
                "underlying_price": underlying_price,
                "open_price": open_price,
                "lower_strike": lower_strike,
                "upper_strike": upper_strike,
                "width_points": width,
                "entry_price": entry_price,
                "max_risk": max_risk,
                "max_gain": max_gain,
                "expected_intrinsic": expected_intrinsic,
                "expected_value": expected_value,
                "ev_to_risk": expected_value / max_risk if max_risk else 0.0,
                "predicted_terminal_value_proxy": predicted_terminal_value_proxy,
                "predicted_profit_proxy": predicted_profit_proxy,
                "profit_to_cost_ratio_proxy": profit_to_cost_ratio_proxy,
                "cost_to_profit_ratio_proxy": cost_to_profit_ratio_proxy,
                "short_distance_points": float(vertical["short_distance_points"]),
                "short_distance_pct_spot": float(vertical["short_distance_pct_spot"]),
                "near_touch_probability": near.touch_probability,
                "far_touch_probability": far.touch_probability,
                "near_close_beyond_probability": near.close_beyond_probability,
                "far_close_beyond_probability": far.close_beyond_probability,
                "near_continuation_basis": near.continuation_basis,
                "far_continuation_basis": far.continuation_basis,
                "near_threshold_pct_from_open": near.threshold_return * 100.0,
                "far_threshold_pct_from_open": far.threshold_return * 100.0,
            }
        )
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored
    return scored.sort_values(["ev_to_risk", "expected_value"], ascending=False).reset_index(drop=True)


def score_vertical_opportunities_after_touch(
    *,
    prediction: LiveExcursionPrediction,
    touch_target_lookup: Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, Dict[float, Dict[str, float | str]]]]]]],
    vertical_estimates: pd.DataFrame,
    touched_side: str,
    touch_threshold_return: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    open_price = float(prediction.open_price)
    for _, vertical in vertical_estimates.iterrows():
        strategy = str(vertical["strategy"])
        lower_strike = float(vertical["lower_strike"])
        upper_strike = float(vertical["upper_strike"])
        width = float(vertical["width_points"])
        estimated_net = float(vertical["estimated_net_price"])
        underlying_price = float(vertical["underlying_price"])

        if strategy in {"bull_call_debit", "bear_call_credit"}:
            if lower_strike < open_price or upper_strike < open_price:
                continue
            close_side = "upside"
            near_threshold = (lower_strike - open_price) / open_price
            far_threshold = (upper_strike - open_price) / open_price
            near = estimate_close_beyond_threshold_given_touch(
                prediction=prediction,
                touch_target_lookup=touch_target_lookup,
                touched_side=touched_side,
                touch_threshold_return=touch_threshold_return,
                close_side=close_side,
                close_threshold_return=near_threshold,
            )
            far = estimate_close_beyond_threshold_given_touch(
                prediction=prediction,
                touch_target_lookup=touch_target_lookup,
                touched_side=touched_side,
                touch_threshold_return=touch_threshold_return,
                close_side=close_side,
                close_threshold_return=far_threshold,
            )
            expected_intrinsic = width * far.close_probability + 0.5 * width * max(
                near.close_probability - far.close_probability,
                0.0,
            )
            if strategy == "bull_call_debit":
                entry_price = estimated_net
                if entry_price <= 0:
                    continue
                expected_value = expected_intrinsic - entry_price
                max_risk = entry_price
                max_gain = max(width - entry_price, 0.0)
                outlook = "bullish continuation debit" if touched_side == "upside_touch" else "bullish fade debit"
            else:
                credit = -estimated_net
                if credit <= 0:
                    continue
                expected_value = credit - expected_intrinsic
                max_risk = max(width - credit, 0.0)
                max_gain = credit
                entry_price = credit
                outlook = "bearish fade credit" if touched_side == "upside_touch" else "bearish continuation credit"
        elif strategy in {"bear_put_debit", "bull_put_credit"}:
            if lower_strike > open_price or upper_strike > open_price:
                continue
            close_side = "downside"
            near_threshold = (open_price - upper_strike) / open_price
            far_threshold = (open_price - lower_strike) / open_price
            near = estimate_close_beyond_threshold_given_touch(
                prediction=prediction,
                touch_target_lookup=touch_target_lookup,
                touched_side=touched_side,
                touch_threshold_return=touch_threshold_return,
                close_side=close_side,
                close_threshold_return=near_threshold,
            )
            far = estimate_close_beyond_threshold_given_touch(
                prediction=prediction,
                touch_target_lookup=touch_target_lookup,
                touched_side=touched_side,
                touch_threshold_return=touch_threshold_return,
                close_side=close_side,
                close_threshold_return=far_threshold,
            )
            expected_intrinsic = width * far.close_probability + 0.5 * width * max(
                near.close_probability - far.close_probability,
                0.0,
            )
            if strategy == "bear_put_debit":
                entry_price = estimated_net
                if entry_price <= 0:
                    continue
                expected_value = expected_intrinsic - entry_price
                max_risk = entry_price
                max_gain = max(width - entry_price, 0.0)
                outlook = "bearish continuation debit" if touched_side == "downside_touch" else "bearish fade debit"
            else:
                credit = -estimated_net
                if credit <= 0:
                    continue
                expected_value = credit - expected_intrinsic
                max_risk = max(width - credit, 0.0)
                max_gain = credit
                entry_price = credit
                outlook = "bullish fade credit" if touched_side == "downside_touch" else "bullish continuation credit"
        else:
            continue

        if max_risk <= 0:
            continue

        predicted_terminal_value_proxy = expected_intrinsic
        predicted_profit_proxy = expected_value
        profit_to_cost_ratio_proxy = predicted_profit_proxy / max_risk if max_risk else 0.0
        cost_to_profit_ratio_proxy = (
            max_risk / predicted_profit_proxy if predicted_profit_proxy > 0 else float("inf")
        )

        rows.append(
            {
                "snapshot_id": str(vertical["snapshot_id"]),
                "timestamp": vertical["timestamp"],
                "strategy": strategy,
                "outlook": outlook,
                "conditioning_touch_side": touched_side,
                "conditioning_touch_threshold_pct_from_open": touch_threshold_return * 100.0,
                "underlying_price": underlying_price,
                "open_price": open_price,
                "lower_strike": lower_strike,
                "upper_strike": upper_strike,
                "width_points": width,
                "entry_price": entry_price,
                "max_risk": max_risk,
                "max_gain": max_gain,
                "expected_intrinsic": expected_intrinsic,
                "expected_value": expected_value,
                "ev_to_risk": expected_value / max_risk if max_risk else 0.0,
                "predicted_terminal_value_proxy": predicted_terminal_value_proxy,
                "predicted_profit_proxy": predicted_profit_proxy,
                "profit_to_cost_ratio_proxy": profit_to_cost_ratio_proxy,
                "cost_to_profit_ratio_proxy": cost_to_profit_ratio_proxy,
                "short_distance_points": float(vertical["short_distance_points"]),
                "short_distance_pct_spot": float(vertical["short_distance_pct_spot"]),
                "near_close_probability_given_touch": near.close_probability,
                "far_close_probability_given_touch": far.close_probability,
                "near_continuation_basis": near.continuation_basis,
                "far_continuation_basis": far.continuation_basis,
                "near_threshold_pct_from_open": near.close_threshold_return * 100.0,
                "far_threshold_pct_from_open": far.close_threshold_return * 100.0,
                "touch_threshold_used_pct": near.touch_threshold_used * 100.0,
                "near_close_threshold_used_pct": near.close_threshold_used * 100.0,
                "far_close_threshold_used_pct": far.close_threshold_used * 100.0,
                "near_avg_close_return_pct": near.avg_close_return * 100.0,
                "far_avg_close_return_pct": far.avg_close_return * 100.0,
            }
        )

    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored
    return scored.sort_values(["ev_to_risk", "expected_value"], ascending=False).reset_index(drop=True)
