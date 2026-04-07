from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Deque, Dict, List, Sequence, Tuple

from .data import DailyBar
from .events import collect_event_names, event_flags


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _rsi(closes: Sequence[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for idx in range(len(closes) - period, len(closes)):
        change = closes[idx] - closes[idx - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _lagged(target: Sequence[float], index: int, lag: int) -> float:
    position = index - lag
    if position < 0:
        return 0.0
    return target[position]


@dataclass
class FeatureRow:
    date: date
    feature_names: List[str]
    values: List[float]
    target_close: float
    target_return: float
    target_high_return: float
    target_low_return: float
    open_price: float
    high_price: float
    low_price: float
    actual_close: float
    metadata: Dict[str, float]


def build_feature_rows(
    spx_bars: Sequence[DailyBar],
    vix_bars: Sequence[DailyBar],
    event_calendar: Dict[date, set[str]],
    max_lag: int = 5,
) -> List[FeatureRow]:
    if len(spx_bars) != len(vix_bars):
        raise ValueError("SPX and VIX series must already be aligned by date")

    spx_closes = [bar.close for bar in spx_bars]
    spx_opens = [bar.open for bar in spx_bars]
    spx_highs = [bar.high for bar in spx_bars]
    spx_lows = [bar.low for bar in spx_bars]
    vix_closes = [bar.close for bar in vix_bars]
    vix_opens = [bar.open for bar in vix_bars]
    dates = [bar.date for bar in spx_bars]

    known_events = collect_event_names(event_calendar)
    rows: List[FeatureRow] = []

    close_window_5: Deque[float] = deque(maxlen=5)
    close_window_10: Deque[float] = deque(maxlen=10)
    close_window_20: Deque[float] = deque(maxlen=20)
    return_window_5: Deque[float] = deque(maxlen=5)
    return_window_10: Deque[float] = deque(maxlen=10)

    for idx in range(len(spx_bars)):
        prev_close = _lagged(spx_closes, idx, 1)
        current_close = spx_closes[idx]
        current_open = spx_opens[idx]
        current_high = spx_highs[idx]
        current_low = spx_lows[idx]
        current_vix = vix_closes[idx]
        prev_vix = _lagged(vix_closes, idx, 1)

        day_return = _safe_div(current_close - prev_close, prev_close)
        overnight_gap = _safe_div(current_open - prev_close, prev_close)
        intraday_return = _safe_div(current_close - current_open, current_open)
        high_return = _safe_div(current_high - current_open, current_open)
        low_return = _safe_div(current_low - current_open, current_open)
        range_pct = _safe_div(current_high - current_low, current_open)
        vix_return = _safe_div(current_vix - prev_vix, prev_vix)

        close_window_5.append(current_close)
        close_window_10.append(current_close)
        close_window_20.append(current_close)
        return_window_5.append(day_return)
        return_window_10.append(day_return)

        rolling_mean_5 = _mean(list(close_window_5))
        rolling_mean_10 = _mean(list(close_window_10))
        rolling_mean_20 = _mean(list(close_window_20))
        rolling_std_5 = _std(list(return_window_5))
        rolling_std_10 = _std(list(return_window_10))
        close_momentum_5 = _safe_div(current_close - _lagged(spx_closes, idx, 5), _lagged(spx_closes, idx, 5))
        close_momentum_10 = _safe_div(current_close - _lagged(spx_closes, idx, 10), _lagged(spx_closes, idx, 10))
        vix_momentum_5 = _safe_div(current_vix - _lagged(vix_closes, idx, 5), _lagged(vix_closes, idx, 5))
        rsi_14 = _rsi(spx_closes[: idx + 1], 14)

        realized_features: Dict[str, float] = {
            "spx_open": current_open,
            "spx_high": current_high,
            "spx_low": current_low,
            "spx_close": current_close,
            "spx_prev_close": prev_close,
            "spx_day_return": day_return,
            "spx_overnight_gap": overnight_gap,
            "spx_intraday_return": intraday_return,
            "spx_range_pct": range_pct,
            "spx_rolling_mean_5": rolling_mean_5,
            "spx_rolling_mean_10": rolling_mean_10,
            "spx_rolling_mean_20": rolling_mean_20,
            "spx_rolling_std_5": rolling_std_5,
            "spx_rolling_std_10": rolling_std_10,
            "spx_momentum_5": close_momentum_5,
            "spx_momentum_10": close_momentum_10,
            "spx_rsi_14": rsi_14,
            "vix_open": vix_opens[idx],
            "vix_high": vix_bars[idx].high,
            "vix_low": vix_bars[idx].low,
            "vix_close": current_vix,
            "vix_prev_close": prev_vix,
            "vix_return": vix_return,
            "vix_momentum_5": vix_momentum_5,
        }
        realized_features.update(event_flags(dates[idx], event_calendar, known_events))

        open_time_features: Dict[str, float] = {
            "current_spx_open": current_open,
            "current_spx_prev_close": prev_close,
            "current_spx_overnight_gap": overnight_gap,
            "current_vix_open": vix_opens[idx],
            "current_vix_prev_close": prev_vix,
        }
        open_time_features.update(event_flags(dates[idx], event_calendar, known_events))

        feature_values: Dict[str, float] = {}
        feature_values.update(open_time_features)
        for name in realized_features:
            for lag in range(1, max_lag + 1):
                lagged_value = 0.0
                if idx - lag >= 0:
                    prior_bar_features = rows[idx - lag].metadata
                    lagged_value = prior_bar_features.get(name, 0.0)
                feature_values[f"{name}_lag_{lag}"] = lagged_value

        feature_names = sorted(feature_values)
        values = [feature_values[name] for name in feature_names]
        rows.append(
            FeatureRow(
                date=dates[idx],
                feature_names=feature_names,
                values=values,
                target_close=current_close,
                target_return=intraday_return,
                target_high_return=high_return,
                target_low_return=low_return,
                open_price=current_open,
                high_price=current_high,
                low_price=current_low,
                actual_close=current_close,
                metadata=realized_features,
            )
        )

    warmup = max(20, max_lag + 1)
    return rows[warmup:]


def align_and_build(
    spx_bars: Sequence[DailyBar],
    vix_bars: Sequence[DailyBar],
    event_calendar: Dict[date, set[str]],
    max_lag: int = 5,
) -> List[FeatureRow]:
    vix_by_date = {bar.date: bar for bar in vix_bars}
    aligned_spx: List[DailyBar] = []
    aligned_vix: List[DailyBar] = []
    for bar in spx_bars:
        match = vix_by_date.get(bar.date)
        if match is None:
            continue
        aligned_spx.append(bar)
        aligned_vix.append(match)
    return build_feature_rows(aligned_spx, aligned_vix, event_calendar, max_lag=max_lag)
