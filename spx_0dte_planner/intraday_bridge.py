from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from intraday_condor_research.io import load_intraday_frame
from intraday_condor_research.session import filter_rth, keep_complete_sessions

from .data import DailyBar
from .events import load_event_calendar
from .features import FeatureRow, align_and_build
from .live import (
    build_live_feature_row,
    classify_regime_context,
    compute_regime_cutoffs,
    predict_live_excursions,
)
from .model import fit_train_backtest_range_model
from .opportunity_screen import estimate_close_beyond_threshold


@dataclass
class IntradayBridgeResult:
    rows: pd.DataFrame
    summary: pd.DataFrame


def _parse_thresholds(raw: Iterable[float]) -> list[float]:
    return sorted(float(value) for value in raw)


def _build_move_bucket_labels(edges: Sequence[float]) -> list[str]:
    labels: list[str] = []
    full_edges = [-np.inf, *edges, np.inf]
    for low, high in zip(full_edges[:-1], full_edges[1:]):
        if np.isneginf(low):
            labels.append(f"<= {high * 100.0:.2f}%")
        elif np.isposinf(high):
            labels.append(f"> {low * 100.0:.2f}%")
        else:
            labels.append(f"{low * 100.0:.2f}% to {high * 100.0:.2f}%")
    return labels


def _bucket_move(value: float, edges: Sequence[float]) -> str:
    labels = _build_move_bucket_labels(edges)
    bins = [-np.inf, *edges, np.inf]
    index = int(np.digitize([value], bins[1:-1], right=True)[0])
    return labels[index]


def prepare_intraday_structure_frame(
    intraday: pd.DataFrame,
    *,
    checkpoints: Sequence[str],
    move_bucket_edges: Sequence[float],
) -> pd.DataFrame:
    working = intraday.copy()
    working["session_date"] = pd.to_datetime(working["timestamp"]).dt.date.astype(str)
    working["clock"] = pd.to_datetime(working["timestamp"]).dt.strftime("%H:%M")
    rows: list[dict[str, float | str]] = []
    for session_date, day_frame in working.groupby("session_date", sort=True):
        ordered = day_frame.sort_values("timestamp").reset_index(drop=True)
        session_open = float(ordered.iloc[0]["open"])
        session_close = float(ordered.iloc[-1]["close"])
        if session_open <= 0:
            continue
        for checkpoint in checkpoints:
            checkpoint_rows = ordered[ordered["clock"] == checkpoint]
            if checkpoint_rows.empty:
                continue
            checkpoint_row = checkpoint_rows.iloc[-1]
            checkpoint_close = float(checkpoint_row["close"])
            move_from_open_pct = (checkpoint_close - session_open) / session_open
            close_from_open_pct = (session_close - session_open) / session_open
            rows.append(
                {
                    "session_date": str(session_date),
                    "checkpoint": checkpoint,
                    "checkpoint_close": checkpoint_close,
                    "session_open": session_open,
                    "session_close": session_close,
                    "move_from_open_pct": move_from_open_pct,
                    "move_bucket": _bucket_move(move_from_open_pct, move_bucket_edges),
                    "close_from_open_pct": close_from_open_pct,
                }
            )
    return pd.DataFrame(rows)


def translate_proxy_intraday_to_target(
    proxy_intraday: pd.DataFrame,
    *,
    target_daily_bars: Sequence[DailyBar],
    target_symbol: str = "SPX",
) -> pd.DataFrame:
    working = proxy_intraday.copy()
    working["session_date"] = pd.to_datetime(working["timestamp"]).dt.date
    target_open_lookup = {bar.date: float(bar.open) for bar in target_daily_bars}
    translated_rows: list[pd.DataFrame] = []
    for session_date, day_frame in working.groupby("session_date", sort=True):
        target_open = target_open_lookup.get(session_date)
        if target_open is None or target_open <= 0:
            continue
        ordered = day_frame.sort_values("timestamp").reset_index(drop=True).copy()
        proxy_open = float(ordered.iloc[0]["open"])
        if proxy_open <= 0:
            continue
        scale = target_open / proxy_open
        for column in ["open", "high", "low", "close"]:
            ordered[column] = ordered[column].astype(float) * scale
        if "symbol" in ordered.columns:
            ordered["proxy_symbol"] = ordered["symbol"]
        ordered["symbol"] = target_symbol
        ordered["target_session_open"] = target_open
        ordered["proxy_session_open"] = proxy_open
        ordered["proxy_to_target_scale"] = scale
        translated_rows.append(ordered)
    if not translated_rows:
        raise ValueError("No overlapping session dates between proxy intraday data and target daily bars")
    translated = pd.concat(translated_rows, ignore_index=True)
    translated["session_date"] = translated["session_date"].astype(str)
    return translated


def build_intraday_close_lookup(
    structure_rows: pd.DataFrame,
    *,
    thresholds: Sequence[float],
    checkpoints: Sequence[str],
    move_bucket_edges: Sequence[float],
) -> Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float]]]]]:
    move_buckets = _build_move_bucket_labels(move_bucket_edges)
    lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float]]]]] = {
        "upside": {checkpoint: {} for checkpoint in checkpoints},
        "downside": {checkpoint: {} for checkpoint in checkpoints},
    }
    for side in ["upside", "downside"]:
        for checkpoint in checkpoints:
            checkpoint_rows = structure_rows[structure_rows["checkpoint"] == checkpoint]
            for bucket in move_buckets:
                subset = checkpoint_rows[checkpoint_rows["move_bucket"] == bucket]
                threshold_map: Dict[float, Dict[str, float]] = {}
                for threshold in thresholds:
                    if side == "upside":
                        actuals = (subset["close_from_open_pct"] >= threshold).astype(float) if not subset.empty else pd.Series(dtype=float)
                    else:
                        actuals = (subset["close_from_open_pct"] <= -threshold).astype(float) if not subset.empty else pd.Series(dtype=float)
                    threshold_map[float(threshold)] = {
                        "samples": float(len(subset)),
                        "actual_rate": float(actuals.mean()) if len(actuals) else 0.0,
                    }
                lookup[side][checkpoint][bucket] = threshold_map
    return lookup


def build_actual_continuation_lookup(
    rows: Sequence[FeatureRow],
    *,
    thresholds: Sequence[float],
    cutoffs,
) -> Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]]:
    lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]] = {
        "upside": {"overall": {"overall": {}}, "vix_regime": {}, "range_regime": {}, "gap_regime": {}, "combo_regime": {}},
        "downside": {"overall": {"overall": {}}, "vix_regime": {}, "range_regime": {}, "gap_regime": {}, "combo_regime": {}},
    }
    enriched: list[dict[str, object]] = []
    name_index = {name: idx for idx, name in enumerate(rows[0].feature_names)} if rows else {}

    def feature(row: FeatureRow, name: str) -> float:
        return row.values[name_index[name]]

    for row in rows:
        regime = classify_regime_context(
            prediction_date=row.date,
            current_vix_open=feature(row, "current_vix_open"),
            prev_range_pct=feature(row, "spx_range_pct_lag_1"),
            overnight_gap=feature(row, "current_spx_overnight_gap"),
            cutoffs=cutoffs,
        )
        enriched.append(
            {
                "vix_regime": regime.vix_regime,
                "range_regime": regime.range_regime,
                "gap_regime": regime.gap_regime,
                "combo_regime": f"{regime.vix_regime}|{regime.range_regime}|{regime.gap_regime}",
                "actual_high_return": row.target_high_return,
                "actual_low_return": row.target_low_return,
                "actual_close_return": row.target_return,
            }
        )

    def summarize(subset: list[dict[str, object]], side: str, threshold: float) -> dict[str, float | str]:
        if side == "upside":
            touched = [item for item in subset if float(item["actual_high_return"]) >= threshold]
            close_returns = np.asarray([float(item["actual_close_return"]) for item in touched], dtype=float)
            return {
                "samples": float(len(touched)),
                "close_above_touch_rate": float(np.mean(close_returns >= threshold)) if len(close_returns) else 0.0,
            }
        touched = [item for item in subset if float(item["actual_low_return"]) <= -threshold]
        close_returns = np.asarray([float(item["actual_close_return"]) for item in touched], dtype=float)
        return {
            "samples": float(len(touched)),
            "close_below_touch_rate": float(np.mean(close_returns <= -threshold)) if len(close_returns) else 0.0,
        }

    buckets = {
        "vix_regime": ["low_vix", "mid_vix", "high_vix"],
        "range_regime": ["low_prev_range", "mid_prev_range", "high_prev_range"],
        "gap_regime": ["flat_gap", "small_gap", "medium_gap", "large_gap"],
        "combo_regime": sorted({str(item["combo_regime"]) for item in enriched}),
    }
    for side in ["upside", "downside"]:
        for threshold in thresholds:
            lookup[side]["overall"]["overall"][float(threshold)] = summarize(enriched, side, threshold)
        for family, values in buckets.items():
            family_map: Dict[str, Dict[float, Dict[str, float | str]]] = {}
            for value in values:
                subset = [item for item in enriched if item[family] == value]
                family_map[value] = {
                    float(threshold): dict(summarize(subset, side, threshold), basis_label=f"{family}: {value}")
                    for threshold in thresholds
                }
            lookup[side][family] = family_map
    return lookup


def _combine_probabilities(prior_probability: float, intraday_probability: float, intraday_samples: float, prior_strength: float) -> float:
    intraday_weight = max(intraday_samples, 0.0)
    return (prior_strength * prior_probability + intraday_weight * intraday_probability) / (prior_strength + intraday_weight)


def backtest_intraday_excursion_bridge(
    *,
    intraday_frame: pd.DataFrame,
    spx_bars: Sequence[DailyBar],
    vix_bars: Sequence[DailyBar],
    events_path: str,
    checkpoints: Sequence[str],
    thresholds: Sequence[float],
    move_bucket_edges: Sequence[float],
    train_end_min_rows: int = 120,
    max_lag: int = 5,
    pca_variance_ratio: float = 0.95,
    prior_strength: float = 20.0,
) -> IntradayBridgeResult:
    thresholds = _parse_thresholds(thresholds)
    events = load_event_calendar(events_path)
    daily_rows = align_and_build(spx_bars, vix_bars, events, max_lag=max_lag)
    structure_rows = prepare_intraday_structure_frame(
        intraday_frame,
        checkpoints=checkpoints,
        move_bucket_edges=move_bucket_edges,
    )
    if structure_rows.empty:
        raise ValueError("No usable intraday structure rows were produced")

    results: list[dict[str, float | str]] = []
    session_dates = sorted(pd.to_datetime(structure_rows["session_date"]).dt.date.unique())
    for session_date in session_dates:
        train_daily_rows = [row for row in daily_rows if row.date < session_date]
        train_intraday_rows = structure_rows[pd.to_datetime(structure_rows["session_date"]).dt.date < session_date]
        if len(train_daily_rows) < train_end_min_rows or train_intraday_rows["session_date"].nunique() < 20:
            continue

        fit_result = fit_train_backtest_range_model(
            train_daily_rows,
            train_ratio=0.8,
            pca_variance_ratio=pca_variance_ratio,
        )
        cutoffs = compute_regime_cutoffs(fit_result.train_rows)
        continuation_lookup = build_actual_continuation_lookup(fit_result.train_rows, thresholds=thresholds, cutoffs=cutoffs)
        intraday_lookup = build_intraday_close_lookup(
            train_intraday_rows,
            thresholds=thresholds,
            checkpoints=checkpoints,
            move_bucket_edges=move_bucket_edges,
        )

        spx_today = next((bar for bar in spx_bars if bar.date == session_date), None)
        vix_today = next((bar for bar in vix_bars if bar.date == session_date), None)
        if spx_today is None or vix_today is None:
            continue
        live_row = build_live_feature_row(
            spx_bars,
            vix_bars,
            events,
            prediction_date=session_date,
            current_spx_open=spx_today.open,
            current_vix_open=vix_today.open,
            max_lag=max_lag,
            selected_events=[],
        )
        live_prediction = predict_live_excursions(fit_result, live_row, cutoffs)

        day_rows = structure_rows[pd.to_datetime(structure_rows["session_date"]).dt.date == session_date]
        for _, row in day_rows.iterrows():
            checkpoint = str(row["checkpoint"])
            move_bucket = str(row["move_bucket"])
            close_from_open_pct = float(row["close_from_open_pct"])
            for side in ["upside", "downside"]:
                for threshold in thresholds:
                    prior = estimate_close_beyond_threshold(
                        prediction=live_prediction,
                        continuation_lookup=continuation_lookup,
                        side=side,
                        threshold_return=threshold,
                    )
                    intraday_cell = intraday_lookup[side][checkpoint][move_bucket][threshold]
                    combined = _combine_probabilities(
                        prior_probability=prior.close_beyond_probability,
                        intraday_probability=float(intraday_cell["actual_rate"]),
                        intraday_samples=float(intraday_cell["samples"]),
                        prior_strength=prior_strength,
                    )
                    actual = 1.0 if (close_from_open_pct >= threshold if side == "upside" else close_from_open_pct <= -threshold) else 0.0
                    results.append(
                        {
                            "session_date": str(session_date),
                            "checkpoint": checkpoint,
                            "move_bucket": move_bucket,
                            "side": side,
                            "threshold_return": threshold,
                            "threshold_pct": threshold * 100.0,
                            "daily_prior_probability": prior.close_beyond_probability,
                            "intraday_probability": float(intraday_cell["actual_rate"]),
                            "intraday_samples": float(intraday_cell["samples"]),
                            "combined_probability": combined,
                            "actual": actual,
                        }
                    )
    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        return IntradayBridgeResult(rows=result_frame, summary=pd.DataFrame())
    summary_rows = []
    for model_name in ["daily_prior_probability", "intraday_probability", "combined_probability"]:
        for side in ["upside", "downside"]:
            subset = result_frame[result_frame["side"] == side]
            brier = float(np.mean((subset[model_name] - subset["actual"]) ** 2))
            summary_rows.append(
                {
                    "model": model_name,
                    "side": side,
                    "samples": float(len(subset)),
                    "avg_probability": float(subset[model_name].mean()),
                    "actual_rate": float(subset["actual"].mean()),
                    "brier": brier,
                }
            )
    return IntradayBridgeResult(rows=result_frame, summary=pd.DataFrame(summary_rows))


def load_spx_intraday_frame(path: str) -> pd.DataFrame:
    raw = load_intraday_frame(path)
    rth = filter_rth(raw, start_time="09:30", end_time="16:00")
    return keep_complete_sessions(rth, checkpoints=["10:00", "10:30", "12:00", "14:00", "15:00", "15:30"])
