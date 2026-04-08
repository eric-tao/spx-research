from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any

from intraday_condor_research.vertical_fills import estimate_vertical_fills_from_quotes, load_vertical_fill_inputs
from python_api.schemas import ScenarioRequest
from spx_0dte_planner.live import PriorDayOverrides, build_live_feature_row, default_prediction_date, predict_live_excursions
from spx_0dte_planner.opportunity_screen import score_vertical_opportunities_after_touch
from spx_0dte_planner.webapp import (
    AppState,
    _build_featured_playbook,
    _choose_best_overall_vertical,
    _filter_vertical_estimates_for_current_spot,
    _reanchor_vertical_estimates_to_spot,
    _score_decision_items,
    _touch_consistency_message,
    create_app_state,
)
from spx_0dte_planner.live import select_continuation_stats, threshold_probabilities


@dataclass(frozen=True)
class ApiConfig:
    spx_path: str = "data/spx_daily.csv"
    vix_path: str = "data/vix_daily.csv"
    events_path: str = "data/events.csv"
    underlying_label: str = "SPX"
    train_end_date: str = "2024-12-31"
    max_lag: int = 5
    pca_variance_ratio: float = 0.95
    thresholds: tuple[float, ...] = tuple(0.0025 + 0.0005 * idx for idx in range(16))
    fill_samples_path: str = "data/tradingview_fill_samples_template.csv"
    quotes_path: str = "data/tradingview_single_option_quotes_template.csv"
    vix_snapshots_path: str = "data/tradingview_vix_snapshots_template.csv"
    auto_refresh: bool = False


def _threshold_rows(state: AppState, prediction) -> list[dict[str, float]]:
    return threshold_probabilities(prediction, state.thresholds)


def _touch_confirmation_source(
    *,
    touched_side: str,
    touch_price: float,
    current_spot: float,
    high_so_far: float,
    low_so_far: float,
) -> tuple[bool, str]:
    if touched_side == "upside_touch":
        if high_so_far >= touch_price:
            return True, "high_so_far"
        if current_spot >= touch_price:
            return False, "spot_only"
        return False, "not_confirmed"
    if low_so_far <= touch_price:
        return True, "low_so_far"
    if current_spot <= touch_price:
        return False, "spot_only"
    return False, "not_confirmed"


def _build_touch_table(state: AppState, side: str, threshold_rows: list[dict[str, float]], prediction) -> list[dict[str, Any]]:
    context = prediction.regime_context
    current_probability_key = "upside_probability" if side == "upside" else "downside_probability"
    rows: list[dict[str, Any]] = []
    for item in threshold_rows:
        threshold = float(item["threshold_return"])
        overall = state.regime_lookup[side]["overall"]["overall"][threshold]
        vix = state.regime_lookup[side]["vix_regime"][context.vix_regime][threshold]
        gap = state.regime_lookup[side]["gap_regime"][context.gap_regime][threshold]
        prev_range = state.regime_lookup[side]["range_regime"][context.range_regime][threshold]
        touch_price = prediction.open_price * (1.0 + threshold if side == "upside" else 1.0 - threshold)
        rows.append(
            {
                "thresholdPct": float(item["threshold_pct"]),
                "thresholdReturn": threshold,
                "points": prediction.open_price * threshold,
                "touchPrice": touch_price,
                "todayProbability": float(item[current_probability_key]),
                "overallHitRate": float(overall["actual_rate"]),
                "regimeHitRates": {
                    "vix": float(vix["actual_rate"]),
                    "gap": float(gap["actual_rate"]),
                    "priorRange": float(prev_range["actual_rate"]),
                },
                "regimeSamples": {
                    "vix": float(vix["samples"]),
                    "gap": float(gap["samples"]),
                    "priorRange": float(prev_range["samples"]),
                },
            }
        )
    return rows


def _build_continuation_table(state: AppState, side: str, threshold_rows: list[dict[str, float]], prediction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in threshold_rows:
        threshold = float(item["threshold_return"])
        continuation = select_continuation_stats(
            state.continuation_lookup,
            side,
            threshold,
            prediction.regime_context,
        )
        rows.append(
            {
                "thresholdPct": float(item["threshold_pct"]),
                "thresholdReturn": threshold,
                "basis": str(continuation.get("basis_label", "overall")),
                "samples": float(continuation.get("samples", 0.0)),
                "avgCloseReturn": float(continuation.get("avg_close_return", 0.0)),
                "closeReturnQ25": float(continuation.get("close_return_q25", 0.0)),
                "closeReturnQ75": float(continuation.get("close_return_q75", 0.0)),
                "closeOnSideRate": float(
                    continuation.get("close_above_open_rate" if side == "upside" else "close_below_open_rate", 0.0)
                ),
                "closePastTouchRate": float(
                    continuation.get("close_above_touch_rate" if side == "upside" else "close_below_touch_rate", 0.0)
                ),
            }
        )
    return rows


def _value_bucket(summary: dict[str, Any], request: ScenarioRequest) -> tuple[str, str]:
    predicted_profit = float(summary.get("predictedProfitProxy", 0.0))
    ratio = float(summary.get("profitToRiskRatioProxy", 0.0))
    strong_profit = request.valueBreakpoints.strongProfitThreshold
    strong_ratio = request.valueBreakpoints.strongRatioThreshold
    watch_profit = request.valueBreakpoints.watchProfitThreshold
    watch_ratio = request.valueBreakpoints.watchRatioThreshold
    if predicted_profit <= 0:
        return "pass", "Modeled value is not there after execution cost."
    if predicted_profit >= strong_profit and ratio >= strong_ratio:
        return "strong value", "This is the kind of setup where the modeled payout is meaningfully ahead of cost."
    if predicted_profit >= watch_profit and ratio >= watch_ratio:
        return "watch", "There is modeled value here, but the edge is thinner and execution matters more."
    return "thin", "The setup is positive on paper, but only modestly so."


@lru_cache(maxsize=2)
def get_app_state(config: ApiConfig = ApiConfig()) -> AppState:
    return create_app_state(
        spx_path=config.spx_path,
        vix_path=config.vix_path,
        events_path=config.events_path,
        underlying_label=config.underlying_label,
        train_end_date=config.train_end_date,
        max_lag=config.max_lag,
        pca_variance_ratio=config.pca_variance_ratio,
        thresholds=config.thresholds,
        auto_refresh=config.auto_refresh,
        fill_samples_path=config.fill_samples_path,
        quotes_path=config.quotes_path,
        vix_snapshots_path=config.vix_snapshots_path,
    )


def build_bootstrap_payload(state: AppState, config: ApiConfig) -> dict[str, Any]:
    return {
        "underlyingLabel": state.underlying_label,
        "latestCommonHistoryDate": state.latest_common_history_date.isoformat() if state.latest_common_history_date else None,
        "suggestedTradeDate": default_prediction_date(state.spx_bars, state.vix_bars).isoformat(),
        "defaultThresholdsPct": [threshold * 100.0 for threshold in state.thresholds],
        "eventNames": state.event_names,
        "defaults": {
            "touchedSide": "upside_touch",
            "touchedThresholdPct": 0.50,
            "verticalWidthPoints": 10.0,
            "strongProfitThreshold": 1.50,
            "strongRatioThreshold": 0.75,
            "watchProfitThreshold": 0.75,
            "watchRatioThreshold": 0.35,
        },
        "dataStatus": {
            "verticalInputsAvailable": state.vertical_inputs_available,
            "refreshNote": state.refresh_note,
        },
        "modelConfig": {
            "trainEndDate": config.train_end_date,
            "maxLag": config.max_lag,
            "pcaVarianceRatio": config.pca_variance_ratio,
            "thresholdsPct": [threshold * 100.0 for threshold in config.thresholds],
        },
    }


def build_scenario_payload(state: AppState, config: ApiConfig, request: ScenarioRequest) -> dict[str, Any]:
    overrides = PriorDayOverrides(
        spx_open=request.priorDayOverrides.spxOpen,
        spx_high=request.priorDayOverrides.spxHigh,
        spx_low=request.priorDayOverrides.spxLow,
        spx_close=request.priorDayOverrides.spxClose,
        vix_open=request.priorDayOverrides.vixOpen,
        vix_high=request.priorDayOverrides.vixHigh,
        vix_low=request.priorDayOverrides.vixLow,
        vix_close=request.priorDayOverrides.vixClose,
    )
    live_row = build_live_feature_row(
        state.spx_bars,
        state.vix_bars,
        state.event_calendar,
        prediction_date=date.fromisoformat(request.predictionDate),
        current_spx_open=request.spxOpen,
        current_vix_open=request.vixOpen,
        max_lag=state.max_lag,
        selected_events=request.selectedEvents,
        prior_day_overrides=overrides,
    )
    prediction = predict_live_excursions(state.range_fit, live_row, state.cutoffs)
    threshold_rows = _threshold_rows(state, prediction)

    touch_price = request.spxOpen * (
        1.0 + request.touchSelection.touchedThresholdPct / 100.0
        if request.touchSelection.touchedSide == "upside_touch"
        else 1.0 - request.touchSelection.touchedThresholdPct / 100.0
    )
    touch_confirmed, touch_source = _touch_confirmation_source(
        touched_side=request.touchSelection.touchedSide,
        touch_price=touch_price,
        current_spot=request.currentSpot,
        high_so_far=request.highSoFar,
        low_so_far=request.lowSoFar,
    )

    vertical_strategy: dict[str, Any] = {
        "mode": "unavailable",
        "pricingProvenance": {
            "pricingMode": "unavailable",
            "reanchored": False,
            "filteredCandidateCount": 0,
            "sourceSnapshotCount": 0,
            "maxSnapshotGapPct": None,
        },
        "summary": None,
        "ranked": [],
        "notes": [],
    }
    if state.vertical_inputs_available:
        fill_samples, quote_snapshots, vix_by_snapshot = load_vertical_fill_inputs(
            state.fill_samples_path,
            state.quotes_path,
            state.vix_snapshots_path,
        )
        vertical_estimates = estimate_vertical_fills_from_quotes(
            fill_samples=fill_samples,
            quote_snapshots=quote_snapshots,
            vix_by_snapshot=vix_by_snapshot,
            width_points=request.verticalSelection.widthPoints,
        )
        filtered_verticals = _filter_vertical_estimates_for_current_spot(vertical_estimates, current_spot=request.currentSpot)
        candidate_verticals = filtered_verticals
        pricing_mode = "quote_nearby"
        notes: list[str] = []
        reanchored = False
        if filtered_verticals.empty and not vertical_estimates.empty:
            candidate_verticals = _reanchor_vertical_estimates_to_spot(vertical_estimates, current_spot=request.currentSpot)
            pricing_mode = "reanchored_percent_distance"
            reanchored = True
            notes.append(
                "No quote snapshot was close enough to current SPX spot, so the backend fell back to a re-anchored stale-chain proxy."
            )
        filtered_count = len(vertical_estimates) - len(filtered_verticals)
        if filtered_count > 0:
            notes.append(
                f"Filtered out {filtered_count} vertical candidates because their quote snapshot underlying was more than 5% away from current SPX spot."
            )
        scored_verticals = score_vertical_opportunities_after_touch(
            prediction=prediction,
            touch_target_lookup=state.touch_target_lookup,
            vertical_estimates=candidate_verticals,
            touched_side=request.touchSelection.touchedSide,
            touch_threshold_return=request.touchSelection.touchedThresholdPct / 100.0,
        )
        best_vertical = _choose_best_overall_vertical(scored_verticals)
        summary = None
        if best_vertical is not None:
            summary = {
                "strategy": str(best_vertical["strategy"]),
                "outlook": str(best_vertical["outlook"]),
                "lowerStrike": float(best_vertical["lower_strike"]),
                "upperStrike": float(best_vertical["upper_strike"]),
                "entryPrice": float(best_vertical["entry_price"]),
                "predictedTerminalValueProxy": float(best_vertical["predicted_terminal_value_proxy"]),
                "predictedProfitProxy": float(best_vertical["predicted_profit_proxy"]),
                "profitToRiskRatioProxy": float(best_vertical["profit_to_cost_ratio_proxy"]),
            }
            value_bucket, explanation = _value_bucket(summary, request)
            summary["valueBucket"] = value_bucket
            summary["valueBucketExplanation"] = explanation
        vertical_strategy = {
            "mode": pricing_mode,
            "pricingProvenance": {
                "pricingMode": pricing_mode,
                "reanchored": reanchored,
                "filteredCandidateCount": filtered_count,
                "sourceSnapshotCount": int(candidate_verticals["snapshot_id"].nunique()) if not candidate_verticals.empty else 0,
                "maxSnapshotGapPct": 5.0,
            },
            "summary": summary,
            "ranked": [] if scored_verticals.empty else scored_verticals.head(8).to_dict(orient="records"),
            "notes": notes,
        }

    return {
        "forecast": {
            "predictedHighFromOpenPct": prediction.predicted_high_return * 100.0,
            "predictedLowFromOpenPct": prediction.predicted_low_return * 100.0,
            "predictedHighPrice": prediction.predicted_high_price,
            "predictedLowPrice": prediction.predicted_low_price,
        },
        "intradayState": {
            "currentMovePct": ((request.currentSpot - request.spxOpen) / request.spxOpen) * 100.0,
            "highMovePct": ((request.highSoFar - request.spxOpen) / request.spxOpen) * 100.0,
            "lowMovePct": ((request.lowSoFar - request.spxOpen) / request.spxOpen) * 100.0,
            "touchPrice": touch_price,
            "touchConfirmed": touch_confirmed,
            "touchConfirmationSource": touch_source,
            "touchConsistencyLabel": _touch_consistency_message(
                touched_side=request.touchSelection.touchedSide,
                touch_price=touch_price,
                current_spot=request.currentSpot,
                high_so_far=request.highSoFar,
                low_so_far=request.lowSoFar,
            ),
        },
        "regimeContext": {
            "weekday": prediction.regime_context.weekday,
            "vixRegime": prediction.regime_context.vix_regime,
            "rangeRegime": prediction.regime_context.range_regime,
            "gapRegime": prediction.regime_context.gap_regime,
        },
        "decisionSummary": {
            "upside": _score_decision_items(state, "upside", threshold_rows, prediction)[:5],
            "downside": _score_decision_items(state, "downside", threshold_rows, prediction)[:5],
        },
        "featuredPlaybooks": {
            "upside": _build_featured_playbook(state, "upside", threshold_rows, prediction),
            "downside": _build_featured_playbook(state, "downside", threshold_rows, prediction),
        },
        "touchTables": {
            "upside": _build_touch_table(state, "upside", threshold_rows, prediction),
            "downside": _build_touch_table(state, "downside", threshold_rows, prediction),
        },
        "continuationTables": {
            "upside": _build_continuation_table(state, "upside", threshold_rows, prediction),
            "downside": _build_continuation_table(state, "downside", threshold_rows, prediction),
        },
        "verticalStrategy": vertical_strategy,
        "modelConfig": {
            "trainEndDate": config.train_end_date,
            "maxLag": config.max_lag,
            "pcaVarianceRatio": config.pca_variance_ratio,
            "thresholdsPct": [threshold * 100.0 for threshold in config.thresholds],
        },
    }
