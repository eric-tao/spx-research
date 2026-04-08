import pandas as pd

from spx_0dte_planner.backtest import DebitSpreadConfig, backtest_strategy, strategy_metrics
from spx_0dte_planner.data import DailyBar
from spx_0dte_planner.data import load_daily_bars
from spx_0dte_planner.events import load_event_calendar
from spx_0dte_planner.features import align_and_build
from spx_0dte_planner.live import LiveExcursionPrediction, RegimeContext
from spx_0dte_planner.intraday_bridge import translate_proxy_intraday_to_target
from spx_0dte_planner.model import (
    conditional_close_given_touch_by_regime,
    conditional_close_targets_given_touch_by_regime,
    conditional_close_given_touch,
    directional_metrics,
    excursion_probability_backtest,
    excursion_threshold_metrics,
    fit_direction_classifier,
    fit_train_backtest_range_model,
    fit_train_backtest_model,
    fit_two_stage_classifier,
    predict_backtest,
    predict_backtest_classifier,
    predict_backtest_two_stage,
    predict_backtest_range,
    range_metrics,
    regression_metrics,
)
from spx_0dte_planner.opportunity_screen import score_vertical_opportunities, score_vertical_opportunities_after_touch


def test_end_to_end_pipeline():
    spx = load_daily_bars("data/spx_daily.csv")
    vix = load_daily_bars("data/vix_daily.csv")
    events = load_event_calendar("data/events.csv")
    rows = align_and_build(spx, vix, events, max_lag=3)
    fit_result = fit_train_backtest_model(rows, ridge_lambda=1.0, train_ratio=0.6, pca_variance_ratio=0.95)
    classifier_fit = fit_direction_classifier(rows, train_ratio=0.6, pca_variance_ratio=0.95)
    range_fit = fit_train_backtest_range_model(rows, ridge_lambda=1.0, train_ratio=0.6, pca_variance_ratio=0.95)
    two_stage_fit = fit_two_stage_classifier(rows, move_threshold=0.005, train_ratio=0.6, pca_variance_ratio=0.95)
    preds = predict_backtest(fit_result)
    classifier_preds = predict_backtest_classifier(classifier_fit, sigma_return=fit_result.sigma_return)
    range_preds = predict_backtest_range(range_fit)
    two_stage_preds = predict_backtest_two_stage(two_stage_fit, sigma_return=fit_result.sigma_return)
    metrics = regression_metrics(preds)
    range_stats = range_metrics(range_preds)
    excursion_stats = excursion_threshold_metrics(range_preds, thresholds=[0.0025, 0.005])
    excursion_probability_stats = excursion_probability_backtest(range_preds, thresholds=[0.0025, 0.005])
    conditional_close_stats = conditional_close_given_touch(range_preds, thresholds=[0.0025, 0.005])
    regime_close_stats = conditional_close_given_touch_by_regime(
        range_fit.train_rows,
        range_fit.backtest_rows,
        range_preds,
        thresholds=[0.0025, 0.005],
    )
    touch_target_stats = conditional_close_targets_given_touch_by_regime(
        range_fit.train_rows,
        range_fit.backtest_rows,
        range_preds,
        touch_thresholds=[0.0025, 0.005],
        close_thresholds=[0.0025, 0.005],
    )
    direction = directional_metrics(classifier_preds, confidence_threshold=0.55)
    config = DebitSpreadConfig(width=5.0, premium=3.0, confidence_threshold=0.55)
    trades = backtest_strategy(preds, config)
    strategy = strategy_metrics(trades, total_backtest_days=len(preds))

    assert rows
    assert preds
    assert classifier_preds
    assert range_preds
    assert two_stage_preds
    assert metrics["samples"] > 0
    assert range_stats["samples"] > 0
    assert len(excursion_stats["upside"]) == 2
    assert len(excursion_stats["downside"]) == 2
    assert len(excursion_probability_stats["upside"]) == 2
    assert len(excursion_probability_stats["downside"]) == 2
    assert excursion_probability_stats["upside"][0]["samples"] > 0
    assert len(conditional_close_stats["upside"]) == 2
    assert len(conditional_close_stats["downside"]) == 2
    assert conditional_close_stats["upside"][0]["touch_rate"] >= 0.0
    assert "close_return_q25" in conditional_close_stats["upside"][0]
    assert "close_return_q75" in conditional_close_stats["downside"][0]
    assert "vix_regime" in regime_close_stats["upside"]
    assert "combo_regime" in regime_close_stats["upside"]
    assert len(regime_close_stats["upside"]["vix_regime"]) == 6
    assert any(row["regime_family"] == "combo_regime" for row in regime_close_stats["upside"]["combo_regime"])
    assert "upside" in touch_target_stats["upside_touch"]
    assert "combo_regime" in touch_target_stats["upside_touch"]["upside"]
    assert any(row["close_threshold_return"] == 0.0025 for row in touch_target_stats["downside_touch"]["downside"]["gap_regime"])
    assert direction["samples"] > 0
    assert strategy["backtest_days"] == len(preds)
    assert fit_result.reduced_feature_count <= fit_result.input_feature_count
    assert range_fit.high_reduced_feature_count <= range_fit.input_feature_count
    assert range_fit.low_reduced_feature_count <= range_fit.input_feature_count
    if trades:
        assert trades[0].direction in {"bull_call_debit", "bear_put_debit"}


def test_vertical_opportunity_screen_scores_candidates():
    prediction = LiveExcursionPrediction(
        prediction_date=load_daily_bars("data/spx_daily.csv")[-1].date,
        open_price=5000.0,
        prev_close=4988.0,
        predicted_high_return=0.012,
        predicted_low_return=-0.009,
        predicted_high_price=5060.0,
        predicted_low_price=4955.0,
        sigma_high_return=0.004,
        sigma_low_return=0.0045,
        regime_context=RegimeContext(
            weekday="Monday",
            current_vix_open=22.0,
            prev_range_pct=0.012,
            overnight_gap=0.001,
            vix_regime="mid_vix",
            range_regime="mid_prev_range",
            gap_regime="small_gap",
        ),
    )
    continuation_lookup = {
        "upside": {
            "overall": {"overall": {0.005: {"samples": 50.0}}},
            "combo_regime": {
                "mid_vix|mid_prev_range|small_gap": {
                    0.005: {
                        "samples": 12.0,
                        "close_above_touch_rate": 0.62,
                        "basis_label": "combo: mid_vix|mid_prev_range|small_gap",
                    }
                }
            },
            "vix_regime": {"mid_vix": {0.005: {"samples": 20.0, "close_above_touch_rate": 0.60}}},
            "range_regime": {"mid_prev_range": {0.005: {"samples": 18.0, "close_above_touch_rate": 0.58}}},
            "gap_regime": {"small_gap": {0.005: {"samples": 25.0, "close_above_touch_rate": 0.57}}},
        },
        "downside": {
            "overall": {"overall": {0.005: {"samples": 50.0}}},
            "combo_regime": {
                "mid_vix|mid_prev_range|small_gap": {
                    0.005: {
                        "samples": 12.0,
                        "close_below_touch_rate": 0.56,
                        "basis_label": "combo: mid_vix|mid_prev_range|small_gap",
                    }
                }
            },
            "vix_regime": {"mid_vix": {0.005: {"samples": 20.0, "close_below_touch_rate": 0.54}}},
            "range_regime": {"mid_prev_range": {0.005: {"samples": 18.0, "close_below_touch_rate": 0.53}}},
            "gap_regime": {"small_gap": {0.005: {"samples": 25.0, "close_below_touch_rate": 0.52}}},
        },
    }
    vertical_estimates = pd.DataFrame(
        [
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-06 10:00:00",
                "strategy": "bull_call_debit",
                "underlying_price": 5002.0,
                "width_points": 5.0,
                "lower_strike": 5025.0,
                "upper_strike": 5030.0,
                "estimated_net_price": 1.60,
                "short_distance_points": 28.0,
                "short_distance_pct_spot": 28.0 / 5002.0,
            },
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-06 10:00:00",
                "strategy": "bull_put_credit",
                "underlying_price": 5002.0,
                "width_points": 5.0,
                "lower_strike": 4970.0,
                "upper_strike": 4975.0,
                "estimated_net_price": -1.25,
                "short_distance_points": 27.0,
                "short_distance_pct_spot": 27.0 / 5002.0,
            },
        ]
    )

    scored = score_vertical_opportunities(
        prediction=prediction,
        continuation_lookup=continuation_lookup,
        vertical_estimates=vertical_estimates,
    )

    assert not scored.empty
    assert {
        "expected_value",
        "ev_to_risk",
        "near_touch_probability",
        "far_close_beyond_probability",
        "predicted_terminal_value_proxy",
        "predicted_profit_proxy",
        "profit_to_cost_ratio_proxy",
        "cost_to_profit_ratio_proxy",
    }.issubset(scored.columns)
    assert set(scored["strategy"]) == {"bull_call_debit", "bull_put_credit"}


def test_vertical_opportunity_screen_scores_candidates_after_touch():
    prediction = LiveExcursionPrediction(
        prediction_date=load_daily_bars("data/spx_daily.csv")[-1].date,
        open_price=5000.0,
        prev_close=4988.0,
        predicted_high_return=0.012,
        predicted_low_return=-0.009,
        predicted_high_price=5060.0,
        predicted_low_price=4955.0,
        sigma_high_return=0.004,
        sigma_low_return=0.0045,
        regime_context=RegimeContext(
            weekday="Monday",
            current_vix_open=22.0,
            prev_range_pct=0.012,
            overnight_gap=0.001,
            vix_regime="mid_vix",
            range_regime="mid_prev_range",
            gap_regime="small_gap",
        ),
    )
    touch_target_lookup = {
        "upside_touch": {
            "upside": {
                "combo_regime": {
                    "mid_vix|mid_prev_range|small_gap": {
                        0.005: {
                            0.005: {"samples": 14.0, "close_rate": 0.64, "avg_close_return": 0.008},
                            0.006: {"samples": 14.0, "close_rate": 0.48, "avg_close_return": 0.008},
                        }
                    }
                },
                "vix_regime": {"mid_vix": {0.005: {0.005: {"samples": 20.0, "close_rate": 0.62, "avg_close_return": 0.007}}}},
                "range_regime": {"mid_prev_range": {0.005: {0.005: {"samples": 18.0, "close_rate": 0.61, "avg_close_return": 0.007}}}},
                "gap_regime": {"small_gap": {0.005: {0.005: {"samples": 25.0, "close_rate": 0.60, "avg_close_return": 0.007}}}},
            },
            "downside": {
                "combo_regime": {
                    "mid_vix|mid_prev_range|small_gap": {
                        0.005: {
                            0.005: {"samples": 14.0, "close_rate": 0.22, "avg_close_return": 0.008}
                        }
                    }
                },
                "vix_regime": {"mid_vix": {0.005: {0.005: {"samples": 20.0, "close_rate": 0.24, "avg_close_return": 0.007}}}},
                "range_regime": {"mid_prev_range": {0.005: {0.005: {"samples": 18.0, "close_rate": 0.25, "avg_close_return": 0.007}}}},
                "gap_regime": {"small_gap": {0.005: {0.005: {"samples": 25.0, "close_rate": 0.23, "avg_close_return": 0.007}}}},
            },
        },
        "downside_touch": {"upside": {}, "downside": {}},
    }
    vertical_estimates = pd.DataFrame(
        [
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-06 11:15:00",
                "strategy": "bull_call_debit",
                "underlying_price": 5030.0,
                "width_points": 5.0,
                "lower_strike": 5025.0,
                "upper_strike": 5030.0,
                "estimated_net_price": 1.80,
                "short_distance_points": 0.0,
                "short_distance_pct_spot": 0.0,
            },
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-06 11:15:00",
                "strategy": "bear_put_debit",
                "underlying_price": 5030.0,
                "width_points": 5.0,
                "lower_strike": 4970.0,
                "upper_strike": 4975.0,
                "estimated_net_price": 1.10,
                "short_distance_points": 55.0,
                "short_distance_pct_spot": 55.0 / 5030.0,
            },
        ]
    )

    scored = score_vertical_opportunities_after_touch(
        prediction=prediction,
        touch_target_lookup=touch_target_lookup,
        vertical_estimates=vertical_estimates,
        touched_side="upside_touch",
        touch_threshold_return=0.005,
    )

    assert not scored.empty
    assert {
        "conditioning_touch_side",
        "near_close_probability_given_touch",
        "far_close_probability_given_touch",
        "predicted_terminal_value_proxy",
        "predicted_profit_proxy",
        "profit_to_cost_ratio_proxy",
        "cost_to_profit_ratio_proxy",
    }.issubset(scored.columns)
    assert set(scored["strategy"]) == {"bull_call_debit", "bear_put_debit"}


def test_translate_spy_intraday_to_spx_preserves_session_returns():
    proxy_intraday = pd.DataFrame(
        [
            {"timestamp": "2026-04-01 09:30:00", "open": 500.0, "high": 500.2, "low": 499.8, "close": 500.0},
            {"timestamp": "2026-04-01 10:00:00", "open": 500.0, "high": 503.0, "low": 499.9, "close": 502.5},
        ]
    )
    target_daily = [
        DailyBar(
            date=pd.Timestamp("2026-04-01").date(),
            open=5600.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0.0,
        )
    ]
    translated = translate_proxy_intraday_to_target(proxy_intraday, target_daily_bars=target_daily, target_symbol="SPX")

    first_open = float(translated.iloc[0]["open"])
    second_close = float(translated.iloc[1]["close"])
    assert abs(first_open - 5600.0) < 1e-9
    assert abs((second_close / first_open) - (502.5 / 500.0)) < 1e-9
