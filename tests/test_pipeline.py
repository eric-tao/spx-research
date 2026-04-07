from spx_0dte_planner.backtest import DebitSpreadConfig, backtest_strategy, strategy_metrics
from spx_0dte_planner.data import load_daily_bars
from spx_0dte_planner.events import load_event_calendar
from spx_0dte_planner.features import align_and_build
from spx_0dte_planner.model import (
    conditional_close_given_touch_by_regime,
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
    assert direction["samples"] > 0
    assert strategy["backtest_days"] == len(preds)
    assert fit_result.reduced_feature_count <= fit_result.input_feature_count
    assert range_fit.high_reduced_feature_count <= range_fit.input_feature_count
    assert range_fit.low_reduced_feature_count <= range_fit.input_feature_count
    if trades:
        assert trades[0].direction in {"bull_call_debit", "bear_put_debit"}
