from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from spx_0dte_planner.live import next_trade_date, required_history_date
from spx_0dte_planner.webapp import _reanchor_vertical_estimates_to_spot, create_app_state, render_app


def test_webapp_renders_with_default_inputs():
    state = create_app_state(
        spx_path="data/spx_daily.csv",
        vix_path="data/vix_daily.csv",
        events_path="data/events.csv",
        underlying_label="SPX",
    )
    html = render_app(
        state,
        {
            "prediction_date": ["2026-04-03"],
            "spx_open": ["5248.35"],
            "vix_open": ["21.40"],
            "current_spx_spot": ["5249.10"],
            "checkpoint_time": ["13:15"],
            "high_so_far": ["5278.00"],
            "low_so_far": ["5236.00"],
            "strong_profit_threshold": ["2.00"],
            "strong_ratio_threshold": ["0.90"],
            "watch_profit_threshold": ["1.00"],
            "watch_ratio_threshold": ["0.40"],
            "event_type": ["FOMC"],
        },
    ).decode("utf-8")

    assert "SPX Excursion Probability App" in html
    assert "Decision Summary" in html
    assert "Upside Touch Levels" in html
    assert "Downside Touch Levels" in html
    assert "Upside Continuation If Touched" in html
    assert "Downside Continuation If Touched" in html
    assert "Vertical strategy to watch" in html
    assert "Value breakpoints" in html
    assert "Observed intraday state" in html
    assert "Current SPX spot" in html
    assert "Current time of day (ET)" in html
    assert "High so far" in html
    assert "Low so far" in html
    assert "Strong value: minimum proxy profit (pts)" in html
    assert "Strong value: minimum profit / risk" in html
    assert "Watch: minimum proxy profit (pts)" in html
    assert "Watch: minimum profit / risk" in html
    assert "confirms that the selected upside touch has already happened" in html
    assert "Touched threshold (%)" in html
    assert "Touch price" in html
    assert "VIX daily session open" in html


def test_webapp_resolves_project_relative_paths_from_elsewhere():
    state = create_app_state(
        spx_path="data/spx_daily.csv",
        vix_path="data/vix_daily.csv",
        events_path="data/events.csv",
        underlying_label="SPX",
        auto_refresh=False,
    )
    assert state.spx_path.endswith("/data/spx_daily.csv")
    assert state.vix_path.endswith("/data/vix_daily.csv")


def test_webapp_prefers_latest_extracted_chain_snapshot_when_available():
    state = create_app_state(
        spx_path="data/spx_daily.csv",
        vix_path="data/vix_daily.csv",
        events_path="data/events.csv",
        underlying_label="SPX",
        auto_refresh=False,
    )
    assert state.quotes_path is not None
    assert state.vix_snapshots_path is not None
    assert "artifacts/tradingview_chain_2026_04_07_1010" in state.quotes_path
    assert "artifacts/tradingview_chain_2026_04_07_1010" in state.vix_snapshots_path


def test_next_trade_date_rolls_after_close_and_weekend():
    tz = ZoneInfo("America/New_York")
    assert next_trade_date(datetime(2026, 4, 6, 10, 0, tzinfo=tz)) == datetime(2026, 4, 6, 10, 0, tzinfo=tz).date()
    assert next_trade_date(datetime(2026, 4, 6, 17, 0, tzinfo=tz)).isoformat() == "2026-04-07"
    assert next_trade_date(datetime(2026, 4, 11, 12, 0, tzinfo=tz)).isoformat() == "2026-04-13"
    assert required_history_date(datetime(2026, 4, 11, 12, 0, tzinfo=tz)).isoformat() == "2026-04-10"


def test_webapp_shows_reanchored_proxy_note_when_spot_is_far_from_sample_chain():
    state = create_app_state(
        spx_path="data/spx_daily.csv",
        vix_path="data/vix_daily.csv",
        events_path="data/events.csv",
        underlying_label="SPX",
        auto_refresh=False,
    )
    html = render_app(
        state,
        {
            "prediction_date": ["2026-04-03"],
            "spx_open": ["7000"],
            "vix_open": ["25.33"],
            "current_spx_spot": ["7024.5"],
            "checkpoint_time": ["13:06"],
            "high_so_far": ["7040.0"],
            "low_so_far": ["6988.0"],
            "touched_side": ["upside_touch"],
            "touched_threshold": ["0.50"],
            "vertical_width_points": ["10"],
        },
    ).decode("utf-8")

    assert "re-anchored stale-chain proxy" in html
    assert "Strike band" in html


def test_reanchored_verticals_preserve_requested_width():
    verticals = pd.DataFrame(
        [
            {
                "snapshot_id": "sample",
                "strategy": "bull_call_debit",
                "underlying_price": 5248.35,
                "lower_strike": 5275.0,
                "upper_strike": 5285.0,
                "width_points": 10.0,
                "estimated_net_price": 3.0,
            }
        ]
    )

    reanchored = _reanchor_vertical_estimates_to_spot(verticals, current_spot=6638.0)

    assert len(reanchored) == 1
    row = reanchored.iloc[0]
    assert abs(float(row["upper_strike"]) - float(row["lower_strike"])) == 10.0
    assert float(row["lower_strike"]) % 5.0 == 0.0
    assert float(row["upper_strike"]) % 5.0 == 0.0
