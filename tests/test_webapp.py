from datetime import datetime
from zoneinfo import ZoneInfo

from spx_0dte_planner.live import next_trade_date, required_history_date
from spx_0dte_planner.webapp import create_app_state, render_app


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
            "prediction_date": ["2026-03-26"],
            "spx_open": ["6700"],
            "vix_open": ["25.33"],
            "event_type": ["FOMC"],
        },
    ).decode("utf-8")

    assert "SPX Excursion Probability App" in html
    assert "Decision Summary" in html
    assert "Upside Touch Levels" in html
    assert "Downside Touch Levels" in html
    assert "Upside Continuation If Touched" in html
    assert "Downside Continuation If Touched" in html
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


def test_next_trade_date_rolls_after_close_and_weekend():
    tz = ZoneInfo("America/New_York")
    assert next_trade_date(datetime(2026, 4, 6, 10, 0, tzinfo=tz)) == datetime(2026, 4, 6, 10, 0, tzinfo=tz).date()
    assert next_trade_date(datetime(2026, 4, 6, 17, 0, tzinfo=tz)).isoformat() == "2026-04-07"
    assert next_trade_date(datetime(2026, 4, 11, 12, 0, tzinfo=tz)).isoformat() == "2026-04-13"
    assert required_history_date(datetime(2026, 4, 11, 12, 0, tzinfo=tz)).isoformat() == "2026-04-10"
