from __future__ import annotations

from pathlib import Path

import pandas as pd

from intraday_condor_research.features import build_research_frames
from intraday_condor_research.plots import save_viability_heatmaps
from intraday_condor_research.regimes import attach_regime_columns
from intraday_condor_research.session import filter_rth, keep_complete_sessions
from intraday_condor_research.stats import (
    breakeven_credit_tables,
    decision_credit_tables,
    expected_value_checkpoint_summary_tables,
    expected_value_regime_tables,
    expected_value_tables,
    expected_value_vol_summary_tables,
    probability_tables,
    regime_tables,
)


def _make_intraday_frame() -> pd.DataFrame:
    rows = []
    session_specs = [
        ("2026-01-05", [500.0, 501.0, 503.0, 502.0, 504.5, 505.0, 506.0]),
        ("2026-01-06", [505.0, 503.0, 501.0, 500.5, 499.0, 498.0, 497.5]),
    ]
    checkpoints = ["09:30", "10:00", "10:30", "12:00", "14:00", "15:00", "15:30"]
    for session_date, closes in session_specs:
        for idx, checkpoint in enumerate(checkpoints):
            timestamp = pd.Timestamp(f"{session_date} {checkpoint}")
            close = closes[idx]
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + idx,
                }
            )
        rows.append(
            {
                "timestamp": pd.Timestamp(f"{session_date} 16:00"),
                "open": closes[-1],
                "high": closes[-1] + 0.2,
                "low": closes[-1] - 0.3,
                "close": closes[-1] + 0.1,
                "volume": 2000,
            }
        )
    return pd.DataFrame(rows)


def _make_vix_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "open": 18.0, "high": 19.0, "low": 17.5, "close": 18.5, "volume": 0},
            {"date": "2026-01-05", "open": 20.0, "high": 21.0, "low": 19.5, "close": 20.5, "volume": 0},
            {"date": "2026-01-06", "open": 22.0, "high": 23.0, "low": 21.5, "close": 22.5, "volume": 0},
        ]
    )


def test_intraday_research_pipeline(tmp_path: Path) -> None:
    raw = _make_intraday_frame()
    vix_daily = _make_vix_daily_frame()
    rth = filter_rth(raw)
    complete = keep_complete_sessions(rth, checkpoints=["10:00", "10:30", "12:00", "14:00", "15:00", "15:30"])
    research = build_research_frames(
        complete,
        checkpoints=["10:00", "10:30", "12:00", "14:00", "15:00", "15:30"],
        widths=[25, 50],
        symbol="QQQ",
        vix_daily=vix_daily,
        vxn_multiplier=1.2,
    )
    per_checkpoint = attach_regime_columns(research.per_checkpoint)
    prob = probability_tables(per_checkpoint, widths=[25, 50])
    breakeven = breakeven_credit_tables(per_checkpoint, widths=[25, 50], short_distance_multiples=[0.5, 1.0])
    decision = decision_credit_tables(breakeven, ev_to_max_loss_ratio=0.1)
    ev = expected_value_tables(per_checkpoint, widths=[25, 50], short_distance_multiples=[0.5, 1.0], credit_ratios=[0.1, 0.2])
    ev_checkpoint = expected_value_checkpoint_summary_tables(
        per_checkpoint,
        widths=[25, 50],
        short_distance_multiples=[0.5, 1.0],
        credit_ratios=[0.1, 0.2],
    )
    ev_regime = expected_value_regime_tables(
        per_checkpoint,
        widths=[25, 50],
        short_distance_multiples=[0.5, 1.0],
        credit_ratios=[0.1, 0.2],
    )
    ev_vol = expected_value_vol_summary_tables(
        per_checkpoint,
        widths=[25, 50],
        short_distance_multiples=[0.5, 1.0],
        credit_ratios=[0.1, 0.2],
    )
    regime = regime_tables(per_checkpoint, widths=[25, 50])
    viability_paths = save_viability_heatmaps(
        breakeven,
        tmp_path / "plots",
        credit_ratio_thresholds=[0.1, 0.2],
        width_display_multiplier=1.0,
    )

    assert len(research.per_day) == 2
    assert len(research.per_checkpoint) == 12
    assert "move_bucket" in per_checkpoint.columns
    assert "25" in prob
    assert "25" in breakeven
    assert "25" in decision
    assert "25" in ev
    assert "25" in ev_checkpoint
    assert "25" in ev_regime
    assert "25" in ev_vol
    assert "50" in regime
    assert prob["25"]["observations"].sum() == len(per_checkpoint)
    assert {"vol_regime", "breakeven_credit_ratio", "breakeven_credit_points"}.issubset(breakeven["25"].columns)
    assert {"required_credit_ratio", "required_credit_points", "decision_ev_to_max_loss_ratio"}.issubset(decision["25"].columns)
    assert {"expected_value_widths", "expected_value_points", "credit_ratio", "short_distance_multiple"}.issubset(ev["25"].columns)
    assert {"checkpoint", "expected_value_points", "win_rate"}.issubset(ev_checkpoint["25"].columns)
    assert {"vol_regime", "move_bucket", "max_loss_rate"}.issubset(ev_regime["25"].columns)
    assert {"vol_regime", "expected_value_widths", "full_win_rate"}.issubset(ev_vol["25"].columns)
    assert {"forward_vol_proxy_pct", "forward_vol_from_vix_pct", "forward_vol_from_est_vxn_pct", "projected_intraday_realized_vol_pct", "vix_reference"}.issubset(per_checkpoint.columns)
    assert (per_checkpoint["forward_vol_proxy_pct"] > 0).all()
    assert viability_paths
