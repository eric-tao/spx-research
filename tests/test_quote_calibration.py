from __future__ import annotations

from pathlib import Path

import pandas as pd

from intraday_condor_research.quote_calibration import (
    build_quote_pairs,
    enrich_quote_snapshots,
    fit_linear_quote_model,
    load_tradingview_quote_snapshots,
)


def _make_quote_rows() -> pd.DataFrame:
    rows = [
        {
            "trade_id": "t1",
            "timestamp": "2026-03-25 12:00:00",
            "underlying_price": 500.0,
            "expiry_date": "2026-03-25",
            "short_put_strike": 495,
            "long_put_strike": 490,
            "short_call_strike": 505,
            "long_call_strike": 510,
            "short_put_bid": 1.95,
            "short_put_ask": 2.05,
            "long_put_bid": 0.45,
            "long_put_ask": 0.55,
            "short_call_bid": 1.95,
            "short_call_ask": 2.05,
            "long_call_bid": 0.45,
            "long_call_ask": 0.55,
        },
        {
            "trade_id": "t1",
            "timestamp": "2026-03-25 12:15:00",
            "underlying_price": 501.0,
            "expiry_date": "2026-03-25",
            "short_put_strike": 495,
            "long_put_strike": 490,
            "short_call_strike": 505,
            "long_call_strike": 510,
            "short_put_bid": 1.75,
            "short_put_ask": 1.85,
            "long_put_bid": 0.40,
            "long_put_ask": 0.50,
            "short_call_bid": 2.10,
            "short_call_ask": 2.20,
            "long_call_bid": 0.55,
            "long_call_ask": 0.65,
        },
        {
            "trade_id": "t2",
            "timestamp": "2026-03-25 12:00:00",
            "underlying_price": 500.0,
            "expiry_date": "2026-03-25",
            "short_put_strike": 490,
            "long_put_strike": 485,
            "short_call_strike": 510,
            "long_call_strike": 515,
            "short_put_bid": 1.15,
            "short_put_ask": 1.25,
            "long_put_bid": 0.25,
            "long_put_ask": 0.35,
            "short_call_bid": 1.10,
            "short_call_ask": 1.20,
            "long_call_bid": 0.20,
            "long_call_ask": 0.30,
        },
        {
            "trade_id": "t2",
            "timestamp": "2026-03-25 12:30:00",
            "underlying_price": 498.5,
            "expiry_date": "2026-03-25",
            "short_put_strike": 490,
            "long_put_strike": 485,
            "short_call_strike": 510,
            "long_call_strike": 515,
            "short_put_bid": 1.35,
            "short_put_ask": 1.45,
            "long_put_bid": 0.30,
            "long_put_ask": 0.40,
            "short_call_bid": 0.90,
            "short_call_ask": 1.00,
            "long_call_bid": 0.18,
            "long_call_ask": 0.28,
        },
    ]
    return pd.DataFrame(rows)


def test_quote_calibration_pipeline(tmp_path: Path) -> None:
    raw = _make_quote_rows()
    path = tmp_path / "quotes.csv"
    raw.to_csv(path, index=False)

    loaded = load_tradingview_quote_snapshots(path)
    enriched = enrich_quote_snapshots(loaded)
    pairs = build_quote_pairs(enriched)
    model = fit_linear_quote_model(pairs)

    assert len(enriched) == 4
    assert len(pairs) == 2
    assert {"condor_bid_credit", "condor_mid_credit", "short_distance_widths"}.issubset(enriched.columns)
    assert {"delta_underlying_points", "delta_condor_mid_credit", "minutes_elapsed"}.issubset(pairs.columns)
    assert model.target_name == "delta_condor_mid_credit"
    assert len(model.feature_names) == len(model.coefficients)
