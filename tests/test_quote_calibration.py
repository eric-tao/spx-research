from __future__ import annotations

from pathlib import Path

import pandas as pd

from intraday_condor_research.quote_calibration import (
    build_quote_pairs,
    enrich_quote_snapshots,
    fit_linear_quote_model,
    load_tradingview_quote_snapshots,
)
from intraday_condor_research.chain_snapshot import (
    extract_single_option_quotes_from_chain_snapshot,
    extract_vix_snapshots_from_chain_snapshot,
    load_tradingview_chain_snapshot,
)
from intraday_condor_research.strategy_costs import (
    estimate_strategy_costs_from_trades,
    load_option_trade_prints,
    load_timed_strategy_legs,
)
from intraday_condor_research.fill_sampling import (
    enrich_fill_samples,
    fit_fill_model,
    load_fill_samples,
    summarize_fill_samples,
)
from intraday_condor_research.vertical_fills import estimate_vertical_fills_from_quotes


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


def test_tradingview_chain_snapshot_extraction(tmp_path: Path) -> None:
    chain_rows = pd.DataFrame(
        [
            {
                "snapshot_id": "spx_2026_04_03_1000",
                "timestamp": "2026-04-03 10:00:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "vix_price": 21.85,
                "expiry_date": "2026-04-03",
                "strike": 5220,
                "call_bid": 31.30,
                "call_ask": 32.10,
                "put_bid": 6.70,
                "put_ask": 7.40,
                "put_last_price": 7.10,
            },
            {
                "snapshot_id": "spx_2026_04_03_1000",
                "timestamp": "2026-04-03 10:00:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "vix_price": 21.85,
                "expiry_date": "2026-04-03",
                "strike": 5275,
                "call_bid": 7.10,
                "call_ask": 7.90,
                "call_last_price": 7.40,
                "put_bid": 11.40,
                "put_ask": 12.20,
            },
        ]
    )
    chain_path = tmp_path / "chain_snapshot.csv"
    chain_rows.to_csv(chain_path, index=False)

    loaded = load_tradingview_chain_snapshot(chain_path)
    quotes = extract_single_option_quotes_from_chain_snapshot(loaded)
    vix = extract_vix_snapshots_from_chain_snapshot(loaded)

    assert len(loaded) == 2
    assert len(quotes) == 4
    assert set(quotes["option_type"]) == {"call", "put"}
    assert set(quotes["strike"]) == {5220.0, 5275.0}
    assert len(vix) == 1
    assert float(vix.iloc[0]["vix_price"]) == 21.85


def test_trade_print_strategy_cost_matching(tmp_path: Path) -> None:
    trade_rows = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-03 10:00:12",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5220,
                "trade_price": 7.05,
            },
            {
                "timestamp": "2026-04-03 10:00:18",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.60,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5210,
                "trade_price": 4.55,
            },
            {
                "timestamp": "2026-04-03 10:00:14",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.40,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5275,
                "trade_price": 7.45,
            },
            {
                "timestamp": "2026-04-03 10:00:20",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.70,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5285,
                "trade_price": 4.85,
            },
        ]
    )
    legs_rows = pd.DataFrame(
        [
            {
                "strategy_id": "ic1",
                "entry_timestamp": "2026-04-03 10:00:15",
                "underlying_symbol": "SPX",
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5220,
                "action": "sell",
                "quantity": 1,
                "max_time_diff_minutes": 5,
            },
            {
                "strategy_id": "ic1",
                "entry_timestamp": "2026-04-03 10:00:15",
                "underlying_symbol": "SPX",
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5210,
                "action": "buy",
                "quantity": 1,
                "max_time_diff_minutes": 5,
            },
            {
                "strategy_id": "ic1",
                "entry_timestamp": "2026-04-03 10:00:15",
                "underlying_symbol": "SPX",
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5275,
                "action": "sell",
                "quantity": 1,
                "max_time_diff_minutes": 5,
            },
            {
                "strategy_id": "ic1",
                "entry_timestamp": "2026-04-03 10:00:15",
                "underlying_symbol": "SPX",
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5285,
                "action": "buy",
                "quantity": 1,
                "max_time_diff_minutes": 5,
            },
        ]
    )
    trades_path = tmp_path / "trade_prints.csv"
    legs_path = tmp_path / "timed_legs.csv"
    trade_rows.to_csv(trades_path, index=False)
    legs_rows.to_csv(legs_path, index=False)

    loaded_trades = load_option_trade_prints(trades_path)
    loaded_legs = load_timed_strategy_legs(legs_path)
    matched, summaries = estimate_strategy_costs_from_trades(loaded_trades, loaded_legs)

    assert len(matched) == 4
    assert len(summaries) == 1
    assert summaries.iloc[0]["strategy_id"] == "ic1"
    assert summaries.iloc[0]["leg_count"] == 4
    assert abs(float(summaries.iloc[0]["net_trade_price"]) - (-5.10)) < 1e-9
    assert summaries.iloc[0]["debit_or_credit"] == "credit"


def test_fill_sampling_pipeline(tmp_path: Path) -> None:
    sample_rows = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-03 09:42:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5246.80,
                "vix_price": 21.40,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5230,
                "side": "buy",
                "bid": 9.80,
                "ask": 10.50,
                "actual_fill_price": 10.20,
            },
            {
                "timestamp": "2026-04-03 10:15:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5252.10,
                "vix_price": 21.85,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5275,
                "side": "sell",
                "bid": 7.10,
                "ask": 7.90,
                "actual_fill_price": 7.35,
            },
            {
                "timestamp": "2026-04-03 12:05:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5249.55,
                "vix_price": 22.60,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5225,
                "side": "sell",
                "bid": 5.40,
                "ask": 5.95,
                "actual_fill_price": 5.62,
            },
            {
                "timestamp": "2026-04-03 14:45:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5238.20,
                "vix_price": 24.10,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5260,
                "side": "buy",
                "bid": 3.60,
                "ask": 4.10,
                "actual_fill_price": 3.92,
            },
        ]
    )
    path = tmp_path / "fill_samples.csv"
    sample_rows.to_csv(path, index=False)

    loaded = load_fill_samples(path)
    enriched = enrich_fill_samples(loaded)
    summary = summarize_fill_samples(loaded)
    model = fit_fill_model(loaded)

    assert len(loaded) == 4
    assert len(enriched) == 4
    assert not summary.empty
    assert {"adverse_fill_from_mid", "vix_bucket", "time_bucket", "moneyness_bucket"}.issubset(enriched.columns)
    assert model.target_name == "adverse_fill_from_mid"
    assert len(model.feature_names) == len(model.coefficients)


def test_vertical_fill_estimation_from_single_leg_samples() -> None:
    fill_samples = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-03 09:42:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5246.80,
                "vix_price": 21.40,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5230,
                "side": "buy",
                "bid": 9.80,
                "ask": 10.50,
                "actual_fill_price": 10.20,
            },
            {
                "timestamp": "2026-04-03 10:15:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5252.10,
                "vix_price": 21.85,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5275,
                "side": "sell",
                "bid": 7.10,
                "ask": 7.90,
                "actual_fill_price": 7.35,
            },
            {
                "timestamp": "2026-04-03 12:05:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5249.55,
                "vix_price": 22.60,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5225,
                "side": "sell",
                "bid": 5.40,
                "ask": 5.95,
                "actual_fill_price": 5.62,
            },
            {
                "timestamp": "2026-04-03 14:45:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5238.20,
                "vix_price": 24.10,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5260,
                "side": "buy",
                "bid": 3.60,
                "ask": 4.10,
                "actual_fill_price": 3.92,
            },
        ]
    )
    quote_snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-03 10:00:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5210,
                "bid": 4.20,
                "ask": 4.80,
            },
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-03 10:00:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "expiry_date": "2026-04-03",
                "option_type": "put",
                "strike": 5215,
                "bid": 5.05,
                "ask": 5.65,
            },
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-03 10:00:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5275,
                "bid": 7.10,
                "ask": 7.90,
            },
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-03 10:00:00",
                "underlying_symbol": "SPX",
                "underlying_price": 5248.35,
                "expiry_date": "2026-04-03",
                "option_type": "call",
                "strike": 5280,
                "bid": 5.60,
                "ask": 6.30,
            },
        ]
    )
    vix_snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "snap1",
                "timestamp": "2026-04-03 10:00:00",
                "vix_price": 21.85,
            }
        ]
    )

    estimated = estimate_vertical_fills_from_quotes(
        fill_samples=fill_samples,
        quote_snapshots=quote_snapshots,
        vix_by_snapshot=vix_snapshots,
        width_points=5.0,
    )

    assert len(estimated) == 4
    assert set(estimated["strategy"]) == {
        "bull_call_debit",
        "bear_call_credit",
        "bear_put_debit",
        "bull_put_credit",
    }
    assert {"estimated_net_price", "mid_net_price", "natural_net_price", "short_distance_points"}.issubset(estimated.columns)
