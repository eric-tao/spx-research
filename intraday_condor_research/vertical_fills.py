from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fill_sampling import FillModel, enrich_fill_samples, fit_fill_model, load_fill_samples
from .strategy_costs import enrich_option_quote_snapshots, load_option_quote_snapshots


def _predict_adverse_fill(
    fill_model: FillModel,
    *,
    underlying_price: float,
    vix_price: float,
    timestamp: pd.Timestamp,
    option_type: str,
    side: str,
    strike: float,
    bid: float,
    ask: float,
) -> tuple[float, float]:
    temp = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "underlying_symbol": "SPX",
                "underlying_price": underlying_price,
                "vix_price": vix_price,
                "expiry_date": str(timestamp.date()),
                "option_type": option_type,
                "strike": strike,
                "side": side,
                "bid": bid,
                "ask": ask,
                "actual_fill_price": 0.5 * (bid + ask),
            }
        ]
    )
    enriched = enrich_fill_samples(temp)
    adverse = float(fill_model.predict(enriched)[0])
    spread = float(enriched.iloc[0]["spread"])
    adverse = max(0.0, min(adverse, spread if spread > 0 else 0.0))
    mid = float(enriched.iloc[0]["mid"])
    if side == "buy":
        fill_price = min(max(mid + adverse, bid), ask)
    else:
        fill_price = max(min(mid - adverse, ask), bid)
    return fill_price, adverse


def estimate_vertical_fills_from_quotes(
    *,
    fill_samples: pd.DataFrame,
    quote_snapshots: pd.DataFrame,
    vix_by_snapshot: pd.DataFrame,
    width_points: float = 5.0,
) -> pd.DataFrame:
    fill_samples = fill_samples.copy()
    if not pd.api.types.is_datetime64_any_dtype(fill_samples["timestamp"]):
        fill_samples["timestamp"] = pd.to_datetime(fill_samples["timestamp"])
    quote_snapshots = quote_snapshots.copy()
    if not pd.api.types.is_datetime64_any_dtype(quote_snapshots["timestamp"]):
        quote_snapshots["timestamp"] = pd.to_datetime(quote_snapshots["timestamp"])
    vix_by_snapshot = vix_by_snapshot.copy()
    if not pd.api.types.is_datetime64_any_dtype(vix_by_snapshot["timestamp"]):
        vix_by_snapshot["timestamp"] = pd.to_datetime(vix_by_snapshot["timestamp"])

    fill_model = fit_fill_model(fill_samples, target_column="adverse_fill_from_mid")
    quotes = enrich_option_quote_snapshots(quote_snapshots).copy()
    quotes = quotes.merge(vix_by_snapshot[["snapshot_id", "timestamp", "vix_price"]], on=["snapshot_id", "timestamp"], how="left")
    if quotes["vix_price"].isna().any():
        missing = quotes.loc[quotes["vix_price"].isna(), ["snapshot_id", "timestamp"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"Missing vix_price for quote snapshots: {missing}")

    rows: list[dict[str, float | str]] = []
    for (snapshot_id, option_type), group in quotes.groupby(["snapshot_id", "option_type"], sort=True):
        ordered = group.sort_values("strike").reset_index(drop=True)
        quote_map = {float(row["strike"]): row for _, row in ordered.iterrows()}
        for strike in sorted(quote_map):
            pair_strike = strike + width_points
            if pair_strike not in quote_map:
                continue
            low = quote_map[strike]
            high = quote_map[pair_strike]
            timestamp = pd.Timestamp(low["timestamp"])
            vix_price = float(low["vix_price"])
            underlying_price = float(low["underlying_price"])

            if option_type == "call":
                strategies = [
                    ("bull_call_debit", [("buy", low), ("sell", high)]),
                    ("bear_call_credit", [("sell", low), ("buy", high)]),
                ]
                short_distance_points = float(high["strike"] - underlying_price)
            else:
                strategies = [
                    ("bear_put_debit", [("buy", high), ("sell", low)]),
                    ("bull_put_credit", [("sell", high), ("buy", low)]),
                ]
                short_distance_points = float(underlying_price - high["strike"])

            for strategy_name, legs in strategies:
                leg_rows = []
                estimated_net = 0.0
                natural_net = 0.0
                mid_net = 0.0
                total_leg_spread = 0.0
                for side, leg in legs:
                    estimated_fill, adverse = _predict_adverse_fill(
                        fill_model,
                        underlying_price=underlying_price,
                        vix_price=vix_price,
                        timestamp=timestamp,
                        option_type=str(option_type),
                        side=side,
                        strike=float(leg["strike"]),
                        bid=float(leg["bid"]),
                        ask=float(leg["ask"]),
                    )
                    mid = float(leg["mid"])
                    natural = float(leg["ask"] if side == "buy" else leg["bid"])
                    sign = 1.0 if side == "buy" else -1.0
                    estimated_net += sign * estimated_fill
                    natural_net += sign * natural
                    mid_net += sign * mid
                    total_leg_spread += float(leg["spread"])
                    leg_rows.append(
                        {
                            "leg_side": side,
                            "leg_strike": float(leg["strike"]),
                            "leg_bid": float(leg["bid"]),
                            "leg_ask": float(leg["ask"]),
                            "leg_mid": mid,
                            "leg_estimated_fill": estimated_fill,
                            "leg_adverse_fill_from_mid": adverse,
                        }
                    )
                rows.append(
                    {
                        "snapshot_id": str(snapshot_id),
                        "timestamp": timestamp,
                        "option_type": str(option_type),
                        "strategy": strategy_name,
                        "underlying_price": underlying_price,
                        "vix_price": vix_price,
                        "width_points": width_points,
                        "lower_strike": float(low["strike"]),
                        "upper_strike": float(high["strike"]),
                        "short_distance_points": short_distance_points,
                        "short_distance_pct_spot": short_distance_points / underlying_price,
                        "estimated_net_price": estimated_net,
                        "mid_net_price": mid_net,
                        "natural_net_price": natural_net,
                        "estimated_edge_vs_mid": estimated_net - mid_net,
                        "estimated_edge_vs_natural": estimated_net - natural_net,
                        "total_leg_spread": total_leg_spread,
                        "debit_or_credit": "debit" if strategy_name.endswith("debit") else "credit",
                        "legs": leg_rows,
                    }
                )
    return pd.DataFrame(rows)


def load_vertical_fill_inputs(
    fill_samples_path: str | Path,
    quote_snapshots_path: str | Path,
    vix_by_snapshot_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fill_samples = load_fill_samples(fill_samples_path)
    quote_snapshots = load_option_quote_snapshots(quote_snapshots_path)
    vix_by_snapshot = pd.read_csv(Path(vix_by_snapshot_path))
    required = {"snapshot_id", "timestamp", "vix_price"}
    missing = required - set(vix_by_snapshot.columns)
    if missing:
        raise ValueError(f"Missing required VIX snapshot columns: {sorted(missing)}")
    return fill_samples, quote_snapshots, vix_by_snapshot
