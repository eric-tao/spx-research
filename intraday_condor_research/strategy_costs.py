from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_OPTION_QUOTE_COLUMNS = {
    "snapshot_id",
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "expiry_date",
    "option_type",
    "strike",
    "bid",
    "ask",
}

REQUIRED_STRATEGY_LEG_COLUMNS = {
    "strategy_id",
    "snapshot_id",
    "expiry_date",
    "option_type",
    "strike",
    "action",
    "quantity",
}

REQUIRED_TRADE_PRINT_COLUMNS = {
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "expiry_date",
    "option_type",
    "strike",
    "trade_price",
}

REQUIRED_TIMED_STRATEGY_LEG_COLUMNS = {
    "strategy_id",
    "entry_timestamp",
    "underlying_symbol",
    "expiry_date",
    "option_type",
    "strike",
    "action",
    "quantity",
}


@dataclass
class StrategyCostSummary:
    strategy_id: str
    snapshot_id: str
    underlying_symbol: str
    underlying_price: float
    expiry_date: str
    net_mid: float
    net_natural: float
    net_halfway: float
    total_spread_cost: float
    widest_leg_spread: float
    leg_count: int
    debit_or_credit: str


def load_option_quote_snapshots(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_OPTION_QUOTE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required option quote columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    for column in ["underlying_price", "strike", "bid", "ask"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "last_price" in result.columns:
        result["last_price"] = pd.to_numeric(result["last_price"], errors="coerce")
    if "actual_fill_price" in result.columns:
        result["actual_fill_price"] = pd.to_numeric(result["actual_fill_price"], errors="coerce")
    result["option_type"] = result["option_type"].str.lower().str.strip()
    return result.sort_values(["snapshot_id", "option_type", "strike"]).reset_index(drop=True)


def enrich_option_quote_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["mid"] = 0.5 * (result["bid"] + result["ask"])
    result["spread"] = result["ask"] - result["bid"]
    result["half_spread"] = 0.5 * result["spread"]
    result["distance_points"] = np.where(
        result["option_type"].eq("call"),
        result["strike"] - result["underlying_price"],
        result["underlying_price"] - result["strike"],
    )
    result["abs_distance_points"] = result["distance_points"].abs()
    result["distance_pct_underlying"] = result["distance_points"] / result["underlying_price"]
    result["spread_pct_mid"] = np.where(result["mid"] > 0, result["spread"] / result["mid"], np.nan)
    result["moneyness_bucket"] = pd.cut(
        result["distance_pct_underlying"],
        bins=[-np.inf, -0.02, -0.01, -0.0025, 0.0025, 0.01, 0.02, np.inf],
        labels=["deep_itm", "itm", "slightly_itm", "atm", "slightly_otm", "otm", "far_otm"],
        include_lowest=True,
    ).astype(str)
    return result


def summarize_option_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_option_quote_snapshots(frame)
    grouping = ["option_type", "moneyness_bucket"]
    summary = (
        enriched.groupby(grouping, dropna=False)
        .agg(
            observations=("mid", "size"),
            avg_mid=("mid", "mean"),
            avg_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            avg_spread_pct_mid=("spread_pct_mid", "mean"),
            avg_abs_distance_points=("abs_distance_points", "mean"),
        )
        .reset_index()
        .sort_values(grouping)
    )
    return summary


def load_strategy_legs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_STRATEGY_LEG_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required strategy leg columns: {sorted(missing)}")
    result = frame.copy()
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    result["strike"] = pd.to_numeric(result["strike"], errors="coerce")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    result["action"] = result["action"].str.lower().str.strip()
    invalid_actions = sorted(set(result["action"]) - {"buy", "sell"})
    if invalid_actions:
        raise ValueError(f"Unsupported leg actions: {invalid_actions}")
    return result.sort_values(["strategy_id", "snapshot_id", "option_type", "strike"]).reset_index(drop=True)


def estimate_strategy_costs(quotes: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    enriched_quotes = enrich_option_quote_snapshots(quotes)
    quote_key = ["snapshot_id", "expiry_date", "option_type", "strike"]
    merge_columns = quote_key + ["timestamp", "underlying_symbol", "underlying_price", "bid", "ask", "mid", "spread"]
    merged = legs.merge(enriched_quotes[merge_columns], on=quote_key, how="left", validate="many_to_one")
    missing_quotes = merged[merged["mid"].isna()]
    if not missing_quotes.empty:
        missing_text = missing_quotes[quote_key].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"Missing quote rows for strategy legs: {missing_text}")

    def _signed_cost(action: str, quantity: float, value: float) -> float:
        direction = 1.0 if action == "buy" else -1.0
        return direction * quantity * value

    merged["mid_cost"] = [
        _signed_cost(action, quantity, mid)
        for action, quantity, mid in zip(merged["action"], merged["quantity"], merged["mid"])
    ]
    merged["natural_cost"] = [
        _signed_cost(action, quantity, ask if action == "buy" else bid)
        for action, quantity, bid, ask in zip(merged["action"], merged["quantity"], merged["bid"], merged["ask"])
    ]
    merged["best_case_cost"] = [
        _signed_cost(action, quantity, bid if action == "buy" else ask)
        for action, quantity, bid, ask in zip(merged["action"], merged["quantity"], merged["bid"], merged["ask"])
    ]
    merged["spread_cost"] = (merged["natural_cost"] - merged["best_case_cost"]).abs()

    rows: list[StrategyCostSummary] = []
    for (strategy_id, snapshot_id), group in merged.groupby(["strategy_id", "snapshot_id"], sort=True):
        first = group.iloc[0]
        net_mid = float(group["mid_cost"].sum())
        net_natural = float(group["natural_cost"].sum())
        net_best_case = float(group["best_case_cost"].sum())
        net_halfway = 0.5 * (net_mid + net_natural)
        debit_or_credit = "debit" if net_mid >= 0 else "credit"
        rows.append(
            StrategyCostSummary(
                strategy_id=strategy_id,
                snapshot_id=snapshot_id,
                underlying_symbol=str(first["underlying_symbol"]),
                underlying_price=float(first["underlying_price"]),
                expiry_date=str(first["expiry_date"]),
                net_mid=net_mid,
                net_natural=net_natural,
                net_halfway=net_halfway,
                total_spread_cost=float(group["spread_cost"].sum()),
                widest_leg_spread=float(group["spread"].max()),
                leg_count=int(len(group)),
                debit_or_credit=debit_or_credit,
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def load_option_trade_prints(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_TRADE_PRINT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required trade print columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    for column in ["underlying_price", "strike", "trade_price"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "trade_size" in result.columns:
        result["trade_size"] = pd.to_numeric(result["trade_size"], errors="coerce")
    result["option_type"] = result["option_type"].str.lower().str.strip()
    return result.sort_values(["timestamp", "option_type", "strike"]).reset_index(drop=True)


def load_timed_strategy_legs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_TIMED_STRATEGY_LEG_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required timed strategy leg columns: {sorted(missing)}")
    result = frame.copy()
    result["entry_timestamp"] = pd.to_datetime(result["entry_timestamp"])
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    result["strike"] = pd.to_numeric(result["strike"], errors="coerce")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    if "max_time_diff_minutes" in result.columns:
        result["max_time_diff_minutes"] = pd.to_numeric(result["max_time_diff_minutes"], errors="coerce")
    else:
        result["max_time_diff_minutes"] = 5.0
    result["action"] = result["action"].str.lower().str.strip()
    invalid_actions = sorted(set(result["action"]) - {"buy", "sell"})
    if invalid_actions:
        raise ValueError(f"Unsupported leg actions: {invalid_actions}")
    result["option_type"] = result["option_type"].str.lower().str.strip()
    return result.sort_values(["strategy_id", "entry_timestamp", "option_type", "strike"]).reset_index(drop=True)


def estimate_strategy_costs_from_trades(trade_prints: pd.DataFrame, timed_legs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows: list[dict[str, object]] = []
    for _, leg in timed_legs.iterrows():
        candidates = trade_prints[
            (trade_prints["underlying_symbol"] == leg["underlying_symbol"])
            & (trade_prints["expiry_date"] == leg["expiry_date"])
            & (trade_prints["option_type"] == leg["option_type"])
            & (trade_prints["strike"] == leg["strike"])
        ].copy()
        if candidates.empty:
            raise ValueError(
                "No trade prints found for leg "
                f"{leg['strategy_id']} {leg['option_type']} {leg['strike']} {leg['expiry_date']}"
            )
        candidates["time_diff_seconds"] = (candidates["timestamp"] - leg["entry_timestamp"]).abs().dt.total_seconds()
        best = candidates.sort_values("time_diff_seconds").iloc[0]
        max_time_diff_seconds = float(leg["max_time_diff_minutes"]) * 60.0
        if float(best["time_diff_seconds"]) > max_time_diff_seconds:
            raise ValueError(
                "Nearest trade print is outside the allowed window for leg "
                f"{leg['strategy_id']} {leg['option_type']} {leg['strike']}: "
                f"{float(best['time_diff_seconds']) / 60.0:.2f} minutes"
            )
        signed_cost = float(leg["quantity"]) * float(best["trade_price"]) * (1.0 if leg["action"] == "buy" else -1.0)
        matched_rows.append(
            {
                "strategy_id": str(leg["strategy_id"]),
                "entry_timestamp": leg["entry_timestamp"],
                "underlying_symbol": str(leg["underlying_symbol"]),
                "expiry_date": str(leg["expiry_date"]),
                "option_type": str(leg["option_type"]),
                "strike": float(leg["strike"]),
                "action": str(leg["action"]),
                "quantity": float(leg["quantity"]),
                "matched_trade_timestamp": best["timestamp"],
                "matched_underlying_price": float(best["underlying_price"]),
                "matched_trade_price": float(best["trade_price"]),
                "time_diff_seconds": float(best["time_diff_seconds"]),
                "signed_cost": signed_cost,
            }
        )
    matched = pd.DataFrame(matched_rows)
    summaries = (
        matched.groupby(["strategy_id", "entry_timestamp", "underlying_symbol", "expiry_date"], sort=True)
        .agg(
            net_trade_price=("signed_cost", "sum"),
            avg_matched_underlying_price=("matched_underlying_price", "mean"),
            avg_time_diff_seconds=("time_diff_seconds", "mean"),
            max_time_diff_seconds=("time_diff_seconds", "max"),
            leg_count=("signed_cost", "size"),
        )
        .reset_index()
    )
    summaries["debit_or_credit"] = np.where(summaries["net_trade_price"] >= 0.0, "debit", "credit")
    return matched, summaries
