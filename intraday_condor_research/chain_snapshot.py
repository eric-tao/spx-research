from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_CHAIN_SNAPSHOT_COLUMNS = {
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "expiry_date",
    "strike",
    "call_bid",
    "call_ask",
    "put_bid",
    "put_ask",
}


def load_tradingview_chain_snapshot(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_CHAIN_SNAPSHOT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required chain snapshot columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    numeric_columns = [
        "underlying_price",
        "strike",
        "call_bid",
        "call_ask",
        "put_bid",
        "put_ask",
    ]
    optional_numeric_columns = ["vix_price", "call_last_price", "put_last_price", "call_fill_price", "put_fill_price"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in optional_numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "snapshot_id" not in result.columns:
        result["snapshot_id"] = result.apply(
            lambda row: _default_snapshot_id(
                underlying_symbol=str(row["underlying_symbol"]),
                timestamp=pd.Timestamp(row["timestamp"]),
            ),
            axis=1,
        )
    result["snapshot_id"] = result["snapshot_id"].astype(str)
    result["underlying_symbol"] = result["underlying_symbol"].astype(str).str.strip()
    if "note" not in result.columns:
        result["note"] = ""
    return result.sort_values(["snapshot_id", "strike"]).reset_index(drop=True)


def extract_single_option_quotes_from_chain_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        base = {
            "snapshot_id": str(row["snapshot_id"]),
            "timestamp": pd.Timestamp(row["timestamp"]),
            "underlying_symbol": str(row["underlying_symbol"]),
            "underlying_price": float(row["underlying_price"]),
            "expiry_date": str(row["expiry_date"]),
            "strike": float(row["strike"]),
            "note": str(row.get("note", "")),
        }
        call_bid = float(row["call_bid"])
        call_ask = float(row["call_ask"])
        put_bid = float(row["put_bid"])
        put_ask = float(row["put_ask"])
        if pd.notna(call_bid) and pd.notna(call_ask):
            rows.append(
                {
                    **base,
                    "option_type": "call",
                    "bid": call_bid,
                    "ask": call_ask,
                    "last_price": _optional_float(row, "call_last_price"),
                    "actual_fill_price": _optional_float(row, "call_fill_price"),
                }
            )
        if pd.notna(put_bid) and pd.notna(put_ask):
            rows.append(
                {
                    **base,
                    "option_type": "put",
                    "bid": put_bid,
                    "ask": put_ask,
                    "last_price": _optional_float(row, "put_last_price"),
                    "actual_fill_price": _optional_float(row, "put_fill_price"),
                }
            )
    return pd.DataFrame(rows).sort_values(["snapshot_id", "option_type", "strike"]).reset_index(drop=True)


def extract_vix_snapshots_from_chain_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    if "vix_price" not in frame.columns:
        return pd.DataFrame(columns=["snapshot_id", "timestamp", "vix_price", "note"])
    available = frame.dropna(subset=["vix_price"]).copy()
    if available.empty:
        return pd.DataFrame(columns=["snapshot_id", "timestamp", "vix_price", "note"])
    rows: list[dict[str, object]] = []
    for snapshot_id, group in available.groupby("snapshot_id", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "snapshot_id": str(snapshot_id),
                "timestamp": pd.Timestamp(first["timestamp"]),
                "vix_price": float(first["vix_price"]),
                "note": str(first.get("note", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["snapshot_id", "timestamp"]).reset_index(drop=True)


def _default_snapshot_id(*, underlying_symbol: str, timestamp: pd.Timestamp) -> str:
    return f"{underlying_symbol.lower()}_{timestamp.strftime('%Y_%m_%d_%H%M')}"


def _optional_float(row: pd.Series, key: str) -> float | None:
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])
