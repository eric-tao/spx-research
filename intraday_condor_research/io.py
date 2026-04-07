from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close"}
REQUIRED_DAILY_COLUMNS = {"date", "open", "high", "low", "close"}


def load_intraday_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported input type: {suffix}")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=False)
    result = result.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in result.columns:
        result["volume"] = 0.0
    return result


def load_daily_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported input type: {suffix}")

    missing = REQUIRED_DAILY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "volume" not in result.columns:
        result["volume"] = 0.0
    result = result.sort_values("date").reset_index(drop=True)
    return result
