from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://api.twelvedata.com"


class TwelveDataError(RuntimeError):
    pass


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{BASE_URL}/{path}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("status") == "error" or payload.get("code"):
        raise TwelveDataError(payload.get("message", "Unknown Twelve Data error"))
    return payload


def search_symbol(symbol: str, api_key: str) -> pd.DataFrame:
    payload = _get_json("symbol_search", {"symbol": symbol, "show_plan": "true", "apikey": api_key})
    return pd.DataFrame(payload.get("data", []))


def fetch_time_series(
    symbol: str,
    api_key: str,
    interval: str = "5min",
    outputsize: int = 5000,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    payload = _get_json("time_series", params)
    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(values)
    rename_map = {"datetime": "timestamp"}
    frame = frame.rename(columns=rename_map)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def save_time_series(frame: pd.DataFrame, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.suffix.lower() == ".parquet":
        frame.to_parquet(file_path, index=False)
    else:
        frame.to_csv(file_path, index=False)
    return file_path
