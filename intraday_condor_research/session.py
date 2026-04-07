from __future__ import annotations

from typing import Iterable

import pandas as pd


def filter_rth(frame: pd.DataFrame, start_time: str = "09:30", end_time: str = "16:00") -> pd.DataFrame:
    result = frame.copy()
    result["session_date"] = result["timestamp"].dt.date
    intraday_time = result["timestamp"].dt.strftime("%H:%M")
    mask = (intraday_time >= start_time) & (intraday_time <= end_time)
    return result.loc[mask].reset_index(drop=True)


def keep_complete_sessions(frame: pd.DataFrame, checkpoints: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    result["clock"] = result["timestamp"].dt.strftime("%H:%M")
    required = set(checkpoints) | {"09:30"}
    latest_checkpoint = max(checkpoints)

    def _is_complete(values: pd.Series) -> bool:
        clocks = set(values)
        if not required.issubset(clocks):
            return False
        return max(clocks) >= latest_checkpoint

    counts = result.groupby("session_date")["clock"].agg(_is_complete)
    keep_dates = counts[counts].index
    return result[result["session_date"].isin(keep_dates)].reset_index(drop=True)
