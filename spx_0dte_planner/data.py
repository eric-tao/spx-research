from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class DailyBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    return float(raw)


def load_daily_bars(path: str | Path) -> List[DailyBar]:
    rows: List[DailyBar] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "open", "high", "low", "close"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns in {path}: {missing}")
        for row in reader:
            rows.append(
                DailyBar(
                    date=_parse_date(row["date"]),
                    open=_parse_float(row, "open"),
                    high=_parse_float(row, "high"),
                    low=_parse_float(row, "low"),
                    close=_parse_float(row, "close"),
                    volume=_parse_float(row, "volume", 0.0),
                )
            )
    rows.sort(key=lambda item: item.date)
    return rows


def bar_map(bars: Iterable[DailyBar]) -> Dict[date, DailyBar]:
    return {bar.date: bar for bar in bars}


def intersect_dates(*collections: Iterable[DailyBar]) -> List[date]:
    date_sets = [{item.date for item in entries} for entries in collections]
    shared = set.intersection(*date_sets) if date_sets else set()
    return sorted(shared)
