from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set


def load_event_calendar(path: str | Path) -> Dict[date, Set[str]]:
    calendar: DefaultDict[date, Set[str]] = defaultdict(set)
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "event_type"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns in {path}: {missing}")
        for row in reader:
            calendar[date.fromisoformat(row["date"].strip())].add(row["event_type"].strip())
    return dict(calendar)


def event_flags(on_date: date, calendar: Dict[date, Set[str]], known_events: Iterable[str]) -> Dict[str, float]:
    todays_events = calendar.get(on_date, set())
    flags = {f"event_{name.lower()}": 1.0 if name in todays_events else 0.0 for name in known_events}
    flags["event_any"] = 1.0 if todays_events else 0.0
    return flags


def collect_event_names(calendar: Dict[date, Set[str]]) -> List[str]:
    names: Set[str] = set()
    for entries in calendar.values():
        names.update(entries)
    return sorted(names)
