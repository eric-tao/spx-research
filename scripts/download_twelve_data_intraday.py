from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intraday_condor_research.twelve_data import TwelveDataError, fetch_time_series, save_time_series, search_symbol


def main() -> None:
    parser = argparse.ArgumentParser(description="Download intraday candles from Twelve Data")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--interval", default="5min")
    parser.add_argument("--outputsize", type=int, default=5000)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output", default="data/qqq_intraday_5min.csv")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("TWELVE_DATA_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing Twelve Data API key. Pass --api-key or set TWELVE_DATA_API_KEY.")

    if args.search_only:
        matches = search_symbol(args.symbol, args.api_key)
        print(matches.to_csv(index=False))
        return

    try:
        frame = fetch_time_series(
            symbol=args.symbol,
            api_key=args.api_key,
            interval=args.interval,
            outputsize=args.outputsize,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except TwelveDataError as exc:
        raise SystemExit(str(exc)) from exc

    path = save_time_series(frame, args.output)
    print(f"saved_rows={len(frame)}")
    print(f"output={path}")


if __name__ == "__main__":
    main()
