from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intraday_condor_research.chain_snapshot import (
    extract_single_option_quotes_from_chain_snapshot,
    extract_vix_snapshots_from_chain_snapshot,
    load_tradingview_chain_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize a manual TradingView 0DTE chain snapshot into single-option quote and VIX snapshot files"
    )
    parser.add_argument("--input", required=True, help="CSV of a wide 0DTE chain snapshot")
    parser.add_argument("--quotes-output", required=True, help="Output CSV for normalized single-option quotes")
    parser.add_argument("--vix-output", help="Optional output CSV for aligned VIX snapshots")
    args = parser.parse_args()

    frame = load_tradingview_chain_snapshot(args.input)
    quotes = extract_single_option_quotes_from_chain_snapshot(frame)
    quotes_output = Path(args.quotes_output)
    quotes_output.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(quotes_output, index=False)

    print("TradingView 0DTE Chain Extraction")
    print(f"- input rows: {len(frame)}")
    print(f"- normalized quote rows: {len(quotes)}")
    print(f"- quotes output: {quotes_output}")

    if args.vix_output:
        vix = extract_vix_snapshots_from_chain_snapshot(frame)
        vix_output = Path(args.vix_output)
        vix_output.parent.mkdir(parents=True, exist_ok=True)
        vix.to_csv(vix_output, index=False)
        print(f"- VIX snapshot rows: {len(vix)}")
        print(f"- VIX output: {vix_output}")


if __name__ == "__main__":
    main()
