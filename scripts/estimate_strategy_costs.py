from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intraday_condor_research.strategy_costs import (
    estimate_strategy_costs,
    load_option_quote_snapshots,
    load_strategy_legs,
    summarize_option_spreads,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SPX option leg spreads and multi-leg strategy costs from manual TradingView quote snapshots"
    )
    parser.add_argument("--quotes", required=True, help="CSV of manual single-option quote snapshots")
    parser.add_argument("--legs", required=True, help="CSV of strategy legs keyed to snapshot_id")
    parser.add_argument("--output-dir", default="artifacts/tradingview_strategy_costs")
    args = parser.parse_args()

    quotes = load_option_quote_snapshots(args.quotes)
    legs = load_strategy_legs(args.legs)
    spread_summary = summarize_option_spreads(quotes)
    strategy_costs = estimate_strategy_costs(quotes, legs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spread_path = output_dir / "option_spread_summary.csv"
    strategy_path = output_dir / "strategy_costs.csv"

    spread_summary.to_csv(spread_path, index=False)
    strategy_costs.to_csv(strategy_path, index=False)

    print("TradingView Strategy Cost Estimation")
    print(f"quote_rows={len(quotes)}")
    print(f"leg_rows={len(legs)}")
    print(f"strategies={strategy_costs['strategy_id'].nunique() if not strategy_costs.empty else 0}")
    print(f"option_spread_summary={spread_path}")
    print(f"strategy_costs={strategy_path}")


if __name__ == "__main__":
    main()

