from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intraday_condor_research.strategy_costs import (
    estimate_strategy_costs_from_trades,
    load_option_trade_prints,
    load_timed_strategy_legs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate option strategy entry costs from timestamped TradingView option trade prints"
    )
    parser.add_argument("--trades", required=True, help="CSV of timestamped single-option trade prints")
    parser.add_argument("--legs", required=True, help="CSV of timed strategy legs")
    parser.add_argument("--output-dir", default="artifacts/tradingview_trade_costs")
    args = parser.parse_args()

    trades = load_option_trade_prints(args.trades)
    legs = load_timed_strategy_legs(args.legs)
    matched, summaries = estimate_strategy_costs_from_trades(trades, legs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = output_dir / "matched_strategy_legs.csv"
    summary_path = output_dir / "strategy_costs_from_trades.csv"
    matched.to_csv(matched_path, index=False)
    summaries.to_csv(summary_path, index=False)

    print("TradingView Trade-Based Strategy Cost Estimation")
    print(f"trade_rows={len(trades)}")
    print(f"leg_rows={len(legs)}")
    print(f"strategies={summaries['strategy_id'].nunique() if not summaries.empty else 0}")
    print(f"matched_strategy_legs={matched_path}")
    print(f"strategy_costs={summary_path}")


if __name__ == "__main__":
    main()

