from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spx_0dte_planner.data import load_daily_bars
from spx_0dte_planner.intraday_bridge import (
    backtest_intraday_excursion_bridge,
    load_spx_intraday_frame,
    translate_proxy_intraday_to_target,
)


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest a combined intraday-structure + daily-excursion threshold model on SPX intraday candles"
    )
    parser.add_argument("--intraday", required=True, help="Path to SPX intraday CSV/parquet with timestamp/open/high/low/close")
    parser.add_argument("--spx", default="data/spx_daily.csv")
    parser.add_argument("--vix", default="data/vix_daily.csv")
    parser.add_argument("--events", default="data/events.csv")
    parser.add_argument(
        "--translate-proxy-to-spx",
        action="store_true",
        help="Treat the intraday file as a proxy series such as SPY and translate its session returns onto SPX levels using the SPX daily open",
    )
    parser.add_argument("--checkpoints", default="10:00,10:30,12:00,14:00,15:00,15:30")
    parser.add_argument("--thresholds", default="0.005,0.0075,0.01")
    parser.add_argument("--move-bucket-edges", default="-0.01,-0.005,-0.0025,0.0025,0.005,0.01")
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/intraday_excursion_bridge")
    args = parser.parse_args()

    spx_bars = load_daily_bars(args.spx)
    vix_bars = load_daily_bars(args.vix)
    intraday = load_spx_intraday_frame(args.intraday)
    if args.translate_proxy_to_spx:
        intraday = translate_proxy_intraday_to_target(intraday, target_daily_bars=spx_bars, target_symbol="SPX")
    result = backtest_intraday_excursion_bridge(
        intraday_frame=intraday,
        spx_bars=spx_bars,
        vix_bars=vix_bars,
        events_path=args.events,
        checkpoints=_parse_list(args.checkpoints),
        thresholds=_parse_floats(args.thresholds),
        move_bucket_edges=_parse_floats(args.move_bucket_edges),
        train_end_min_rows=args.min_train_rows,
        prior_strength=args.prior_strength,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "intraday_bridge_rows.csv"
    summary_path = output_dir / "intraday_bridge_summary.csv"
    result.rows.to_csv(rows_path, index=False)
    result.summary.to_csv(summary_path, index=False)

    print("Intraday Excursion Bridge Backtest")
    print(f"rows={len(result.rows)}")
    print(f"summary_rows={len(result.summary)}")
    print(f"rows_path={rows_path}")
    print(f"summary_path={summary_path}")
    if not result.summary.empty:
        print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
