from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intraday_condor_research.vertical_fills import (
    estimate_vertical_fills_from_quotes,
    load_vertical_fill_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate 5-point SPX vertical fill prices from sampled single-option fills and quote snapshots"
    )
    parser.add_argument("--fill-samples", required=True, help="CSV of sampled single-option fills with bid/ask and VIX")
    parser.add_argument("--quotes", required=True, help="CSV of option quote snapshots for a specific time")
    parser.add_argument("--vix-snapshots", required=True, help="CSV mapping snapshot_id/timestamp to VIX price")
    parser.add_argument("--width-points", type=float, default=5.0, help="Vertical width in points")
    parser.add_argument("--output-dir", default="artifacts/vertical_fill_estimates")
    args = parser.parse_args()

    fill_samples, quote_snapshots, vix_by_snapshot = load_vertical_fill_inputs(
        args.fill_samples,
        args.quotes,
        args.vix_snapshots,
    )
    verticals = estimate_vertical_fills_from_quotes(
        fill_samples=fill_samples,
        quote_snapshots=quote_snapshots,
        vix_by_snapshot=vix_by_snapshot,
        width_points=args.width_points,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verticals_path = output_dir / "vertical_fill_estimates.csv"
    verticals.to_csv(verticals_path, index=False)

    print("Vertical Fill Estimation")
    print(f"fill_samples={len(fill_samples)}")
    print(f"quote_rows={len(quote_snapshots)}")
    print(f"vertical_rows={len(verticals)}")
    print(f"vertical_fill_estimates={verticals_path}")


if __name__ == "__main__":
    main()

