from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intraday_condor_research.quote_calibration import (
    build_quote_pairs,
    enrich_quote_snapshots,
    fit_linear_quote_model,
    load_tradingview_quote_snapshots,
    summarize_fill_offsets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a simple condor quote-change model from manual TradingView quote snapshots")
    parser.add_argument("--input", required=True, help="CSV of manual option quote snapshots")
    parser.add_argument("--output-dir", default="artifacts/tradingview_quote_calibration")
    parser.add_argument(
        "--target-column",
        default="delta_condor_mid_credit",
        choices=["delta_condor_mid_credit", "delta_condor_bid_credit", "delta_actual_fill_credit"],
        help="Target to fit. delta_actual_fill_credit requires actual_fill_credit in the snapshots.",
    )
    args = parser.parse_args()

    snapshots = load_tradingview_quote_snapshots(args.input)
    enriched = enrich_quote_snapshots(snapshots)
    pairs = build_quote_pairs(enriched)
    model = fit_linear_quote_model(pairs, target_column=args.target_column)
    fill_offsets = summarize_fill_offsets(enriched)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "enriched_snapshots.csv"
    pairs_path = output_dir / "quote_pairs.csv"
    coeff_path = output_dir / "model_coefficients.csv"
    fill_path = output_dir / "fill_offset_summary.csv"

    enriched.to_csv(enriched_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    coeff_rows = [{"feature": "intercept", "coefficient": model.intercept}]
    coeff_rows.extend({"feature": name, "coefficient": coef} for name, coef in zip(model.feature_names, model.coefficients))
    import pandas as pd

    pd.DataFrame(coeff_rows).to_csv(coeff_path, index=False)
    if not fill_offsets.empty:
        fill_offsets.to_csv(fill_path, index=False)

    print("TradingView Quote Calibration")
    print(f"snapshots={len(enriched)}")
    print(f"pairs={len(pairs)}")
    print(f"target={model.target_name}")
    print(f"r_squared={model.r_squared:.4f}")
    print(f"mae={model.mae:.4f}")
    print(f"rmse={model.rmse:.4f}")
    print(f"enriched_snapshots={enriched_path}")
    print(f"quote_pairs={pairs_path}")
    print(f"model_coefficients={coeff_path}")
    if not fill_offsets.empty:
        print(f"fill_offset_summary={fill_path}")
    else:
        print("fill_offset_summary=not_available")


if __name__ == "__main__":
    main()
