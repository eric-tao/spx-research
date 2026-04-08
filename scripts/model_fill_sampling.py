from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from intraday_condor_research.fill_sampling import (
    enrich_fill_samples,
    fit_fill_model,
    load_fill_samples,
    summarize_fill_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model 0DTE option fill behavior versus VIX, time of day, and moneyness from sampled observations"
    )
    parser.add_argument("--input", required=True, help="CSV of sampled single-option fills with bid/ask and VIX")
    parser.add_argument("--output-dir", default="artifacts/fill_sampling_model")
    parser.add_argument(
        "--target-column",
        default="adverse_fill_from_mid",
        choices=["adverse_fill_from_mid", "adverse_fill_pct_spread", "adverse_fill_pct_half_spread"],
        help="Target to fit for the linear fill model",
    )
    args = parser.parse_args()

    samples = load_fill_samples(args.input)
    enriched = enrich_fill_samples(samples)
    summary = summarize_fill_samples(samples)
    model = fit_fill_model(samples, target_column=args.target_column)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "enriched_fill_samples.csv"
    summary_path = output_dir / "fill_bucket_summary.csv"
    coeff_path = output_dir / "fill_model_coefficients.csv"

    enriched.to_csv(enriched_path, index=False)
    summary.to_csv(summary_path, index=False)
    coeff_rows = [{"feature": "intercept", "coefficient": model.intercept}]
    coeff_rows.extend({"feature": name, "coefficient": coef} for name, coef in zip(model.feature_names, model.coefficients))
    pd.DataFrame(coeff_rows).to_csv(coeff_path, index=False)

    print("0DTE Fill Sampling Model")
    print(f"samples={len(samples)}")
    print(f"target={model.target_name}")
    print(f"r_squared={model.r_squared:.4f}")
    print(f"mae={model.mae:.4f}")
    print(f"rmse={model.rmse:.4f}")
    print(f"enriched_fill_samples={enriched_path}")
    print(f"fill_bucket_summary={summary_path}")
    print(f"fill_model_coefficients={coeff_path}")


if __name__ == "__main__":
    main()

