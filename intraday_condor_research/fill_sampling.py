from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_FILL_SAMPLE_COLUMNS = {
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "vix_price",
    "expiry_date",
    "option_type",
    "strike",
    "side",
    "bid",
    "ask",
    "actual_fill_price",
}


@dataclass
class FillModel:
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    mae: float
    rmse: float
    target_name: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = frame[self.feature_names].to_numpy(dtype=float)
        return self.intercept + design @ self.coefficients


def load_fill_samples(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_FILL_SAMPLE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required fill sample columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    for column in ["underlying_price", "vix_price", "strike", "bid", "ask", "actual_fill_price"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["option_type"] = result["option_type"].str.lower().str.strip()
    result["side"] = result["side"].str.lower().str.strip()
    invalid_sides = sorted(set(result["side"]) - {"buy", "sell"})
    if invalid_sides:
        raise ValueError(f"Unsupported side values: {invalid_sides}")
    return result.sort_values(["timestamp", "option_type", "strike"]).reset_index(drop=True)


def enrich_fill_samples(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["mid"] = 0.5 * (result["bid"] + result["ask"])
    result["spread"] = result["ask"] - result["bid"]
    result["half_spread"] = 0.5 * result["spread"]
    result["distance_points"] = np.where(
        result["option_type"].eq("call"),
        result["strike"] - result["underlying_price"],
        result["underlying_price"] - result["strike"],
    )
    result["abs_distance_points"] = result["distance_points"].abs()
    result["distance_pct_underlying"] = result["distance_points"] / result["underlying_price"]
    result["minutes_from_open"] = (
        result["timestamp"].dt.hour * 60.0
        + result["timestamp"].dt.minute
        + result["timestamp"].dt.second / 60.0
        - (9 * 60 + 30)
    )
    result["minutes_to_close"] = 390.0 - result["minutes_from_open"]
    result["is_call"] = result["option_type"].eq("call").astype(float)
    result["is_buy"] = result["side"].eq("buy").astype(float)

    # Positive means worse than midpoint for the trader, regardless of side.
    result["adverse_fill_from_mid"] = np.where(
        result["side"].eq("buy"),
        result["actual_fill_price"] - result["mid"],
        result["mid"] - result["actual_fill_price"],
    )
    result["adverse_fill_pct_spread"] = np.where(
        result["spread"] > 0,
        result["adverse_fill_from_mid"] / result["spread"],
        np.nan,
    )
    result["adverse_fill_pct_half_spread"] = np.where(
        result["half_spread"] > 0,
        result["adverse_fill_from_mid"] / result["half_spread"],
        np.nan,
    )
    result["natural_side_price"] = np.where(result["side"].eq("buy"), result["ask"], result["bid"])
    result["best_side_price"] = np.where(result["side"].eq("buy"), result["bid"], result["ask"])
    result["distance_to_natural"] = np.where(
        result["side"].eq("buy"),
        result["ask"] - result["actual_fill_price"],
        result["actual_fill_price"] - result["bid"],
    )
    result["distance_to_best"] = np.where(
        result["side"].eq("buy"),
        result["actual_fill_price"] - result["bid"],
        result["ask"] - result["actual_fill_price"],
    )

    result["vix_bucket"] = pd.cut(
        result["vix_price"],
        bins=[-np.inf, 15, 20, 25, 30, np.inf],
        labels=["sub_15", "15_20", "20_25", "25_30", "30_plus"],
        include_lowest=True,
    ).astype(str)
    result["time_bucket"] = pd.cut(
        result["minutes_from_open"],
        bins=[-np.inf, 30, 90, 210, 330, np.inf],
        labels=["open_30m", "morning", "midday", "afternoon", "late_day"],
        include_lowest=True,
    ).astype(str)
    result["moneyness_bucket"] = pd.cut(
        result["distance_pct_underlying"],
        bins=[-np.inf, -0.02, -0.01, -0.0025, 0.0025, 0.01, 0.02, np.inf],
        labels=["deep_itm", "itm", "slightly_itm", "atm", "slightly_otm", "otm", "far_otm"],
        include_lowest=True,
    ).astype(str)
    return result


def summarize_fill_samples(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_fill_samples(frame)
    summary = (
        enriched.groupby(["vix_bucket", "time_bucket", "option_type", "moneyness_bucket", "side"], dropna=False)
        .agg(
            observations=("mid", "size"),
            avg_vix=("vix_price", "mean"),
            avg_mid=("mid", "mean"),
            avg_spread=("spread", "mean"),
            avg_adverse_fill_from_mid=("adverse_fill_from_mid", "mean"),
            median_adverse_fill_from_mid=("adverse_fill_from_mid", "median"),
            avg_adverse_fill_pct_spread=("adverse_fill_pct_spread", "mean"),
            avg_adverse_fill_pct_half_spread=("adverse_fill_pct_half_spread", "mean"),
        )
        .reset_index()
        .sort_values(["vix_bucket", "time_bucket", "option_type", "moneyness_bucket", "side"])
    )
    return summary


def fit_fill_model(
    frame: pd.DataFrame,
    *,
    target_column: str = "adverse_fill_from_mid",
    feature_names: Sequence[str] | None = None,
) -> FillModel:
    enriched = enrich_fill_samples(frame)
    if feature_names is None:
        feature_names = [
            "vix_price",
            "minutes_from_open",
            "spread",
            "mid",
            "abs_distance_points",
            "is_call",
            "is_buy",
        ]
    design = enriched[list(feature_names)].to_numpy(dtype=float)
    target = enriched[target_column].to_numpy(dtype=float)
    design_with_intercept = np.column_stack([np.ones(len(design)), design])
    coefficients, _, _, _ = np.linalg.lstsq(design_with_intercept, target, rcond=None)
    intercept = float(coefficients[0])
    betas = coefficients[1:]
    predictions = intercept + design @ betas
    residuals = target - predictions
    target_mean = float(np.mean(target))
    total_ss = float(np.sum((target - target_mean) ** 2))
    residual_ss = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return FillModel(
        feature_names=list(feature_names),
        coefficients=betas,
        intercept=intercept,
        r_squared=r_squared,
        mae=mae,
        rmse=rmse,
        target_name=target_column,
    )

