from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_QUOTE_COLUMNS = {
    "trade_id",
    "timestamp",
    "underlying_price",
    "expiry_date",
    "short_put_strike",
    "long_put_strike",
    "short_call_strike",
    "long_call_strike",
    "short_put_bid",
    "short_put_ask",
    "long_put_bid",
    "long_put_ask",
    "short_call_bid",
    "short_call_ask",
    "long_call_bid",
    "long_call_ask",
}


@dataclass
class LinearQuoteModel:
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


def load_tradingview_quote_snapshots(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    frame = pd.read_csv(file_path)
    missing = REQUIRED_QUOTE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["expiry_date"] = pd.to_datetime(result["expiry_date"]).dt.date.astype(str)
    numeric_columns = sorted(REQUIRED_QUOTE_COLUMNS - {"trade_id", "timestamp", "expiry_date"})
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "actual_fill_credit" in result.columns:
        result["actual_fill_credit"] = pd.to_numeric(result["actual_fill_credit"], errors="coerce")
    if "note" not in result.columns:
        result["note"] = ""
    return result.sort_values(["trade_id", "timestamp"]).reset_index(drop=True)


def enrich_quote_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["width_points"] = result["long_call_strike"] - result["short_call_strike"]
    result["put_width_points"] = result["short_put_strike"] - result["long_put_strike"]
    if not np.allclose(result["width_points"], result["put_width_points"]):
        raise ValueError("Put and call wing widths do not match for all rows.")

    result["short_put_mid"] = 0.5 * (result["short_put_bid"] + result["short_put_ask"])
    result["long_put_mid"] = 0.5 * (result["long_put_bid"] + result["long_put_ask"])
    result["short_call_mid"] = 0.5 * (result["short_call_bid"] + result["short_call_ask"])
    result["long_call_mid"] = 0.5 * (result["long_call_bid"] + result["long_call_ask"])

    # Natural sell credit for the whole condor.
    result["condor_bid_credit"] = (
        result["short_put_bid"]
        + result["short_call_bid"]
        - result["long_put_ask"]
        - result["long_call_ask"]
    )
    result["condor_ask_credit"] = (
        result["short_put_ask"]
        + result["short_call_ask"]
        - result["long_put_bid"]
        - result["long_call_bid"]
    )
    result["condor_mid_credit"] = (
        result["short_put_mid"]
        + result["short_call_mid"]
        - result["long_put_mid"]
        - result["long_call_mid"]
    )
    result["condor_halfway_credit"] = 0.5 * (result["condor_bid_credit"] + result["condor_mid_credit"])
    result["short_distance_points"] = (
        (result["short_call_strike"] - result["underlying_price"])
        .abs()
        .combine((result["underlying_price"] - result["short_put_strike"]).abs(), func=min)
    )
    result["short_distance_widths"] = result["short_distance_points"] / result["width_points"]
    result["credit_ratio_mid"] = result["condor_mid_credit"] / result["width_points"]
    result["credit_ratio_bid"] = result["condor_bid_credit"] / result["width_points"]
    if "actual_fill_credit" in result.columns:
        result["fill_minus_mid_credit"] = result["actual_fill_credit"] - result["condor_mid_credit"]
    return result


def build_quote_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for trade_id, group in frame.groupby("trade_id", sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        for idx in range(len(ordered) - 1):
            start = ordered.iloc[idx]
            end = ordered.iloc[idx + 1]
            minutes_elapsed = float((end["timestamp"] - start["timestamp"]).total_seconds() / 60.0)
            if minutes_elapsed <= 0:
                continue
            row: dict[str, float | str] = {
                "trade_id": trade_id,
                "start_timestamp": start["timestamp"],
                "end_timestamp": end["timestamp"],
                "session_date": str(pd.Timestamp(start["timestamp"]).date()),
                "expiry_date": start["expiry_date"],
                "width_points": float(start["width_points"]),
                "start_underlying_price": float(start["underlying_price"]),
                "end_underlying_price": float(end["underlying_price"]),
                "delta_underlying_points": float(end["underlying_price"] - start["underlying_price"]),
                "abs_delta_underlying_points": float(abs(end["underlying_price"] - start["underlying_price"])),
                "minutes_elapsed": minutes_elapsed,
                "start_condor_bid_credit": float(start["condor_bid_credit"]),
                "end_condor_bid_credit": float(end["condor_bid_credit"]),
                "start_condor_mid_credit": float(start["condor_mid_credit"]),
                "end_condor_mid_credit": float(end["condor_mid_credit"]),
                "delta_condor_bid_credit": float(end["condor_bid_credit"] - start["condor_bid_credit"]),
                "delta_condor_mid_credit": float(end["condor_mid_credit"] - start["condor_mid_credit"]),
                "start_credit_ratio_mid": float(start["credit_ratio_mid"]),
                "start_short_distance_widths": float(start["short_distance_widths"]),
            }
            if "actual_fill_credit" in frame.columns and pd.notna(start.get("actual_fill_credit")) and pd.notna(end.get("actual_fill_credit")):
                row["start_actual_fill_credit"] = float(start["actual_fill_credit"])
                row["end_actual_fill_credit"] = float(end["actual_fill_credit"])
                row["delta_actual_fill_credit"] = float(end["actual_fill_credit"] - start["actual_fill_credit"])
            rows.append(row)
    return pd.DataFrame(rows)


def fit_linear_quote_model(
    pairs: pd.DataFrame,
    *,
    target_column: str = "delta_condor_mid_credit",
    feature_names: Sequence[str] | None = None,
) -> LinearQuoteModel:
    if pairs.empty:
        raise ValueError("No quote pairs available to fit.")
    if feature_names is None:
        feature_names = [
            "delta_underlying_points",
            "abs_delta_underlying_points",
            "minutes_elapsed",
            "start_condor_mid_credit",
            "width_points",
            "start_short_distance_widths",
        ]
    design = pairs[list(feature_names)].to_numpy(dtype=float)
    target = pairs[target_column].to_numpy(dtype=float)
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
    return LinearQuoteModel(
        feature_names=list(feature_names),
        coefficients=betas,
        intercept=intercept,
        r_squared=r_squared,
        mae=mae,
        rmse=rmse,
        target_name=target_column,
    )


def summarize_fill_offsets(frame: pd.DataFrame) -> pd.DataFrame:
    if "actual_fill_credit" not in frame.columns or frame["actual_fill_credit"].isna().all():
        return pd.DataFrame()
    enriched = frame.copy()
    enriched["fill_minus_mid_credit"] = enriched["actual_fill_credit"] - enriched["condor_mid_credit"]
    summary = (
        enriched.groupby("trade_id")["fill_minus_mid_credit"]
        .agg(observations="size", avg_fill_minus_mid="mean", min_fill_minus_mid="min", max_fill_minus_mid="max")
        .reset_index()
    )
    return summary
