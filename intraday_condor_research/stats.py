from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def probability_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
    checkpoint_col: str = "checkpoint",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        grouped = (
            per_checkpoint.groupby([checkpoint_col, "move_bucket"])[excursion_col]
            .agg(
                observations="size",
                p_le_0_5x=lambda values: (values <= 0.5).mean(),
                p_le_1_0x=lambda values: (values <= 1.0).mean(),
                p_le_1_5x=lambda values: (values <= 1.5).mean(),
                avg_excursion_widths="mean",
            )
            .reset_index()
        )
        tables[width_label] = grouped
    return tables


def _condor_payoff_from_excursion(
    excursion_widths: pd.Series,
    short_distance_multiple: float,
    credit_ratio: float,
) -> pd.Series:
    # Normalize payoff in units of condor width.
    breach = (excursion_widths - short_distance_multiple).clip(lower=0.0)
    intrinsic = breach.clip(upper=1.0)
    return credit_ratio - intrinsic


def _payoff_summary_row(
    *,
    width: float,
    excursions: pd.Series,
    short_distance: float,
    credit_ratio: float,
) -> dict[str, float]:
    payoff_widths = _condor_payoff_from_excursion(
        excursions,
        short_distance_multiple=short_distance,
        credit_ratio=credit_ratio,
    )
    return {
        "observations": float(len(excursions)),
        "expected_value_widths": float(payoff_widths.mean()),
        "expected_value_points": float(payoff_widths.mean() * width),
        "win_rate": float((payoff_widths > 0).mean()),
        "full_win_rate": float((excursions <= short_distance).mean()),
        "max_loss_rate": float((excursions >= short_distance + 1.0).mean()),
    }


def _breakeven_summary_row(
    *,
    width: float,
    excursions: pd.Series,
    short_distance: float,
) -> dict[str, float]:
    intrinsic_widths = (excursions - short_distance).clip(lower=0.0).clip(upper=1.0)
    breakeven_credit_ratio = float(intrinsic_widths.mean())
    return {
        "observations": float(len(excursions)),
        "breakeven_credit_ratio": breakeven_credit_ratio,
        "breakeven_credit_points": float(breakeven_credit_ratio * width),
        "full_win_rate": float((excursions <= short_distance).mean()),
        "max_loss_rate": float((excursions >= short_distance + 1.0).mean()),
        "avg_intrinsic_widths": breakeven_credit_ratio,
    }


def expected_value_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
    short_distance_multiples: Sequence[float],
    credit_ratios: Sequence[float],
    checkpoint_col: str = "checkpoint",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        rows: list[dict[str, float | str]] = []
        for (checkpoint, move_bucket), group in per_checkpoint.groupby([checkpoint_col, "move_bucket"], dropna=False):
            excursions = group[excursion_col]
            for short_distance in short_distance_multiples:
                for credit_ratio in credit_ratios:
                    row = {
                        "checkpoint": checkpoint,
                        "move_bucket": move_bucket,
                        "short_distance_multiple": short_distance,
                        "credit_ratio": credit_ratio,
                    }
                    row.update(
                        _payoff_summary_row(
                            width=width,
                            excursions=excursions,
                            short_distance=short_distance,
                            credit_ratio=credit_ratio,
                        )
                    )
                    rows.append(row)
        tables[width_label] = pd.DataFrame(rows)
    return tables


def expected_value_checkpoint_summary_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
    short_distance_multiples: Sequence[float],
    credit_ratios: Sequence[float],
    checkpoint_col: str = "checkpoint",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        rows: list[dict[str, float | str]] = []
        for checkpoint, group in per_checkpoint.groupby(checkpoint_col, dropna=False):
            excursions = group[excursion_col]
            for short_distance in short_distance_multiples:
                for credit_ratio in credit_ratios:
                    row = {
                        "checkpoint": checkpoint,
                        "short_distance_multiple": short_distance,
                        "credit_ratio": credit_ratio,
                    }
                    row.update(
                        _payoff_summary_row(
                            width=width,
                            excursions=excursions,
                            short_distance=short_distance,
                            credit_ratio=credit_ratio,
                        )
                    )
                    rows.append(row)
        tables[width_label] = pd.DataFrame(rows)
    return tables


def expected_value_regime_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
    short_distance_multiples: Sequence[float],
    credit_ratios: Sequence[float],
    checkpoint_col: str = "checkpoint",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    group_cols = [checkpoint_col, "vol_regime", "move_bucket"]
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        rows: list[dict[str, float | str]] = []
        for (checkpoint, vol_regime, move_bucket), group in per_checkpoint.groupby(group_cols, dropna=False):
            excursions = group[excursion_col]
            for short_distance in short_distance_multiples:
                for credit_ratio in credit_ratios:
                    row = {
                        "checkpoint": checkpoint,
                        "vol_regime": vol_regime,
                        "move_bucket": move_bucket,
                        "short_distance_multiple": short_distance,
                        "credit_ratio": credit_ratio,
                    }
                    row.update(
                        _payoff_summary_row(
                            width=width,
                            excursions=excursions,
                            short_distance=short_distance,
                            credit_ratio=credit_ratio,
                        )
                    )
                    rows.append(row)
        tables[width_label] = pd.DataFrame(rows)
    return tables


def expected_value_vol_summary_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
    short_distance_multiples: Sequence[float],
    credit_ratios: Sequence[float],
    checkpoint_col: str = "checkpoint",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    group_cols = [checkpoint_col, "vol_regime"]
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        rows: list[dict[str, float | str]] = []
        for (checkpoint, vol_regime), group in per_checkpoint.groupby(group_cols, dropna=False):
            excursions = group[excursion_col]
            for short_distance in short_distance_multiples:
                for credit_ratio in credit_ratios:
                    row = {
                        "checkpoint": checkpoint,
                        "vol_regime": vol_regime,
                        "short_distance_multiple": short_distance,
                        "credit_ratio": credit_ratio,
                    }
                    row.update(
                        _payoff_summary_row(
                            width=width,
                            excursions=excursions,
                            short_distance=short_distance,
                            credit_ratio=credit_ratio,
                        )
                    )
                    rows.append(row)
        tables[width_label] = pd.DataFrame(rows)
    return tables


def breakeven_credit_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
    short_distance_multiples: Sequence[float],
    checkpoint_col: str = "checkpoint",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    group_cols = [checkpoint_col, "vol_regime"]
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        rows: list[dict[str, float | str]] = []
        for (checkpoint, vol_regime), group in per_checkpoint.groupby(group_cols, dropna=False):
            excursions = group[excursion_col]
            for short_distance in short_distance_multiples:
                row = {
                    "checkpoint": checkpoint,
                    "vol_regime": vol_regime,
                    "short_distance_multiple": short_distance,
                }
                row.update(
                    _breakeven_summary_row(
                        width=width,
                        excursions=excursions,
                        short_distance=short_distance,
                    )
                )
                rows.append(row)
        tables[width_label] = pd.DataFrame(rows)
    return tables


def decision_credit_tables(
    breakeven_tables: dict[str, pd.DataFrame],
    ev_to_max_loss_ratio: float,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for width_label, table in breakeven_tables.items():
        width_value = float(width_label)
        result = table.copy()
        result["decision_ev_to_max_loss_ratio"] = ev_to_max_loss_ratio
        result["required_credit_ratio"] = (
            result["breakeven_credit_ratio"] + ev_to_max_loss_ratio
        ) / (1.0 + ev_to_max_loss_ratio)
        result["required_credit_points"] = result["required_credit_ratio"] * width_value
        tables[width_label] = result
    return tables


def regime_tables(
    per_checkpoint: pd.DataFrame,
    widths: Sequence[float],
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for width in widths:
        width_label = str(int(width)) if float(width).is_integer() else str(width)
        excursion_col = f"remaining_abs_excursion_widths_{width_label}"
        table = (
            per_checkpoint.groupby(["checkpoint", "vol_regime", "abs_move_bucket"])[excursion_col]
            .agg(observations="size", p_le_1_0x=lambda values: (values <= 1.0).mean(), avg_excursion="mean")
            .reset_index()
        )
        results[width_label] = table
    return results


def save_tables(output_dir: str | Path, tables: dict[str, pd.DataFrame], prefix: str) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key, table in tables.items():
        path = root / f"{prefix}_{key}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    return paths
