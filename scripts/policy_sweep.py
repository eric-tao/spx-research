from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd


CHECKPOINT_ORDER = ["10:00", "10:30", "12:00", "14:00", "15:00", "15:30"]
WIDTH_MAP = {
    "1.25": 50,
    "1.875": 75,
    "2.5": 100,
    "3.75": 150,
    "5": 200,
}
REGIME_ORDER = ["low_vol", "mid_vol", "high_vol"]


def _all_non_empty_subsets(items: list[str | int]) -> list[tuple[str | int, ...]]:
    subsets: list[tuple[str | int, ...]] = []
    for r in range(1, len(items) + 1):
        subsets.extend(itertools.combinations(items, r))
    return subsets


def load_trade_candidates(artifact_dir: Path, last_sessions: int, risk_budget: float) -> pd.DataFrame:
    per = pd.read_csv(artifact_dir / "per_checkpoint_dataset.csv")
    per["session_date"] = pd.to_datetime(per["session_date"])
    session_dates = sorted(per["session_date"].drop_duplicates())[-last_sessions:]
    per = per[per["session_date"].isin(session_dates)].copy()

    frames: list[pd.DataFrame] = []
    for qqq_width, ndx_width in WIDTH_MAP.items():
        table = pd.read_csv(artifact_dir / "tables" / f"decision_credit_table_{qqq_width}.csv")
        table = table[table["short_distance_multiple"] == 0.5].copy()
        table["width_ndx"] = ndx_width
        table["credit_ratio"] = table["required_credit_ratio"]
        table["credit_points_ndx"] = table["required_credit_ratio"] * ndx_width
        table["max_loss_points_ndx"] = ndx_width - table["credit_points_ndx"]
        table["score"] = table["credit_ratio"] / (1.0 - table["credit_ratio"])
        frames.append(
            table[
                [
                    "checkpoint",
                    "vol_regime",
                    "width_ndx",
                    "credit_ratio",
                    "credit_points_ndx",
                    "max_loss_points_ndx",
                    "score",
                ]
            ]
        )
    decisions = pd.concat(frames, ignore_index=True)
    decisions = decisions[decisions["credit_ratio"] < 1.0 - 1e-9].copy()

    merged = per.merge(decisions, on=["checkpoint", "vol_regime"], how="left")
    merged["remaining_abs_excursion_points_ndx"] = merged["remaining_abs_excursion_points"] * 40.0
    merged["intrinsic_points_ndx"] = (
        merged["remaining_abs_excursion_points_ndx"] - 0.5 * merged["width_ndx"]
    ).clip(lower=0.0)
    merged["intrinsic_points_ndx"] = merged[["intrinsic_points_ndx", "width_ndx"]].min(axis=1)
    merged["pnl_points_ndx"] = merged["credit_points_ndx"] - merged["intrinsic_points_ndx"]
    merged["max_loss_dollars_per_contract"] = merged["max_loss_points_ndx"] * 100.0
    merged["pnl_dollars_per_contract"] = merged["pnl_points_ndx"] * 100.0
    merged["qty"] = merged["max_loss_dollars_per_contract"].apply(
        lambda risk: 1 if risk >= risk_budget else max(1, math.floor(risk_budget / risk))
    )
    merged["trade_pnl"] = merged["pnl_dollars_per_contract"] * merged["qty"]
    merged["trade_risk"] = merged["max_loss_dollars_per_contract"] * merged["qty"]
    merged["cp_order"] = merged["checkpoint"].map({cp: idx for idx, cp in enumerate(CHECKPOINT_ORDER)})
    return merged


def evaluate_policy(
    candidates: pd.DataFrame,
    *,
    allowed_widths: tuple[int, ...],
    allowed_regimes: tuple[str, ...],
    min_checkpoint: str,
    fill_probability: float,
) -> dict[str, object]:
    working = candidates[
        candidates["width_ndx"].isin(allowed_widths)
        & candidates["vol_regime"].isin(allowed_regimes)
        & (candidates["cp_order"] >= CHECKPOINT_ORDER.index(min_checkpoint))
    ].copy()
    if working.empty:
        return {
            "allowed_widths": ",".join(str(item) for item in allowed_widths),
            "allowed_regimes": ",".join(allowed_regimes),
            "min_checkpoint": min_checkpoint,
            "expected_total_pnl": 0.0,
            "expected_trade_count": 0.0,
            "trade_probability_per_day": 0.0,
            "active_days": 0,
        }

    checkpoint_choices = (
        working.sort_values(["session_date", "cp_order", "score"], ascending=[True, True, False])
        .groupby(["session_date", "checkpoint"], as_index=False)
        .head(1)
        .sort_values(["session_date", "cp_order"])
    )

    expected_total = 0.0
    expected_trade_count = 0.0
    active_days = 0
    for _, group in checkpoint_choices.groupby("session_date"):
        active_days += 1
        survival = 1.0
        for _, row in group.sort_values("cp_order").iterrows():
            fill_weight = fill_probability * survival
            expected_total += fill_weight * row["trade_pnl"]
            expected_trade_count += fill_weight
            survival *= 1.0 - fill_probability

    return {
        "allowed_widths": ",".join(str(item) for item in allowed_widths),
        "allowed_regimes": ",".join(allowed_regimes),
        "min_checkpoint": min_checkpoint,
        "expected_total_pnl": expected_total,
        "expected_trade_count": expected_trade_count,
        "trade_probability_per_day": expected_trade_count / candidates["session_date"].nunique(),
        "active_days": active_days,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep intraday condor policies on the saved width sweep artifacts")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/qqq_intraday_condor_width_sweep",
        help="Artifact directory containing per_checkpoint_dataset.csv and decision tables",
    )
    parser.add_argument("--last-sessions", type=int, default=30)
    parser.add_argument("--fill-probability", type=float, default=0.25)
    parser.add_argument("--starting-equity", type=float, default=100000.0)
    parser.add_argument("--risk-budget", type=float, default=5000.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--output",
        default="artifacts/qqq_intraday_condor_width_sweep/policy_sweep_last30.csv",
        help="Where to write the full policy ranking CSV",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    candidates = load_trade_candidates(artifact_dir, last_sessions=args.last_sessions, risk_budget=args.risk_budget)

    width_subsets = _all_non_empty_subsets(sorted(WIDTH_MAP.values()))
    regime_subsets = _all_non_empty_subsets(REGIME_ORDER)
    rows: list[dict[str, object]] = []
    for widths in width_subsets:
        for regimes in regime_subsets:
            for checkpoint in CHECKPOINT_ORDER:
                row = evaluate_policy(
                    candidates,
                    allowed_widths=tuple(int(item) for item in widths),
                    allowed_regimes=tuple(str(item) for item in regimes),
                    min_checkpoint=checkpoint,
                    fill_probability=args.fill_probability,
                )
                row["expected_end_equity"] = args.starting_equity + float(row["expected_total_pnl"])
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["expected_end_equity", "expected_trade_count", "active_days"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_path, index=False)

    active = ranking[ranking["expected_trade_count"] > 0].copy()
    print(f"saved_policy_ranking={output_path}")
    print("top_overall")
    print(ranking.head(args.top).to_string(index=False))
    print("\ntop_active")
    if active.empty:
        print("No active policies found.")
    else:
        print(active.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
