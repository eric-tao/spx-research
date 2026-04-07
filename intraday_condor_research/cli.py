from __future__ import annotations

import argparse
import os
from pathlib import Path

from .features import build_research_frames
from .io import load_daily_frame, load_intraday_frame
from .plots import save_expected_value_lines, save_probability_heatmaps, save_probability_lines, save_viability_heatmaps
from .regimes import attach_regime_columns
from .session import filter_rth, keep_complete_sessions
from .stats import (
    breakeven_credit_tables,
    decision_credit_tables,
    expected_value_checkpoint_summary_tables,
    expected_value_regime_tables,
    expected_value_tables,
    expected_value_vol_summary_tables,
    probability_tables,
    regime_tables,
    save_tables,
)


def _parse_widths(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_checkpoints(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday QQQ/NDX condor research pipeline")
    parser.add_argument("--input", required=True, help="Path to intraday CSV or parquet")
    parser.add_argument("--symbol", default="QQQ", help="Symbol label for outputs")
    parser.add_argument("--output-dir", default="artifacts/intraday_condor", help="Directory for datasets, tables, and plots")
    parser.add_argument("--vix-daily", default="data/vix_daily.csv", help="Path to daily VIX OHLC CSV/parquet used for forward-vol proxy")
    parser.add_argument("--vxn-multiplier", type=float, default=1.15, help="Multiplier applied to VIX to estimate VXN for NDX/QQQ forward-vol proxy")
    parser.add_argument("--widths", default="25,50,100,200", help="Comma-separated condor widths")
    parser.add_argument("--checkpoints", default="10:00,10:30,12:00,14:00,15:00,15:30", help="Comma-separated checkpoints")
    parser.add_argument(
        "--short-distance-multiples",
        default="0.5",
        help=(
            "Comma-separated short strike distances as multiples of width. "
            "Default 0.5 means the checkpoint spot is centered between the shorts, "
            "so short-put to short-call distance equals the condor width."
        ),
    )
    parser.add_argument("--credit-ratios", default="0.1,0.2,0.3", help="Comma-separated premium credits as fractions of width")
    parser.add_argument("--viability-credit-ratios", default="0.1,0.15,0.2", help="Comma-separated premium thresholds for viability heatmaps")
    parser.add_argument("--decision-ev-to-max-loss-ratio", type=float, default=0.1, help="Minimum EV as a fraction of max loss for the decision-credit tables")
    parser.add_argument("--width-display-multiplier", type=float, default=1.0, help="Multiplier used only for display labels in viability heatmaps")
    parser.add_argument("--rth-start", default="09:30")
    parser.add_argument("--rth-end", default="16:00")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir) / ".mplconfig"))
    widths = _parse_widths(args.widths)
    checkpoints = _parse_checkpoints(args.checkpoints)
    short_distance_multiples = _parse_floats(args.short_distance_multiples)
    credit_ratios = _parse_floats(args.credit_ratios)
    viability_credit_ratios = _parse_floats(args.viability_credit_ratios)

    raw = load_intraday_frame(args.input)
    vix_daily = load_daily_frame(args.vix_daily) if args.vix_daily else None
    rth = filter_rth(raw, start_time=args.rth_start, end_time=args.rth_end)
    complete = keep_complete_sessions(rth, checkpoints=checkpoints)
    research = build_research_frames(
        complete,
        checkpoints=checkpoints,
        widths=widths,
        symbol=args.symbol,
        vix_daily=vix_daily,
        vxn_multiplier=args.vxn_multiplier,
    )
    per_checkpoint = attach_regime_columns(research.per_checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_day_path = output_dir / "per_day_dataset.csv"
    per_checkpoint_path = output_dir / "per_checkpoint_dataset.csv"
    research.per_day.to_csv(per_day_path, index=False)
    per_checkpoint.to_csv(per_checkpoint_path, index=False)

    prob_tables = probability_tables(per_checkpoint, widths=widths)
    ev_tables = expected_value_tables(
        per_checkpoint,
        widths=widths,
        short_distance_multiples=short_distance_multiples,
        credit_ratios=credit_ratios,
    )
    ev_checkpoint_tables = expected_value_checkpoint_summary_tables(
        per_checkpoint,
        widths=widths,
        short_distance_multiples=short_distance_multiples,
        credit_ratios=credit_ratios,
    )
    ev_regime_tables = expected_value_regime_tables(
        per_checkpoint,
        widths=widths,
        short_distance_multiples=short_distance_multiples,
        credit_ratios=credit_ratios,
    )
    ev_vol_tables = expected_value_vol_summary_tables(
        per_checkpoint,
        widths=widths,
        short_distance_multiples=short_distance_multiples,
        credit_ratios=credit_ratios,
    )
    breakeven_tables = breakeven_credit_tables(
        per_checkpoint,
        widths=widths,
        short_distance_multiples=short_distance_multiples,
    )
    decision_tables = decision_credit_tables(
        breakeven_tables,
        ev_to_max_loss_ratio=args.decision_ev_to_max_loss_ratio,
    )
    regime_breakdowns = regime_tables(per_checkpoint, widths=widths)
    save_tables(output_dir / "tables", prob_tables, prefix="probability_table")
    save_tables(output_dir / "tables", breakeven_tables, prefix="breakeven_credit_table")
    save_tables(output_dir / "tables", decision_tables, prefix="decision_credit_table")
    save_tables(output_dir / "tables", ev_tables, prefix="expected_value_table")
    save_tables(output_dir / "tables", ev_checkpoint_tables, prefix="expected_value_checkpoint_table")
    save_tables(output_dir / "tables", ev_regime_tables, prefix="expected_value_regime_table")
    save_tables(output_dir / "tables", ev_vol_tables, prefix="expected_value_vol_table")
    save_tables(output_dir / "tables", regime_breakdowns, prefix="regime_table")
    heatmaps = save_probability_heatmaps(prob_tables, output_dir / "plots")
    lines = save_probability_lines(prob_tables, output_dir / "plots")
    ev_lines = save_expected_value_lines(ev_tables, output_dir / "plots")
    viability_heatmaps = save_viability_heatmaps(
        breakeven_tables,
        output_dir / "plots",
        credit_ratio_thresholds=viability_credit_ratios,
        width_display_multiplier=args.width_display_multiplier,
    )

    print("Intraday Condor Research")
    print(f"input_rows={len(raw)}, rth_rows={len(rth)}, complete_session_rows={len(complete)}")
    print(f"per_day_rows={len(research.per_day)}, per_checkpoint_rows={len(per_checkpoint)}")
    print(f"per_day_dataset={per_day_path}")
    print(f"per_checkpoint_dataset={per_checkpoint_path}")
    print(f"plot_count={len(heatmaps) + len(lines) + len(ev_lines) + len(viability_heatmaps)}")
    print("Limitations: this is not an options PnL backtest; width is only a normalization proxy and no chain/implied move data is used yet.")


if __name__ == "__main__":
    main()
