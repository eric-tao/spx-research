from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class ResearchFrames:
    per_day: pd.DataFrame
    per_checkpoint: pd.DataFrame


def _prepare_vix_daily(vix_daily: pd.DataFrame | None) -> pd.DataFrame | None:
    if vix_daily is None or vix_daily.empty:
        return None
    prepared = vix_daily.copy()
    prepared["date_ts"] = pd.to_datetime(prepared["date"])
    prepared["prev_close"] = prepared["close"].shift(1)
    return prepared.sort_values("date_ts").reset_index(drop=True)


def _lookup_vix_context(vix_daily: pd.DataFrame | None, session_date: object) -> dict[str, float | bool]:
    if vix_daily is None or vix_daily.empty:
        return {
            "vix_open": np.nan,
            "vix_prev_close": np.nan,
            "vix_reference": np.nan,
            "vix_exact_match": False,
        }
    session_ts = pd.Timestamp(session_date)
    eligible = vix_daily[vix_daily["date_ts"] <= session_ts]
    if eligible.empty:
        return {
            "vix_open": np.nan,
            "vix_prev_close": np.nan,
            "vix_reference": np.nan,
            "vix_exact_match": False,
        }
    row = eligible.iloc[-1]
    exact_match = bool(row["date_ts"] == session_ts)
    vix_open = float(row["open"]) if exact_match else np.nan
    vix_prev_close = float(row["prev_close"]) if exact_match and pd.notna(row["prev_close"]) else float(row["close"])
    vix_reference = float(row["open"]) if exact_match else float(row["close"])
    return {
        "vix_open": vix_open,
        "vix_prev_close": vix_prev_close,
        "vix_reference": vix_reference,
        "vix_exact_match": exact_match,
    }


def _daily_implied_move_pct(index_level: float) -> float:
    if not np.isfinite(index_level) or index_level <= 0:
        return 0.0
    return index_level / 100.0 / np.sqrt(252.0)


def _project_intraday_realized_vol_pct(last_bars: pd.DataFrame, total_bars: int) -> float:
    if len(last_bars) < 5 or total_bars <= 1:
        return 0.0
    ohlc = last_bars[["open", "high", "low", "close"]].astype(float)
    if (ohlc <= 0).any().any():
        return 0.0
    log_hl = np.log(ohlc["high"] / ohlc["low"])
    log_co = np.log(ohlc["close"] / ohlc["open"])
    per_bar_variance = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    mean_bar_variance = float(per_bar_variance.clip(lower=0.0).mean())
    projected_variance = total_bars * mean_bar_variance
    return float(np.sqrt(max(projected_variance, 0.0)))


def build_research_frames(
    frame: pd.DataFrame,
    checkpoints: Sequence[str],
    widths: Sequence[float],
    symbol: str,
    vix_daily: pd.DataFrame | None = None,
    vxn_multiplier: float = 1.15,
) -> ResearchFrames:
    working = frame.copy()
    working["clock"] = working["timestamp"].dt.strftime("%H:%M")
    grouped = list(working.groupby("session_date", sort=True))
    session_lengths = [len(day_frame) for _, day_frame in grouped]
    total_session_bars = int(np.median(session_lengths)) if session_lengths else 0
    prepared_vix = _prepare_vix_daily(vix_daily)

    day_rows: list[dict[str, float | str]] = []
    checkpoint_rows: list[dict[str, float | str]] = []

    for session_date, day_frame in grouped:
        day_frame = day_frame.sort_values("timestamp").reset_index(drop=True)
        session_open = float(day_frame.iloc[0]["open"])
        session_high = float(day_frame["high"].max())
        session_low = float(day_frame["low"].min())
        session_close = float(day_frame.iloc[-1]["close"])
        if session_open == 0:
            continue
        day_range_pct = (session_high - session_low) / session_open
        realized_move_close = (session_close - session_open) / session_open
        vix_context = _lookup_vix_context(prepared_vix, session_date)
        vix_reference = float(vix_context["vix_reference"]) if np.isfinite(vix_context["vix_reference"]) else np.nan
        est_vxn_reference = vix_reference * vxn_multiplier if np.isfinite(vix_reference) else np.nan
        forward_vol_from_vix_pct = _daily_implied_move_pct(vix_reference)
        forward_vol_from_est_vxn_pct = _daily_implied_move_pct(est_vxn_reference)
        forward_vol_proxy_pct = (
            0.5 * forward_vol_from_vix_pct + 0.5 * forward_vol_from_est_vxn_pct
            if np.isfinite(vix_reference)
            else 0.0
        )

        day_rows.append(
            {
                "symbol": symbol,
                "session_date": str(session_date),
                "session_open": session_open,
                "session_high": session_high,
                "session_low": session_low,
                "session_close": session_close,
                "day_range_pct": day_range_pct,
                "close_move_from_open_pct": realized_move_close,
                "expected_move_proxy_pct": forward_vol_proxy_pct,
                "forward_vol_proxy_pct": forward_vol_proxy_pct,
                "forward_vol_from_vix_pct": forward_vol_from_vix_pct,
                "forward_vol_from_est_vxn_pct": forward_vol_from_est_vxn_pct,
                "projected_intraday_realized_vol_pct": 0.0,
                "vix_open": vix_context["vix_open"],
                "vix_prev_close": vix_context["vix_prev_close"],
                "vix_reference": vix_reference,
                "vix_exact_match": float(vix_context["vix_exact_match"]),
                "estimated_vxn_reference": est_vxn_reference,
                "vxn_multiplier": vxn_multiplier,
                "bar_count": float(len(day_frame)),
            }
        )

        checkpoints_frame = day_frame[day_frame["clock"].isin(checkpoints)]
        for _, row in checkpoints_frame.iterrows():
            checkpoint_time = row["clock"]
            checkpoint_close = float(row["close"])
            checkpoint_open = float(row["open"])
            move_from_open = (checkpoint_close - session_open) / session_open
            prior = day_frame[day_frame["timestamp"] < row["timestamp"]]
            last_five_bars = prior.tail(5)
            projected_intraday_realized_vol_pct = _project_intraday_realized_vol_pct(
                last_five_bars,
                total_bars=total_session_bars,
            )
            combined_vol_proxy_pct = max(forward_vol_proxy_pct, projected_intraday_realized_vol_pct)

            remaining = day_frame[day_frame["timestamp"] >= row["timestamp"]]
            remaining_high = float(remaining["high"].max())
            remaining_low = float(remaining["low"].min())
            remaining_up_excursion = max(remaining_high - checkpoint_close, 0.0)
            remaining_down_excursion = max(checkpoint_close - remaining_low, 0.0)
            remaining_abs_excursion = max(remaining_up_excursion, remaining_down_excursion)

            checkpoint_row: dict[str, float | str] = {
                "symbol": symbol,
                "session_date": str(session_date),
                "checkpoint": checkpoint_time,
                "checkpoint_open": checkpoint_open,
                "checkpoint_close": checkpoint_close,
                "session_open": session_open,
                "move_from_open_points": checkpoint_close - session_open,
                "move_from_open_pct": move_from_open,
                "remaining_up_excursion_points": remaining_up_excursion,
                "remaining_down_excursion_points": remaining_down_excursion,
                "remaining_abs_excursion_points": remaining_abs_excursion,
                "remaining_up_excursion_pct": remaining_up_excursion / session_open,
                "remaining_down_excursion_pct": remaining_down_excursion / session_open,
                "remaining_abs_excursion_pct": remaining_abs_excursion / session_open,
                "expected_move_proxy_pct": combined_vol_proxy_pct,
                "forward_vol_proxy_pct": forward_vol_proxy_pct,
                "forward_vol_from_vix_pct": forward_vol_from_vix_pct,
                "forward_vol_from_est_vxn_pct": forward_vol_from_est_vxn_pct,
                "projected_intraday_realized_vol_pct": projected_intraday_realized_vol_pct,
                "vix_open": vix_context["vix_open"],
                "vix_prev_close": vix_context["vix_prev_close"],
                "vix_reference": vix_reference,
                "vix_exact_match": float(vix_context["vix_exact_match"]),
                "estimated_vxn_reference": est_vxn_reference,
                "vxn_multiplier": vxn_multiplier,
                "elapsed_bars": float(len(prior)),
                "realized_vol_bar_count": float(len(last_five_bars)),
                "total_session_bars": float(total_session_bars),
                "day_range_pct": day_range_pct,
            }
            expected_move_points = combined_vol_proxy_pct * session_open
            for width in widths:
                width_label = str(int(width)) if float(width).is_integer() else str(width)
                checkpoint_row[f"move_from_open_widths_{width_label}"] = (checkpoint_close - session_open) / width
                checkpoint_row[f"remaining_abs_excursion_widths_{width_label}"] = remaining_abs_excursion / width
                checkpoint_row[f"remaining_abs_excursion_le_{width_label}"] = 1.0 if remaining_abs_excursion <= width else 0.0
                checkpoint_row[f"remaining_abs_excursion_le_0.5x_{width_label}"] = 1.0 if remaining_abs_excursion <= 0.5 * width else 0.0
                checkpoint_row[f"remaining_abs_excursion_le_1.5x_{width_label}"] = 1.0 if remaining_abs_excursion <= 1.5 * width else 0.0
                checkpoint_row[f"remaining_abs_excursion_em_{width_label}"] = remaining_abs_excursion / expected_move_points if expected_move_points else 0.0
            checkpoint_rows.append(checkpoint_row)

    return ResearchFrames(
        per_day=pd.DataFrame(day_rows),
        per_checkpoint=pd.DataFrame(checkpoint_rows),
    )
