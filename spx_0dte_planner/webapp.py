from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server
from zoneinfo import ZoneInfo

from intraday_condor_research.vertical_fills import estimate_vertical_fills_from_quotes, load_vertical_fill_inputs
import pandas as pd

from .data import load_daily_bars
from .events import collect_event_names, load_event_calendar
from .features import align_and_build
from .live import (
    PriorDayOverrides,
    build_live_feature_row,
    build_continuation_lookup,
    build_regime_probability_lookup,
    build_touch_target_lookup,
    compute_regime_cutoffs,
    default_prediction_date,
    required_history_date,
    predict_live_excursions,
    select_continuation_stats,
    threshold_probabilities,
)
from .model import fit_train_backtest_range_model, predict_backtest_range
from .opportunity_screen import score_vertical_opportunities_after_touch

try:
    from scripts.download_market_data import download_cboe_vix, download_stooq_spx, download_yahoo_spx, write_csv
except Exception:  # pragma: no cover - optional runtime helper
    download_cboe_vix = download_stooq_spx = download_yahoo_spx = write_csv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppState:
    spx_path: str
    vix_path: str
    events_path: str
    underlying_label: str
    max_lag: int
    thresholds: List[float]
    event_names: List[str]
    spx_bars: list
    vix_bars: list
    event_calendar: Dict[date, set[str]]
    range_fit: object
    cutoffs: object
    regime_lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float]]]]]
    continuation_lookup: Dict[str, Dict[str, Dict[str, Dict[float, Dict[str, float | str]]]]]
    touch_target_lookup: Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, Dict[float, Dict[str, float | str]]]]]]]
    vix_open_definition: str
    latest_common_history_date: date | None
    refresh_note: str | None
    fill_samples_path: str | None
    quotes_path: str | None
    vix_snapshots_path: str | None
    vertical_inputs_available: bool


def _parse_thresholds(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _safe_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    cwd_resolved = (Path.cwd() / candidate).resolve()
    if cwd_resolved.exists():
        return str(cwd_resolved)
    project_resolved = (PROJECT_ROOT / candidate).resolve()
    return str(project_resolved)


def _latest_extracted_chain_inputs() -> tuple[str | None, str | None]:
    artifacts_root = PROJECT_ROOT / "artifacts"
    if not artifacts_root.exists():
        return None, None
    candidates: list[tuple[str, str]] = []
    for chain_dir in sorted(artifacts_root.glob("tradingview_chain_*")):
        quotes_matches = sorted(chain_dir.glob("tradingview_single_option_quotes_*.csv"))
        vix_matches = sorted(chain_dir.glob("tradingview_vix_snapshots_*.csv"))
        if quotes_matches and vix_matches:
            candidates.append((str(quotes_matches[-1]), str(vix_matches[-1])))
    if not candidates:
        return None, None
    return candidates[-1]


def _display_path(path: str) -> str:
    try:
        resolved = Path(path).resolve()
        return str(resolved.relative_to(PROJECT_ROOT))
    except Exception:
        return path


def _maybe_refresh_daily_data(spx_path: str, vix_path: str, now: datetime | None = None) -> str | None:
    spx_path = _resolve_path(spx_path)
    vix_path = _resolve_path(vix_path)
    if download_stooq_spx is None or download_cboe_vix is None or download_yahoo_spx is None or write_csv is None:
        return None
    current_dt = now or datetime.now(ZoneInfo("America/New_York"))
    try:
        spx_bars = load_daily_bars(spx_path)
        vix_bars = load_daily_bars(vix_path)
        latest_common = min(spx_bars[-1].date, vix_bars[-1].date)
    except Exception:
        latest_common = None
    needed_through = required_history_date(now=current_dt)
    if latest_common is not None and latest_common >= needed_through:
        return None
    try:
        spx_rows = download_stooq_spx("2021-01-01", current_dt.date().isoformat())
        if not spx_rows:
            spx_rows = download_yahoo_spx("2021-01-01", current_dt.date().isoformat())
        vix_rows = download_cboe_vix("2021-01-01", current_dt.date().isoformat())
        if not spx_rows or not vix_rows:
            return "Auto-refresh skipped: source data came back empty, so local files were left unchanged."
        write_csv(Path(spx_path), spx_rows, ["date", "open", "high", "low", "close", "volume"])
        write_csv(Path(vix_path), vix_rows, ["date", "open", "high", "low", "close", "volume"])
        return f"Auto-refreshed local SPX/VIX history through {current_dt.date().isoformat()}."
    except Exception as exc:  # pragma: no cover - depends on network at runtime
        return f"Auto-refresh skipped: {exc}"


def create_app_state(
    spx_path: str,
    vix_path: str,
    events_path: str,
    underlying_label: str = "SPX",
    train_end_date: str = "2024-12-31",
    max_lag: int = 5,
    ridge_lambda: float = 1.0,
    pca_variance_ratio: float = 0.95,
    thresholds: Iterable[float] | None = None,
    auto_refresh: bool = True,
    fill_samples_path: str | None = "data/tradingview_fill_samples_template.csv",
    quotes_path: str | None = "data/tradingview_single_option_quotes_template.csv",
    vix_snapshots_path: str | None = "data/tradingview_vix_snapshots_template.csv",
) -> AppState:
    spx_path = _resolve_path(spx_path)
    vix_path = _resolve_path(vix_path)
    events_path = _resolve_path(events_path)
    threshold_list = list(thresholds or [0.0025 + 0.0005 * idx for idx in range(16)])
    refresh_note = _maybe_refresh_daily_data(spx_path, vix_path) if auto_refresh else None
    spx_bars = load_daily_bars(spx_path)
    vix_bars = load_daily_bars(vix_path)
    if len(spx_bars) < 50 or len(vix_bars) < 50:
        raise ValueError(
            "Local SPX/VIX history is too short to train the app. "
            f"spx_path={_display_path(spx_path)} rows={len(spx_bars)}, "
            f"vix_path={_display_path(vix_path)} rows={len(vix_bars)}. "
            "Check the CSVs or rerun scripts/download_market_data.py."
        )
    event_calendar = load_event_calendar(events_path)
    rows = align_and_build(
        spx_bars,
        vix_bars,
        event_calendar,
        max_lag=max_lag,
    )
    if len(rows) < 50:
        raise ValueError(
            "Aligned SPX/VIX feature history is empty or too short. "
            f"spx_path={_display_path(spx_path)}, vix_path={_display_path(vix_path)}, rows={len(rows)}. "
            "This usually means one of the CSVs is stale or malformed."
        )
    range_fit = fit_train_backtest_range_model(
        rows,
        ridge_lambda=ridge_lambda,
        train_end_date=train_end_date,
        pca_variance_ratio=pca_variance_ratio,
    )
    backtest_predictions = predict_backtest_range(range_fit)
    cutoffs = compute_regime_cutoffs(range_fit.train_rows)
    regime_lookup = build_regime_probability_lookup(range_fit, backtest_predictions, cutoffs, threshold_list)
    continuation_lookup = build_continuation_lookup(range_fit, backtest_predictions, threshold_list)
    touch_target_lookup = build_touch_target_lookup(range_fit, backtest_predictions, threshold_list, threshold_list)
    resolved_fill_samples_path = _resolve_path(fill_samples_path) if fill_samples_path else None
    preferred_quotes_path, preferred_vix_snapshots_path = _latest_extracted_chain_inputs()
    use_preferred_chain = (
        quotes_path in {None, "data/tradingview_single_option_quotes_template.csv"}
        and vix_snapshots_path in {None, "data/tradingview_vix_snapshots_template.csv"}
        and preferred_quotes_path is not None
        and preferred_vix_snapshots_path is not None
    )
    resolved_quotes_path = preferred_quotes_path if use_preferred_chain else (_resolve_path(quotes_path) if quotes_path else None)
    resolved_vix_snapshots_path = (
        preferred_vix_snapshots_path
        if use_preferred_chain
        else (_resolve_path(vix_snapshots_path) if vix_snapshots_path else None)
    )
    if use_preferred_chain:
        preferred_note = (
            "Using the latest extracted SPX 0DTE chain snapshot for vertical proxying: "
            f"{_display_path(resolved_quotes_path)}"
        )
        refresh_note = f"{refresh_note} {preferred_note}".strip() if refresh_note else preferred_note
    vertical_inputs_available = bool(
        resolved_fill_samples_path
        and resolved_quotes_path
        and resolved_vix_snapshots_path
        and Path(resolved_fill_samples_path).exists()
        and Path(resolved_quotes_path).exists()
        and Path(resolved_vix_snapshots_path).exists()
    )
    return AppState(
        spx_path=spx_path,
        vix_path=vix_path,
        events_path=events_path,
        underlying_label=underlying_label,
        max_lag=max_lag,
        thresholds=threshold_list,
        event_names=collect_event_names(event_calendar),
        spx_bars=spx_bars,
        vix_bars=vix_bars,
        event_calendar=event_calendar,
        range_fit=range_fit,
        cutoffs=cutoffs,
        regime_lookup=regime_lookup,
        continuation_lookup=continuation_lookup,
        touch_target_lookup=touch_target_lookup,
        vix_open_definition="For this training set, VIX open is the daily session open from Cboe's VIX_History.csv OPEN field, aligned by trade date. It is not a separate 9:30 a.m. spot sample.",
        latest_common_history_date=min(spx_bars[-1].date, vix_bars[-1].date) if spx_bars and vix_bars else None,
        refresh_note=refresh_note,
        fill_samples_path=resolved_fill_samples_path,
        quotes_path=resolved_quotes_path,
        vix_snapshots_path=resolved_vix_snapshots_path,
        vertical_inputs_available=vertical_inputs_available,
    )


def _selected_values(form: Mapping[str, List[str]], key: str) -> List[str]:
    return [value for value in form.get(key, []) if value]


def _render_number(value: float, digits: int = 2, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}"


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _render_auc(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.3f}"


def _score_decision_items(state: AppState, side: str, threshold_rows: List[Dict[str, float]], prediction) -> List[Dict[str, float | str]]:
    context = prediction.regime_context
    current_probability_key = "upside_probability" if side == "upside" else "downside_probability"
    decision_rows: List[Dict[str, float | str]] = []
    for item in threshold_rows:
        threshold = float(item["threshold_return"])
        current_probability = float(item[current_probability_key])
        overall = state.regime_lookup[side]["overall"]["overall"][threshold]
        vix = state.regime_lookup[side]["vix_regime"][context.vix_regime][threshold]
        gap = state.regime_lookup[side]["gap_regime"][context.gap_regime][threshold]
        prev_range = state.regime_lookup[side]["range_regime"][context.range_regime][threshold]
        regime_aucs = [item["auc"] for item in [vix, gap, prev_range] if item["auc"] == item["auc"]]
        support_auc = sum(regime_aucs) / len(regime_aucs) if regime_aucs else float("nan")
        edge = current_probability - float(overall["actual_rate"])
        score = edge * support_auc if support_auc == support_auc else edge
        if edge >= 0.08 and support_auc == support_auc and support_auc >= 0.65:
            label = "focus"
        elif edge >= 0.03 and support_auc == support_auc and support_auc >= 0.60:
            label = "watch"
        else:
            label = "background"
        decision_rows.append(
            {
                "threshold_return": threshold,
                "threshold_pct": float(item["threshold_pct"]),
                "today_probability": current_probability,
                "overall_rate": float(overall["actual_rate"]),
                "edge": edge,
                "support_auc": support_auc,
                "label": label,
                "score": score,
            }
        )
    decision_rows.sort(key=lambda row: (row["score"], row["today_probability"]), reverse=True)
    return decision_rows


def _build_decision_rows(state: AppState, side: str, threshold_rows: List[Dict[str, float]], prediction) -> List[List[str]]:
    decision_rows = _score_decision_items(state, side, threshold_rows, prediction)
    top_rows = []
    for row in decision_rows[:5]:
        label_class = "good" if row["label"] in {"focus", "watch"} else ""
        top_rows.append(
            [
                f"{row['threshold_pct']:.2f}%",
                f"<span class=\"{label_class}\">{html.escape(str(row['label']))}</span>",
                f"{float(row['today_probability']) * 100.0:.1f}%",
                f"{float(row['overall_rate']) * 100.0:.1f}%",
                f"{float(row['edge']) * 100.0:+.1f}%",
                _render_auc(float(row["support_auc"])),
            ]
        )
    return top_rows


def _format_pct(value: float, signed: bool = False) -> str:
    return f"{value * 100.0:+.2f}%" if signed else f"{value * 100.0:.1f}%"


def _default_checkpoint_time(prediction_date_text: str) -> str:
    try:
        prediction_date = date.fromisoformat(prediction_date_text)
    except ValueError:
        return "12:00"
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if prediction_date == now_ny.date():
        return now_ny.strftime("%H:%M")
    return "12:00"


def _touch_consistency_message(
    *,
    touched_side: str,
    touch_price: float,
    current_spot: float,
    high_so_far: float,
    low_so_far: float,
) -> str:
    if touched_side == "upside_touch":
        if high_so_far >= touch_price:
            return "High so far confirms that the selected upside touch has already happened."
        if current_spot >= touch_price:
            return "Current spot is above the selected upside touch level, but high so far does not confirm it yet. Double-check the intraday inputs."
        return "High so far is still below the selected upside touch level, so this remains a planning view rather than a confirmed touch."
    if low_so_far <= touch_price:
        return "Low so far confirms that the selected downside touch has already happened."
    if current_spot <= touch_price:
        return "Current spot is below the selected downside touch level, but low so far does not confirm it yet. Double-check the intraday inputs."
    return "Low so far is still above the selected downside touch level, so this remains a planning view rather than a confirmed touch."


def _round_to_spx_strike(value: float, increment: float = 5.0) -> float:
    return round(value / increment) * increment


def _snap_vertical_pair_to_spx_grid(
    projected_lower: float,
    projected_upper: float,
    *,
    desired_width_points: float,
    increment: float = 5.0,
) -> tuple[float, float]:
    desired_width = max(increment, round(desired_width_points / increment) * increment)
    projected_mid = (projected_lower + projected_upper) / 2.0
    candidate_lowers = {
        _round_to_spx_strike(projected_lower, increment),
        _round_to_spx_strike(projected_mid - desired_width / 2.0, increment),
        _round_to_spx_strike(projected_upper - desired_width, increment),
    }
    best_pair: tuple[float, float] | None = None
    best_score = float("inf")
    for lower in candidate_lowers:
        upper = lower + desired_width
        score = abs(lower - projected_lower) + abs(upper - projected_upper)
        if score < best_score:
            best_score = score
            best_pair = (lower, upper)
    if best_pair is None:
        lower = _round_to_spx_strike(projected_lower, increment)
        return lower, lower + desired_width
    return best_pair


def _describe_spot_vs_strike(current_spot: float, strike: float) -> str:
    diff = strike - current_spot
    if abs(diff) < 1e-9:
        return f"at the strike ({strike:.2f})"
    if diff > 0:
        return f"{abs(diff):.1f} points below the strike ({strike:.2f})"
    return f"{abs(diff):.1f} points above the strike ({strike:.2f})"


def _build_featured_playbook(
    state: AppState,
    side: str,
    threshold_rows: List[Dict[str, float]],
    prediction,
) -> Dict[str, str]:
    decision_items = _score_decision_items(state, side, threshold_rows, prediction)
    chosen = next((item for item in decision_items if item["label"] in {"focus", "watch"}), decision_items[0])
    threshold = float(chosen["threshold_return"])
    continuation = select_continuation_stats(
        state.continuation_lookup,
        side,
        threshold,
        prediction.regime_context,
    )
    points = prediction.open_price * threshold
    touch_price = prediction.open_price * (1.0 + threshold if side == "upside" else 1.0 - threshold)
    avg_close_return = float(continuation.get("avg_close_return", 0.0))
    close_q25 = float(continuation.get("close_return_q25", 0.0))
    close_q75 = float(continuation.get("close_return_q75", 0.0))
    same_side_key = "close_above_open_rate" if side == "upside" else "close_below_open_rate"
    past_touch_key = "close_above_touch_rate" if side == "upside" else "close_below_touch_rate"
    same_side_rate = float(continuation.get(same_side_key, 0.0))
    past_touch_rate = float(continuation.get(past_touch_key, 0.0))
    direction_label = "upside" if side == "upside" else "downside"
    finishing_label = "above the open" if side == "upside" else "below the open"
    past_touch_label = "above the touched level" if side == "upside" else "below the touched level"
    return {
        "eyebrow": f"Featured {direction_label} setup",
        "title": f"Watch {chosen['threshold_pct']:.2f}% ({points:.1f} pts, {touch_price:.2f})",
        "summary": (
            f"Today's model puts this touch at {float(chosen['today_probability']) * 100.0:.1f}% versus a "
            f"{float(chosen['overall_rate']) * 100.0:.1f}% background rate."
        ),
        "detail": (
            f"If that level is reached on similar days, the close has usually finished around "
            f"{_format_pct(avg_close_return, signed=True)} with a middle band from "
            f"{_format_pct(close_q25, signed=True)} to {_format_pct(close_q75, signed=True)}. "
            f"Historically, {same_side_rate * 100.0:.1f}% still finished {finishing_label}, and "
            f"{past_touch_rate * 100.0:.1f}% finished {past_touch_label}."
        ),
        "basis": str(continuation.get("basis_label", "overall")),
        "label": str(chosen["label"]),
    }


def _render_collapsible(title: str, intro: str, content: str, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return f"""
    <details class="collapsible"{open_attr}>
      <summary>{html.escape(title)}</summary>
      <p class="hint">{html.escape(intro)}</p>
      {content}
    </details>
    """


def _html_page(title: str, body: str) -> bytes:
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f0e7;
      --paper: #fffaf3;
      --ink: #1f2430;
      --muted: #6d665c;
      --accent: #b3561d;
      --accent-soft: #f2dcc8;
      --border: #dbcbb8;
      --good: #1c6b4a;
      --bad: #8d2f23;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(179, 86, 29, 0.16), transparent 28%),
        linear-gradient(180deg, #f9f3eb 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 2rem; }}
    h2 {{ font-size: 1.25rem; margin-top: 24px; }}
    p, li {{ color: var(--muted); line-height: 1.45; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(300px, 340px) minmax(0, 700px);
      gap: 20px;
      align-items: start;
      justify-content: center;
    }}
    .card {{
      background: rgba(255, 250, 243, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 24px rgba(56, 35, 18, 0.08);
    }}
    form .field {{
      margin-bottom: 14px;
    }}
    label {{
      display: block;
      font-size: 0.92rem;
      color: var(--ink);
      margin-bottom: 5px;
    }}
    input[type="text"], input[type="number"], input[type="date"], select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .subgrid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .checkbox-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .checkbox-list label {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 0;
      color: var(--muted);
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      background: linear-gradient(135deg, #c05f24, var(--accent));
      color: white;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(179, 86, 29, 0.24);
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .decision-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 12px 0 20px;
    }}
    .metric {{
      background: var(--accent-soft);
      border-radius: 14px;
      padding: 14px;
    }}
    .spotlight-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 18px 0 22px;
    }}
    .spotlight {{
      background: linear-gradient(180deg, rgba(242, 220, 200, 0.96), rgba(255, 250, 243, 0.96));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
    }}
    .eyebrow {{
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .spotlight h3 {{
      margin-bottom: 8px;
    }}
    .spotlight p {{
      margin: 8px 0;
    }}
    .callout {{
      background: rgba(255, 250, 243, 0.94);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin: 14px 0 18px;
    }}
    .decision-card {{
      background: var(--accent-soft);
      border-radius: 14px;
      padding: 14px;
    }}
    .metric strong {{
      display: block;
      font-size: 1.2rem;
      color: var(--ink);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f0e5d8;
      color: var(--ink);
      font-size: 0.88rem;
      margin: 0 8px 8px 0;
    }}
    .pill.good {{
      background: rgba(28, 107, 74, 0.12);
    }}
    .pill.focus {{
      background: rgba(28, 107, 74, 0.12);
      color: var(--good);
    }}
    .pill.watch {{
      background: rgba(179, 86, 29, 0.14);
    }}
    .collapsible {{
      margin: 14px 0;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(255, 250, 243, 0.75);
      overflow: hidden;
    }}
    .collapsible summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-weight: 700;
      color: var(--ink);
      background: rgba(242, 220, 200, 0.5);
    }}
    .collapsible summary::-webkit-details-marker {{
      display: none;
    }}
    .collapsible > *:not(summary) {{
      padding-left: 16px;
      padding-right: 16px;
    }}
    .collapsible .table-wrap {{
      padding-bottom: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.89rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid #eadfd2;
      vertical-align: top;
    }}
    thead th {{
      color: var(--ink);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .hint {{ font-size: 0.88rem; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    @media (min-width: 1280px) {{
      main {{
        max-width: 1120px;
      }}
    }}
    @media (max-width: 980px) {{
      .grid, .summary, .decision-grid, .spotlight-grid {{ grid-template-columns: 1fr; }}
      .checkbox-list, .subgrid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>"""
    return page.encode("utf-8")


def _build_probability_rows(
    state: AppState,
    side: str,
    threshold_rows: List[Dict[str, float]],
    prediction,
) -> List[List[str]]:
    current_probability_key = "upside_probability" if side == "upside" else "downside_probability"
    rows: List[List[str]] = []
    context = prediction.regime_context
    combo_key = f"{context.vix_regime}|{context.range_regime}|{context.gap_regime}"
    for item in threshold_rows:
        threshold = float(item["threshold_return"])
        overall = state.regime_lookup[side]["overall"]["overall"][threshold]
        vix = state.regime_lookup[side]["vix_regime"][context.vix_regime][threshold]
        gap = state.regime_lookup[side]["gap_regime"][context.gap_regime][threshold]
        prev_range = state.regime_lookup[side]["range_regime"][context.range_regime][threshold]
        combo_rows = [
            state.regime_lookup[side]["vix_regime"][context.vix_regime][threshold],
            state.regime_lookup[side]["gap_regime"][context.gap_regime][threshold],
            state.regime_lookup[side]["range_regime"][context.range_regime][threshold],
        ]
        combo_rate = sum(float(row["actual_rate"]) for row in combo_rows) / len(combo_rows)
        combo_samples = min(float(row["samples"]) for row in combo_rows)
        current_probability = float(item[current_probability_key]) * 100.0
        points = prediction.open_price * threshold
        touch_price = prediction.open_price * (1.0 + threshold if side == "upside" else 1.0 - threshold)
        rows.append(
            [
                f"{item['threshold_pct']:.2f}%",
                f"{points:.1f}",
                f"{touch_price:.2f}",
                f"{current_probability:.1f}%",
                f"{overall['actual_rate'] * 100.0:.1f}%",
                f"{combo_rate * 100.0:.1f}%",
                f"{vix['actual_rate'] * 100.0:.1f}%",
                f"{gap['actual_rate'] * 100.0:.1f}%",
                f"{prev_range['actual_rate'] * 100.0:.1f}%",
                combo_key,
                f"{combo_samples:.0f}",
            ]
        )
    return rows


def _build_continuation_rows(
    state: AppState,
    side: str,
    threshold_rows: List[Dict[str, float]],
    prediction,
) -> List[List[str]]:
    rows: List[List[str]] = []
    close_on_side_key = "close_above_open_rate" if side == "upside" else "close_below_open_rate"
    close_past_touch_key = "close_above_touch_rate" if side == "upside" else "close_below_touch_rate"
    for item in threshold_rows:
        threshold = float(item["threshold_return"])
        continuation = select_continuation_stats(
            state.continuation_lookup,
            side,
            threshold,
            prediction.regime_context,
        )
        threshold_points = prediction.open_price * threshold
        touch_price = prediction.open_price * (1.0 + threshold if side == "upside" else 1.0 - threshold)
        avg_close_return = float(continuation.get("avg_close_return", 0.0))
        close_return_q25 = float(continuation.get("close_return_q25", 0.0))
        close_return_q75 = float(continuation.get("close_return_q75", 0.0))
        avg_close_points = prediction.open_price * avg_close_return
        avg_close_price = prediction.open_price * (1.0 + avg_close_return)
        band_low_price = prediction.open_price * (1.0 + close_return_q25)
        band_high_price = prediction.open_price * (1.0 + close_return_q75)
        rows.append(
            [
                f"{item['threshold_pct']:.2f}%",
                f"{threshold_points:.1f}",
                f"{touch_price:.2f}",
                str(continuation.get("basis_label", "overall")),
                f"{float(continuation.get('samples', 0.0)):.0f}",
                f"{avg_close_return * 100.0:+.2f}% / {avg_close_points:+.1f}",
                f"{avg_close_price:.2f}",
                f"{close_return_q25 * 100.0:+.2f}% to {close_return_q75 * 100.0:+.2f}%",
                f"{band_low_price:.2f} to {band_high_price:.2f}",
                f"{float(continuation.get(close_on_side_key, 0.0)) * 100.0:.1f}%",
                f"{float(continuation.get(close_past_touch_key, 0.0)) * 100.0:.1f}%",
            ]
        )
    return rows


def _filter_vertical_estimates_for_current_spot(scored_input, *, current_spot: float, max_snapshot_gap_ratio: float = 0.05):
    if getattr(scored_input, "empty", True):
        return scored_input
    distance_ratio = (scored_input["underlying_price"].astype(float) - current_spot).abs() / max(current_spot, 1e-9)
    return scored_input.loc[distance_ratio <= max_snapshot_gap_ratio].copy()


def _reanchor_vertical_estimates_to_spot(vertical_estimates, *, current_spot: float):
    if getattr(vertical_estimates, "empty", True):
        return vertical_estimates
    translated = vertical_estimates.copy()
    new_rows = []
    for _, row in translated.iterrows():
        base_underlying = float(row["underlying_price"])
        if base_underlying <= 0:
            continue
        lower_offset = (float(row["lower_strike"]) - base_underlying) / base_underlying
        upper_offset = (float(row["upper_strike"]) - base_underlying) / base_underlying
        projected_lower = current_spot * (1.0 + lower_offset)
        projected_upper = current_spot * (1.0 + upper_offset)
        desired_width_points = float(row.get("width_points", abs(float(row["upper_strike"]) - float(row["lower_strike"]))))
        lower_strike, upper_strike = _snap_vertical_pair_to_spx_grid(
            projected_lower,
            projected_upper,
            desired_width_points=desired_width_points,
        )
        width_points = abs(upper_strike - lower_strike)
        strategy = str(row["strategy"])
        if strategy in {"bull_call_debit", "bear_call_credit"}:
            short_distance_points = upper_strike - current_spot
        else:
            short_distance_points = current_spot - upper_strike
        updated = dict(row)
        updated["snapshot_id"] = f"{row['snapshot_id']}_reanchored"
        updated["underlying_price"] = current_spot
        updated["lower_strike"] = lower_strike
        updated["upper_strike"] = upper_strike
        updated["width_points"] = width_points
        updated["short_distance_points"] = short_distance_points
        updated["short_distance_pct_spot"] = short_distance_points / max(current_spot, 1e-9)
        updated["quote_proxy_mode"] = "reanchored_percent_distance"
        new_rows.append(updated)
    return pd.DataFrame(new_rows)


def _choose_best_vertical(scored_rows, keyword: str):
    filtered = [row for _, row in scored_rows.iterrows() if keyword in str(row["outlook"])]
    if not filtered:
        return None
    filtered.sort(key=lambda row: (float(row["profit_to_cost_ratio_proxy"]), float(row["expected_value"])), reverse=True)
    return filtered[0]


def _choose_best_overall_vertical(scored_rows):
    if getattr(scored_rows, "empty", True):
        return None
    filtered = [row for _, row in scored_rows.iterrows() if float(row["predicted_profit_proxy"]) > 0]
    if not filtered:
        filtered = [row for _, row in scored_rows.iterrows()]
    filtered.sort(key=lambda row: (float(row["profit_to_cost_ratio_proxy"]), float(row["expected_value"])), reverse=True)
    return filtered[0] if filtered else None


def _value_bucket(
    row,
    *,
    strong_profit_threshold: float,
    strong_ratio_threshold: float,
    watch_profit_threshold: float,
    watch_ratio_threshold: float,
) -> tuple[str, str]:
    predicted_profit = float(row["predicted_profit_proxy"])
    ratio = float(row["profit_to_cost_ratio_proxy"])
    if predicted_profit <= 0.0:
        return ("pass", "Modeled value is not there after execution cost.")
    if predicted_profit >= strong_profit_threshold and ratio >= strong_ratio_threshold:
        return ("strong value", "This is the kind of setup where the modeled payout is meaningfully ahead of cost.")
    if predicted_profit >= watch_profit_threshold and ratio >= watch_ratio_threshold:
        return ("watch", "There is modeled value here, but the edge is thinner and execution matters more.")
    return ("thin", "The setup is positive on paper, but only modestly so.")


def _render_value_breakpoints(
    row,
    *,
    strong_profit_threshold: float,
    strong_ratio_threshold: float,
    watch_profit_threshold: float,
    watch_ratio_threshold: float,
) -> str:
    label, summary = _value_bucket(
        row,
        strong_profit_threshold=strong_profit_threshold,
        strong_ratio_threshold=strong_ratio_threshold,
        watch_profit_threshold=watch_profit_threshold,
        watch_ratio_threshold=watch_ratio_threshold,
    )
    predicted_profit = float(row["predicted_profit_proxy"])
    ratio = float(row["profit_to_cost_ratio_proxy"])
    return f"""
    <div class="spotlight">
      <span class="eyebrow">Value breakpoints</span>
      <h3>{html.escape(label.title())}</h3>
      <p>{html.escape(summary)}</p>
      <p class="hint">Current proxy profit: <strong>{predicted_profit:+.2f}</strong> points. Current profit / risk: <strong>{ratio:+.2f}x</strong>.</p>
      <p class="hint">
        Strong value: at least <strong>+{strong_profit_threshold:.2f}</strong> points and <strong>{strong_ratio_threshold:.2f}x</strong> profit / risk.<br>
        Watch: at least <strong>+{watch_profit_threshold:.2f}</strong> points and <strong>{watch_ratio_threshold:.2f}x</strong> profit / risk.<br>
        Thin: positive, but below the watch threshold.<br>
        Pass: zero or negative modeled value after entry cost.
      </p>
    </div>
    """


def _render_vertical_card(row, *, title: str, open_price: float, current_spot: float | None = None, checkpoint_time_text: str | None = None) -> str:
    lower_strike = float(row["lower_strike"])
    upper_strike = float(row["upper_strike"])
    lower_from_open_pct = ((lower_strike - open_price) / open_price) * 100.0
    upper_from_open_pct = ((upper_strike - open_price) / open_price) * 100.0
    spot_note = ""
    if current_spot is not None:
        time_prefix = f"At {checkpoint_time_text}, " if checkpoint_time_text else ""
        spot_note = (
            f'<p class="hint">{html.escape(time_prefix)}spot is <strong>{current_spot:.2f}</strong>. '
            f'It is <strong>{html.escape(_describe_spot_vs_strike(current_spot, lower_strike))}</strong> and '
            f'<strong>{html.escape(_describe_spot_vs_strike(current_spot, upper_strike))}</strong>.</p>'
        )
    return f"""
    <div class="spotlight">
      <span class="eyebrow">{html.escape(title)}</span>
      <h3>{html.escape(str(row["strategy"]).replace("_", " "))}</h3>
      <p><span class="pill">{html.escape(str(row["outlook"]))}</span> <span class="pill">{html.escape(str(row["near_continuation_basis"]))}</span></p>
      <p>
        Entry is modeled around <strong>{float(row["entry_price"]):.2f}</strong> with a proxy terminal value of
        <strong>{float(row["predicted_terminal_value_proxy"]):.2f}</strong>.
      </p>
      <p class="hint">
        Modeled profit is <strong>{float(row["predicted_profit_proxy"]):+.2f}</strong>, which is
        <strong>{float(row["profit_to_cost_ratio_proxy"]):+.2f}x</strong> the risk capital in this approximation.
      </p>
      <p class="hint">
        Strike band: <strong>{lower_strike:.2f} / {upper_strike:.2f}</strong>.
        From the open that is <strong>{lower_from_open_pct:+.2f}% / {upper_from_open_pct:+.2f}%</strong>.
      </p>
      {spot_note}
      <p class="hint">
        Conditional close odds from the touched state: near <strong>{float(row.get("near_close_probability_given_touch", 0.0)) * 100.0:.1f}%</strong>,
        far <strong>{float(row.get("far_close_probability_given_touch", 0.0)) * 100.0:.1f}%</strong>.
      </p>
    </div>
    """


def _build_vertical_rows(scored_rows: object) -> List[List[str]]:
    rows: List[List[str]] = []
    if getattr(scored_rows, "empty", True):
        return rows
    for _, row in scored_rows.head(8).iterrows():
        rows.append(
            [
                html.escape(str(row["strategy"]).replace("_", " ")),
                html.escape(str(row["outlook"])),
                f"{float(row['lower_strike']):.2f} / {float(row['upper_strike']):.2f}",
                f"{float(row['entry_price']):.2f}",
                f"{float(row['predicted_terminal_value_proxy']):.2f}",
                f"{float(row['predicted_profit_proxy']):+.2f}",
                f"{float(row['profit_to_cost_ratio_proxy']):+.2f}x",
                html.escape(str(row["near_continuation_basis"])),
            ]
        )
    return rows


def render_app(state: AppState, form: Mapping[str, List[str]] | None = None) -> bytes:
    form = form or {}
    suggested_trade_date = default_prediction_date(state.spx_bars, state.vix_bars)
    default_date = suggested_trade_date.isoformat()
    latest_spx = state.spx_bars[-1]
    latest_vix = state.vix_bars[-1]
    prediction_date_text = form.get("prediction_date", [default_date])[0]
    spx_open_text = form.get("spx_open", [f"{latest_spx.close:.2f}"])[0]
    vix_open_text = form.get("vix_open", [f"{latest_vix.open:.2f}" if latest_vix.open else f"{latest_vix.close:.2f}"])[0]
    current_spot_text = form.get("current_spx_spot", [spx_open_text])[0]
    high_so_far_text = form.get("high_so_far", [current_spot_text])[0]
    low_so_far_text = form.get("low_so_far", [current_spot_text])[0]
    checkpoint_time_text = form.get("checkpoint_time", [_default_checkpoint_time(prediction_date_text)])[0]
    touched_side_text = form.get("touched_side", ["upside_touch"])[0]
    touched_threshold_text = form.get("touched_threshold", ["0.50"])[0]
    vertical_width_text = form.get("vertical_width_points", ["10"])[0]
    strong_profit_threshold_text = form.get("strong_profit_threshold", ["1.50"])[0]
    strong_ratio_threshold_text = form.get("strong_ratio_threshold", ["0.75"])[0]
    watch_profit_threshold_text = form.get("watch_profit_threshold", ["0.75"])[0]
    watch_ratio_threshold_text = form.get("watch_ratio_threshold", ["0.35"])[0]
    selected_events = _selected_values(form, "event_type")

    prediction_markup = ""
    error_markup = ""
    try:
        prediction_date = date.fromisoformat(prediction_date_text)
        spx_open = float(spx_open_text)
        vix_open = float(vix_open_text)
        current_spot = float(current_spot_text)
        high_so_far = float(high_so_far_text)
        low_so_far = float(low_so_far_text)
        if high_so_far < low_so_far:
            raise ValueError("High so far must be greater than or equal to low so far.")
        overrides = PriorDayOverrides(
            spx_open=_safe_float(form.get("prior_spx_open", [""])[0]),
            spx_high=_safe_float(form.get("prior_spx_high", [""])[0]),
            spx_low=_safe_float(form.get("prior_spx_low", [""])[0]),
            spx_close=_safe_float(form.get("prior_spx_close", [""])[0]),
            vix_open=_safe_float(form.get("prior_vix_open", [""])[0]),
            vix_high=_safe_float(form.get("prior_vix_high", [""])[0]),
            vix_low=_safe_float(form.get("prior_vix_low", [""])[0]),
            vix_close=_safe_float(form.get("prior_vix_close", [""])[0]),
        )
        live_row = build_live_feature_row(
            state.spx_bars,
            state.vix_bars,
            state.event_calendar,
            prediction_date=prediction_date,
            current_spx_open=spx_open,
            current_vix_open=vix_open,
            max_lag=state.max_lag,
            selected_events=selected_events,
            prior_day_overrides=overrides,
        )
        prediction = predict_live_excursions(state.range_fit, live_row, state.cutoffs)
        threshold_rows = threshold_probabilities(prediction, state.thresholds)
        upside_rows = _build_probability_rows(state, "upside", threshold_rows, prediction)
        downside_rows = _build_probability_rows(state, "downside", threshold_rows, prediction)
        upside_continuation_rows = _build_continuation_rows(state, "upside", threshold_rows, prediction)
        downside_continuation_rows = _build_continuation_rows(state, "downside", threshold_rows, prediction)
        upside_decisions = _build_decision_rows(state, "upside", threshold_rows, prediction)
        downside_decisions = _build_decision_rows(state, "downside", threshold_rows, prediction)
        featured_upside = _build_featured_playbook(state, "upside", threshold_rows, prediction)
        featured_downside = _build_featured_playbook(state, "downside", threshold_rows, prediction)
        selected_touch_side = touched_side_text if touched_side_text in {"upside_touch", "downside_touch"} else "upside_touch"
        touched_threshold_return = float(touched_threshold_text) / 100.0
        vertical_width_points = float(vertical_width_text)
        strong_profit_threshold = float(strong_profit_threshold_text)
        strong_ratio_threshold = float(strong_ratio_threshold_text)
        watch_profit_threshold = float(watch_profit_threshold_text)
        watch_ratio_threshold = float(watch_ratio_threshold_text)
        current_move_return = (current_spot - spx_open) / spx_open if spx_open else 0.0
        high_move_return = (high_so_far - spx_open) / spx_open if spx_open else 0.0
        low_move_return = (low_so_far - spx_open) / spx_open if spx_open else 0.0
        touch_price = spx_open * (1.0 + touched_threshold_return if selected_touch_side == "upside_touch" else 1.0 - touched_threshold_return)
        touch_consistency_label = _touch_consistency_message(
            touched_side=selected_touch_side,
            touch_price=touch_price,
            current_spot=current_spot,
            high_so_far=high_so_far,
            low_so_far=low_so_far,
        )
        vertical_markup = ""
        if state.vertical_inputs_available:
            fill_samples, quote_snapshots, vix_by_snapshot = load_vertical_fill_inputs(
                state.fill_samples_path,
                state.quotes_path,
                state.vix_snapshots_path,
            )
            vertical_estimates = estimate_vertical_fills_from_quotes(
                fill_samples=fill_samples,
                quote_snapshots=quote_snapshots,
                vix_by_snapshot=vix_by_snapshot,
                width_points=vertical_width_points,
            )
            filtered_vertical_estimates = _filter_vertical_estimates_for_current_spot(
                vertical_estimates,
                current_spot=current_spot,
            )
            vertical_proxy_note = ""
            candidate_vertical_estimates = filtered_vertical_estimates
            if filtered_vertical_estimates.empty and not vertical_estimates.empty:
                candidate_vertical_estimates = _reanchor_vertical_estimates_to_spot(
                    vertical_estimates,
                    current_spot=current_spot,
                )
                vertical_proxy_note = (
                    "<p class=\"hint\">No quote snapshot was close enough to current SPX spot, so the app fell back to a "
                    "<strong>re-anchored stale-chain proxy</strong>: it preserved the sample chain’s percentage strike distances "
                    "from its own underlying and mapped them onto today’s current SPX spot. Entry prices are still only a proxy.</p>"
                )
            post_touch_verticals = score_vertical_opportunities_after_touch(
                prediction=prediction,
                touch_target_lookup=state.touch_target_lookup,
                vertical_estimates=candidate_vertical_estimates,
                touched_side=selected_touch_side,
                touch_threshold_return=touched_threshold_return,
            )
            best_overall = _choose_best_overall_vertical(post_touch_verticals)
            best_continuation = _choose_best_vertical(post_touch_verticals, "continuation")
            best_fade = _choose_best_vertical(post_touch_verticals, "fade")
            vertical_cards = ""
            if best_overall is not None:
                vertical_cards += _render_vertical_card(
                    best_overall,
                    title="Vertical strategy to watch",
                    open_price=prediction.open_price,
                    current_spot=current_spot,
                    checkpoint_time_text=checkpoint_time_text,
                )
                vertical_cards += _render_value_breakpoints(
                    best_overall,
                    strong_profit_threshold=strong_profit_threshold,
                    strong_ratio_threshold=strong_ratio_threshold,
                    watch_profit_threshold=watch_profit_threshold,
                    watch_ratio_threshold=watch_ratio_threshold,
                )
            elif best_continuation is not None:
                vertical_cards += _render_vertical_card(
                    best_continuation,
                    title="Vertical strategy to watch",
                    open_price=prediction.open_price,
                    current_spot=current_spot,
                    checkpoint_time_text=checkpoint_time_text,
                )
                vertical_cards += _render_value_breakpoints(
                    best_continuation,
                    strong_profit_threshold=strong_profit_threshold,
                    strong_ratio_threshold=strong_ratio_threshold,
                    watch_profit_threshold=watch_profit_threshold,
                    watch_ratio_threshold=watch_ratio_threshold,
                )
            if not vertical_cards:
                vertical_cards = '<div class="callout"><p class="hint">No verticals matched this touch setup with the current quote snapshot and width.</p></div>'
            vertical_rows = _build_vertical_rows(post_touch_verticals)
            vertical_table = (
                '<div class="table-wrap">'
                + _render_table(
                    ["Strategy", "Playbook", "Strikes", "Entry", "Proxy terminal", "Proxy profit", "Profit / risk", "Basis"],
                    vertical_rows,
                )
                + "</div>"
            ) if vertical_rows else '<p class="hint">No vertical candidates were generated for this width and quote snapshot.</p>'
            touch_label = "upside" if selected_touch_side == "upside_touch" else "downside"
            filtered_count = len(vertical_estimates) - len(filtered_vertical_estimates)
            snapshot_warning = ""
            if filtered_count > 0:
                snapshot_warning = (
                    f'<p class="hint">Filtered out <strong>{filtered_count}</strong> vertical candidates because their quote snapshot underlying was more than 5% away from the current SPX spot. '
                    'This keeps stale sample chains from looking tradeable when they are clearly disconnected from today’s market.</p>'
                )
            vertical_markup = f"""
            <h3>Vertical strategy to watch</h3>
            <p class="hint">
              This section assumes the day has already touched <strong>{float(touched_threshold_text):.2f}% {html.escape(touch_label)}</strong> from the open.
              It uses the touch-conditioned close model plus the sampled quote-fill proxy to find the setup with the best remaining value, then shows where that setup lands on simple value breakpoints.
            </p>
            {snapshot_warning}
            {vertical_proxy_note}
            <div class="callout">
              <strong>Observed intraday state</strong>
              <p>
                At <strong>{html.escape(checkpoint_time_text)}</strong>, spot is <strong>{current_spot:.2f}</strong>,
                which is <strong>{current_move_return * 100.0:+.2f}% ({current_spot - spx_open:+.1f} pts)</strong> from the open.
                The selected touch price is <strong>{touch_price:.2f}</strong>.
              </p>
              <p class="hint">
                High so far: <strong>{high_so_far:.2f}</strong> ({high_move_return * 100.0:+.2f}%).
                Low so far: <strong>{low_so_far:.2f}</strong> ({low_move_return * 100.0:+.2f}%).
              </p>
              <p class="hint">{html.escape(touch_consistency_label)}</p>
            </div>
            <div class="spotlight-grid">
              {vertical_cards}
            </div>
            {_render_collapsible(
                "Post-touch vertical details",
                "Use this when you want the ranked list behind the watch card. Proxy terminal value is capped by the spread width in the modeled structure.",
                vertical_table,
                open_by_default=False,
            )}
            """
        else:
            vertical_markup = """
            <h3>Vertical strategy to watch</h3>
            <p class="hint">
              Quote-driven vertical analysis is off because the fill-sample and quote snapshot files were not available at startup.
            </p>
            """
        context = prediction.regime_context
        lead_sentence = (
            f"Today screens as {context.vix_regime.replace('_', ' ')}, "
            f"{context.gap_regime.replace('_', ' ')}, and "
            f"{context.range_regime.replace('_', ' ')}. "
            "Use the featured cards first, then expand the tables only if you want the full historical context."
        )
        prediction_markup = f"""
        <div class="card">
          <h2>Today's Excursion Forecast</h2>
          <div class="summary">
            <div class="metric"><span class="hint">Predicted high from open</span><strong>{prediction.predicted_high_return * 100.0:.2f}%</strong><span>{prediction.predicted_high_price:.2f}</span></div>
            <div class="metric"><span class="hint">Predicted low from open</span><strong>{prediction.predicted_low_return * 100.0:.2f}%</strong><span>{prediction.predicted_low_price:.2f}</span></div>
            <div class="metric"><span class="hint">Overnight gap</span><strong>{context.overnight_gap * 100.0:.2f}%</strong><span>prev close {prediction.prev_close:.2f}</span></div>
          </div>
          <p>
            <span class="pill">Weekday: {html.escape(context.weekday)}</span>
            <span class="pill">VIX regime: {html.escape(context.vix_regime)}</span>
            <span class="pill">Prior-day range regime: {html.escape(context.range_regime)}</span>
            <span class="pill">Gap regime: {html.escape(context.gap_regime)}</span>
          </p>
          <div class="callout">
            <strong>How to read this page</strong>
            <p>{html.escape(lead_sentence)}</p>
            <p class="hint">
              `Today p` is the live touch estimate. `Overall hit` is the unconditional historical rate. `Blended regime hit` is the average of the matching VIX, gap, and prior-range buckets. In the continuation view, `Close on side` asks whether the day still finished above or below the open, while `Close past touch` asks whether it finished beyond the touched level itself.
            </p>
          </div>
          <h3>Decision Summary</h3>
          <p class="hint">
            `focus` means today's probability is meaningfully above background and the matching regimes had useful ranking quality. `watch` is weaker but still worth monitoring. The top cards below keep the key idea visible without requiring you to read the full tables.
          </p>
          <div class="decision-grid">
            <div class="decision-card">
              <span class="hint">Top upside thresholds</span>
              <div class="table-wrap">
                {_render_table(["Threshold", "Label", "Today p", "Base rate", "Edge", "Regime AUC"], upside_decisions)}
              </div>
            </div>
            <div class="decision-card">
              <span class="hint">Top downside thresholds</span>
              <div class="table-wrap">
                {_render_table(["Threshold", "Label", "Today p", "Base rate", "Edge", "Regime AUC"], downside_decisions)}
              </div>
            </div>
          </div>
          <div class="spotlight-grid">
            <div class="spotlight">
              <span class="eyebrow">{html.escape(featured_upside["eyebrow"])}</span>
              <h3>{html.escape(featured_upside["title"])}</h3>
              <p><span class="pill {html.escape(featured_upside['label'])}">{html.escape(featured_upside['label'])}</span> <span class="pill">{html.escape(featured_upside['basis'])}</span></p>
              <p>{html.escape(featured_upside["summary"])}</p>
              <p class="hint">{html.escape(featured_upside["detail"])}</p>
            </div>
            <div class="spotlight">
              <span class="eyebrow">{html.escape(featured_downside["eyebrow"])}</span>
              <h3>{html.escape(featured_downside["title"])}</h3>
              <p><span class="pill {html.escape(featured_downside['label'])}">{html.escape(featured_downside['label'])}</span> <span class="pill">{html.escape(featured_downside['basis'])}</span></p>
              <p>{html.escape(featured_downside["summary"])}</p>
              <p class="hint">{html.escape(featured_downside["detail"])}</p>
            </div>
          </div>
          {_render_collapsible(
              "Upside Touch Levels",
              "Use this when you want the full upside threshold ladder in percent, SPX points, and price. The regime columns are context benchmarks, not live predictions.",
              '<div class="table-wrap">' + _render_table(
                  ["Threshold", "Points", "Touch price", "Today p", "Overall hit", "Blended regime hit", context.vix_regime, context.gap_regime, context.range_regime, "Regime key", "Regime samples"],
                  upside_rows,
              ) + '</div>',
              open_by_default=False,
          )}
          {_render_collapsible(
              "Upside Continuation If Touched",
              "This asks what usually happened by the close after the upside threshold was reached. Read `Close band` as the middle historical range for the close relative to the open.",
              '<div class="table-wrap">' + _render_table(
                  ["Threshold", "Points", "Touch price", "Basis", "Samples", "Avg close", "Avg close px", "Close band", "Close band px", "Close on side", "Close past touch"],
                  upside_continuation_rows,
              ) + '</div>',
              open_by_default=False,
          )}
          {_render_collapsible(
              "Downside Touch Levels",
              "Use this when you want the full downside threshold ladder. Large gaps or high VIX can raise the base rate even when the path becomes less clean.",
              '<div class="table-wrap">' + _render_table(
                  ["Threshold", "Points", "Touch price", "Today p", "Overall hit", "Blended regime hit", context.vix_regime, context.gap_regime, context.range_regime, "Regime key", "Regime samples"],
                  downside_rows,
              ) + '</div>',
              open_by_default=False,
          )}
          {_render_collapsible(
              "Downside Continuation If Touched",
              "This asks where the close usually landed after the downside threshold was hit. `Close on side` means the day still closed below the open.",
              '<div class="table-wrap">' + _render_table(
                  ["Threshold", "Points", "Touch price", "Basis", "Samples", "Avg close", "Avg close px", "Close band", "Close band px", "Close on side", "Close past touch"],
                  downside_continuation_rows,
              ) + '</div>',
              open_by_default=False,
          )}
          {vertical_markup}
        </div>
        """
    except Exception as exc:  # pragma: no cover - exercised in route tests
        error_markup = f'<div class="card"><p class="bad"><strong>Input error:</strong> {html.escape(str(exc))}</p></div>'

    event_markup = "".join(
        f'<label><input type="checkbox" name="event_type" value="{html.escape(name)}"'
        f' {"checked" if name in selected_events else ""}> {html.escape(name)}</label>'
        for name in state.event_names
    )

    body = f"""
      <h1>{html.escape(state.underlying_label)} Excursion Probability App</h1>
      <p>
        This local app uses the trained daily range model plus the saved historical regime tables. We only ask for what is knowable at the open: today's date, {html.escape(state.underlying_label)} regular-session open, VIX daily session open, and optional event flags. Lagged context comes from the local daily history files, with optional prior-day overrides if your CSVs are behind.
      </p>
      <p class="hint">
        {html.escape(state.vix_open_definition)}
      </p>
      <p class="hint">
        Suggested next trade date: <strong>{suggested_trade_date.isoformat()}</strong>.
        Latest local history date: <strong>{state.latest_common_history_date.isoformat() if state.latest_common_history_date else 'n/a'}</strong>.
      </p>
      {f'<p class="hint">{html.escape(state.refresh_note)}</p>' if state.refresh_note else ''}
      <div class="grid">
        <div class="card">
          <h2>Inputs</h2>
          <form method="post">
            <div class="field">
              <label for="prediction_date">Suggested trade date</label>
              <input id="prediction_date" name="prediction_date" type="date" value="{html.escape(prediction_date_text)}">
            </div>
            <div class="field">
              <label for="spx_open">{html.escape(state.underlying_label)} regular-session open (9:30 ET)</label>
              <input id="spx_open" name="spx_open" type="number" step="0.01" value="{html.escape(spx_open_text)}">
            </div>
            <div class="field">
              <label for="vix_open">VIX daily session open</label>
              <input id="vix_open" name="vix_open" type="number" step="0.01" value="{html.escape(vix_open_text)}">
            </div>
            <div class="subgrid">
              <div class="field">
                <label for="current_spx_spot">Current {html.escape(state.underlying_label)} spot</label>
                <input id="current_spx_spot" name="current_spx_spot" type="number" step="0.01" value="{html.escape(current_spot_text)}">
              </div>
              <div class="field">
                <label for="checkpoint_time">Current time of day (ET)</label>
                <input id="checkpoint_time" name="checkpoint_time" type="text" value="{html.escape(checkpoint_time_text)}" placeholder="12:00">
              </div>
            </div>
            <div class="subgrid">
              <div class="field">
                <label for="high_so_far">High so far</label>
                <input id="high_so_far" name="high_so_far" type="number" step="0.01" value="{html.escape(high_so_far_text)}">
              </div>
              <div class="field">
                <label for="low_so_far">Low so far</label>
                <input id="low_so_far" name="low_so_far" type="number" step="0.01" value="{html.escape(low_so_far_text)}">
              </div>
            </div>
            <div class="subgrid">
              <div class="field">
                <label for="touched_side">Touched side to analyze</label>
                <select id="touched_side" name="touched_side">
                  <option value="upside_touch" {"selected" if touched_side_text == "upside_touch" else ""}>Upside touch</option>
                  <option value="downside_touch" {"selected" if touched_side_text == "downside_touch" else ""}>Downside touch</option>
                </select>
              </div>
              <div class="field">
                <label for="touched_threshold">Touched threshold (%)</label>
                <input id="touched_threshold" name="touched_threshold" type="number" step="0.05" value="{html.escape(touched_threshold_text)}">
              </div>
            </div>
            <div class="field">
              <label for="vertical_width_points">Vertical width for quote proxy (points)</label>
              <input id="vertical_width_points" name="vertical_width_points" type="number" step="5" value="{html.escape(vertical_width_text)}">
            </div>
            <h3>Value breakpoints</h3>
            <p class="hint">Tune how strict the vertical watch card should be when classifying a setup as strong, watch, thin, or pass.</p>
            <div class="subgrid">
              <div class="field">
                <label for="strong_profit_threshold">Strong value: minimum proxy profit (pts)</label>
                <input id="strong_profit_threshold" name="strong_profit_threshold" type="number" step="0.05" value="{html.escape(strong_profit_threshold_text)}">
              </div>
              <div class="field">
                <label for="strong_ratio_threshold">Strong value: minimum profit / risk</label>
                <input id="strong_ratio_threshold" name="strong_ratio_threshold" type="number" step="0.05" value="{html.escape(strong_ratio_threshold_text)}">
              </div>
              <div class="field">
                <label for="watch_profit_threshold">Watch: minimum proxy profit (pts)</label>
                <input id="watch_profit_threshold" name="watch_profit_threshold" type="number" step="0.05" value="{html.escape(watch_profit_threshold_text)}">
              </div>
              <div class="field">
                <label for="watch_ratio_threshold">Watch: minimum profit / risk</label>
                <input id="watch_ratio_threshold" name="watch_ratio_threshold" type="number" step="0.05" value="{html.escape(watch_ratio_threshold_text)}">
              </div>
            </div>
            <p class="hint">
              {html.escape(state.underlying_label)} open is prefilled from the latest close as a starting placeholder. Replace it with the actual regular-session open when you have it.
            </p>
            <div class="field">
              <label>Event flags</label>
              <div class="checkbox-list">{event_markup or '<span class="hint">No event types found in the calendar.</span>'}</div>
            </div>
            <h3>Optional prior-day overrides</h3>
            <p class="hint">Use these when the local CSV history is stale and you want to update the latest prior session by hand.</p>
            <div class="subgrid">
              <div class="field"><label>Prior {html.escape(state.underlying_label)} open</label><input name="prior_spx_open" type="number" step="0.01" value="{html.escape(form.get('prior_spx_open', [''])[0])}"></div>
              <div class="field"><label>Prior {html.escape(state.underlying_label)} high</label><input name="prior_spx_high" type="number" step="0.01" value="{html.escape(form.get('prior_spx_high', [''])[0])}"></div>
              <div class="field"><label>Prior {html.escape(state.underlying_label)} low</label><input name="prior_spx_low" type="number" step="0.01" value="{html.escape(form.get('prior_spx_low', [''])[0])}"></div>
              <div class="field"><label>Prior {html.escape(state.underlying_label)} close</label><input name="prior_spx_close" type="number" step="0.01" value="{html.escape(form.get('prior_spx_close', [''])[0])}"></div>
              <div class="field"><label>Prior VIX daily session open</label><input name="prior_vix_open" type="number" step="0.01" value="{html.escape(form.get('prior_vix_open', [''])[0])}"></div>
              <div class="field"><label>Prior VIX high</label><input name="prior_vix_high" type="number" step="0.01" value="{html.escape(form.get('prior_vix_high', [''])[0])}"></div>
              <div class="field"><label>Prior VIX low</label><input name="prior_vix_low" type="number" step="0.01" value="{html.escape(form.get('prior_vix_low', [''])[0])}"></div>
              <div class="field"><label>Prior VIX close</label><input name="prior_vix_close" type="number" step="0.01" value="{html.escape(form.get('prior_vix_close', [''])[0])}"></div>
            </div>
            <button type="submit">Generate today's probabilities</button>
          </form>
        </div>
        <div>
          {error_markup}
          {prediction_markup}
        </div>
      </div>
    """
    return _html_page(f"{state.underlying_label} Excursion App", body)


def create_wsgi_app(state: AppState):
    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        if method == "POST":
            length = int(environ.get("CONTENT_LENGTH") or 0)
            raw_body = environ["wsgi.input"].read(length).decode("utf-8")
            form = parse_qs(raw_body, keep_blank_values=True)
        else:
            form = {}
        payload = render_app(state, form)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(payload)))])
        return [payload]

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SPX excursion webapp")
    parser.add_argument("--spx", default="data/spx_daily.csv", help="Path to SPX daily CSV")
    parser.add_argument("--vix", default="data/vix_daily.csv", help="Path to VIX daily CSV")
    parser.add_argument("--events", default="data/events.csv", help="Path to event calendar CSV")
    parser.add_argument("--underlying-label", default="SPX", help="Label shown in the UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8011, help="Port to bind")
    parser.add_argument("--train-end-date", default="2024-12-31", help="Inclusive end date for model training")
    parser.add_argument("--max-lag", type=int, default=5, help="Maximum lag depth per feature")
    parser.add_argument("--pca-variance-ratio", type=float, default=0.95, help="Target retained PCA variance")
    parser.add_argument("--thresholds", default="0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01", help="Comma-separated excursion thresholds")
    parser.add_argument("--fill-samples", default="data/tradingview_fill_samples_template.csv", help="CSV of sampled single-option fills for quote proxying")
    parser.add_argument("--quotes", default="data/tradingview_single_option_quotes_template.csv", help="CSV of option quote snapshots for vertical proxying")
    parser.add_argument("--vix-snapshots", default="data/tradingview_vix_snapshots_template.csv", help="CSV of VIX snapshots aligned to the quote snapshot file")
    parser.add_argument("--no-auto-refresh", action="store_true", help="Skip the startup pull from the public SPX/VIX sources")
    args = parser.parse_args()

    state = create_app_state(
        spx_path=args.spx,
        vix_path=args.vix,
        events_path=args.events,
        underlying_label=args.underlying_label,
        train_end_date=args.train_end_date,
        max_lag=args.max_lag,
        pca_variance_ratio=args.pca_variance_ratio,
        thresholds=_parse_thresholds(args.thresholds),
        auto_refresh=not args.no_auto_refresh,
        fill_samples_path=args.fill_samples,
        quotes_path=args.quotes,
        vix_snapshots_path=args.vix_snapshots,
    )
    with make_server(args.host, args.port, create_wsgi_app(state)) as server:
        print(f"Serving {args.underlying_label} excursion app on http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
