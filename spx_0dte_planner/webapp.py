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

from .data import load_daily_bars
from .events import collect_event_names, load_event_calendar
from .features import align_and_build
from .live import (
    PriorDayOverrides,
    build_live_feature_row,
    build_continuation_lookup,
    build_regime_probability_lookup,
    compute_regime_cutoffs,
    default_prediction_date,
    required_history_date,
    predict_live_excursions,
    select_continuation_stats,
    threshold_probabilities,
)
from .model import fit_train_backtest_range_model, predict_backtest_range

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
    vix_open_definition: str
    latest_common_history_date: date | None
    refresh_note: str | None


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
            f"spx_path={spx_path} rows={len(spx_bars)}, "
            f"vix_path={vix_path} rows={len(vix_bars)}. "
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
            f"spx_path={spx_path}, vix_path={vix_path}, rows={len(rows)}. "
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
        vix_open_definition="For this training set, VIX open is the daily session open from Cboe's VIX_History.csv OPEN field, aligned by trade date. It is not a separate 9:30 a.m. spot sample.",
        latest_common_history_date=min(spx_bars[-1].date, vix_bars[-1].date) if spx_bars and vix_bars else None,
        refresh_note=refresh_note,
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


def _build_decision_rows(state: AppState, side: str, threshold_rows: List[Dict[str, float]], prediction) -> List[List[str]]:
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
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 2rem; }}
    h2 {{ font-size: 1.25rem; margin-top: 24px; }}
    p, li {{ color: var(--muted); line-height: 1.45; }}
    .grid {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
      align-items: start;
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
    input[type="text"], input[type="number"], input[type="date"] {{
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
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .decision-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin: 12px 0 20px;
    }}
    .metric {{
      background: var(--accent-soft);
      border-radius: 14px;
      padding: 14px;
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
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.92rem;
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
    @media (max-width: 980px) {{
      .grid, .summary, .decision-grid {{ grid-template-columns: 1fr; }}
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


def render_app(state: AppState, form: Mapping[str, List[str]] | None = None) -> bytes:
    form = form or {}
    suggested_trade_date = default_prediction_date(state.spx_bars, state.vix_bars)
    default_date = suggested_trade_date.isoformat()
    latest_spx = state.spx_bars[-1]
    latest_vix = state.vix_bars[-1]
    prediction_date_text = form.get("prediction_date", [default_date])[0]
    spx_open_text = form.get("spx_open", [f"{latest_spx.close:.2f}"])[0]
    vix_open_text = form.get("vix_open", [f"{latest_vix.open:.2f}" if latest_vix.open else f"{latest_vix.close:.2f}"])[0]
    selected_events = _selected_values(form, "event_type")

    prediction_markup = ""
    error_markup = ""
    try:
        prediction_date = date.fromisoformat(prediction_date_text)
        spx_open = float(spx_open_text)
        vix_open = float(vix_open_text)
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
        context = prediction.regime_context
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
          <p class="hint">
            The touch tables show today's forecast in both percentage and absolute SPX terms. `Blended regime hit` averages the matching VIX, gap, and prior-range buckets so we can compare today's probability against a context-aware baseline. The continuation tables answer a second question: if that level is reached on a day like this one, where did the close usually finish historically?
          </p>
          <h3>Decision Summary</h3>
          <p class="hint">
            `focus` means today's probability is meaningfully above the overall base rate and the matching historical regimes had decent ranking quality. `watch` is weaker but still above background.
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
          <h3>Upside Touch Levels</h3>
          <div class="table-wrap">
            {_render_table(
                ["Threshold", "Points", "Touch price", "Today p", "Overall hit", "Blended regime hit", context.vix_regime, context.gap_regime, context.range_regime, "Regime key", "Regime samples"],
                upside_rows,
            )}
          </div>
          <h3>Upside Continuation If Touched</h3>
          <p class="hint">
            `Basis` is the historical bucket used for the continuation band. We use the exact VIX/range/gap combo when it has enough samples; otherwise we fall back to the strongest single regime slice.
          </p>
          <div class="table-wrap">
            {_render_table(
                ["Threshold", "Points", "Touch price", "Basis", "Samples", "Avg close", "Avg close px", "Close band", "Close band px", "Close on side", "Close past touch"],
                upside_continuation_rows,
            )}
          </div>
          <h3>Downside Touch Levels</h3>
          <div class="table-wrap">
            {_render_table(
                ["Threshold", "Points", "Touch price", "Today p", "Overall hit", "Blended regime hit", context.vix_regime, context.gap_regime, context.range_regime, "Regime key", "Regime samples"],
                downside_rows,
            )}
          </div>
          <h3>Downside Continuation If Touched</h3>
          <div class="table-wrap">
            {_render_table(
                ["Threshold", "Points", "Touch price", "Basis", "Samples", "Avg close", "Avg close px", "Close band", "Close band px", "Close on side", "Close past touch"],
                downside_continuation_rows,
            )}
          </div>
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
    )
    with make_server(args.host, args.port, create_wsgi_app(state)) as server:
        print(f"Serving {args.underlying_label} excursion app on http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
