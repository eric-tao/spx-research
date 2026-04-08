import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import {
  BootstrapResponse,
  ContinuationTableRow,
  DecisionSummaryItem,
  FeaturedPlaybook,
  ScenarioRequest,
  ScenarioResponse,
  TouchTableRow,
  VerticalSummary,
  getBootstrap,
  postScenario
} from "./api/client";

type Direction = "upside" | "downside";
type TouchMode = "auto" | "manual";
type TouchSuggestion = {
  touchedSide: ScenarioRequest["touchSelection"]["touchedSide"];
  touchedThresholdPct: number;
  reason: string;
};

type RankedVertical = Record<string, unknown>;
const FORM_STORAGE_KEY = "spx-0dte-planner-form-v1";
const TOUCH_MODE_STORAGE_KEY = "spx-0dte-planner-touch-mode-v1";
const CHECKPOINT_OPTIONS = ["10:00", "10:30", "12:00", "14:00", "15:00", "15:30"] as const;

export function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null);
  const [scenario, setScenario] = useState<ScenarioResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<ScenarioRequest | null>(null);
  const [touchMode, setTouchMode] = useState<TouchMode>(() => loadTouchMode());

  useEffect(() => {
    getBootstrap()
      .then((payload) => {
        setBootstrap(payload);
        const defaultForm = buildDefaultForm(payload);
        setForm(loadStoredForm(payload) ?? defaultForm);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!form) return;
    window.localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(form));
  }, [form]);

  useEffect(() => {
    window.localStorage.setItem(TOUCH_MODE_STORAGE_KEY, touchMode);
  }, [touchMode]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!form) return;
    setSubmitting(true);
    setError(null);
    try {
      const payload = await postScenario(form);
      setScenario(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  const selectedTouchLabel = useMemo(() => {
    if (!form) return "";
    return `${form.touchSelection.touchedSide === "upside_touch" ? "Upside" : "Downside"} ${formatPct(form.touchSelection.touchedThresholdPct)}`;
  }, [form]);

  const autoDetectedTouch = useMemo(() => {
    if (!bootstrap || !form) return null;
    return detectTouchSelection(form, bootstrap.defaultThresholdsPct);
  }, [bootstrap, form]);

  const nearestCheckpoint = useMemo(() => {
    if (!form) return "12:00";
    return nearestCheckpointTime(form.checkpointTime);
  }, [form]);

  useEffect(() => {
    if (!form || touchMode !== "auto" || !autoDetectedTouch) return;
    const nextSide = autoDetectedTouch.touchedSide;
    const nextThreshold = autoDetectedTouch.touchedThresholdPct;
    if (
      form.touchSelection.touchedSide === nextSide &&
      form.touchSelection.touchedThresholdPct === nextThreshold
    ) {
      return;
    }
    setForm({
      ...form,
      touchSelection: {
        touchedSide: nextSide,
        touchedThresholdPct: nextThreshold
      }
    });
  }, [autoDetectedTouch, form, touchMode]);

  if (loading) {
    return (
      <main className="app-shell">
        <section className="loading-state">
          <p className="eyebrow">React Planner</p>
          <h1>Loading the planner state</h1>
          <p>We’re pulling the bootstrap data and default inputs for the first scenario.</p>
        </section>
      </main>
    );
  }

  if (!bootstrap || !form) {
    return (
      <main className="app-shell">
        <section className="loading-state error-surface">
          <h1>Unable to load bootstrap data</h1>
          <p>{error ?? "The frontend could not initialize the planner."}</p>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">React Planner Preview</p>
          <h1>SPX 0DTE planning, with the important signals up front</h1>
          <p className="hero-copy">
            This is the new React frontend on top of the Python analytics API. We keep the Python webapp around for
            comparison, but this view is now structured the same way: forecast first, playbooks second, and the deeper
            evidence behind collapsible sections.
          </p>
        </div>
        <div className="hero-meta">
          <div className="hero-chip">
            <span>Suggested trade date</span>
            <strong>{bootstrap.suggestedTradeDate}</strong>
          </div>
          <div className="hero-chip">
            <span>Latest aligned history</span>
            <strong>{bootstrap.latestCommonHistoryDate ?? "Unavailable"}</strong>
          </div>
          <div className="hero-chip">
            <span>Vertical inputs</span>
            <strong>{bootstrap.dataStatus.verticalInputsAvailable ? "Available" : "Not loaded"}</strong>
          </div>
        </div>
      </section>

      {error ? (
        <section className="banner banner-error">
          <strong>Scenario request failed.</strong>
          <span>{error}</span>
        </section>
      ) : null}

      {bootstrap.dataStatus.refreshNote ? (
        <section className="banner banner-info">
          <strong>Data note.</strong>
          <span>{bootstrap.dataStatus.refreshNote}</span>
        </section>
      ) : null}

      <div className="layout-grid">
        <aside className="sidebar">
          <section className="panel input-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Inputs</p>
                <h2>Scenario setup</h2>
              </div>
              <span className="status-pill status-pill-neutral">React + Python API</span>
            </div>

            <form className="stack" onSubmit={onSubmit}>
              <InputSection title="Session inputs" description="These drive the pre-open forecast and the regime mapping.">
                <div className="form-grid">
                  <label className="field">
                    <span>Prediction date</span>
                    <input
                      type="date"
                      value={form.predictionDate}
                      onChange={(e) => setForm({ ...form, predictionDate: e.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span>Checkpoint time (ET)</span>
                    <input
                      type="time"
                      step="60"
                      value={form.checkpointTime}
                      onChange={(e) => setForm({ ...form, checkpointTime: e.target.value })}
                    />
                    <div className="checkpoint-toolbar">
                      <span className="field-hint">Preset checkpoints</span>
                      <button
                        type="button"
                        className="checkpoint-snap"
                        onClick={() => setForm({ ...form, checkpointTime: nearestCheckpoint })}
                      >
                        Snap to nearest: {nearestCheckpoint}
                      </button>
                    </div>
                    <div className="checkpoint-picker">
                      {CHECKPOINT_OPTIONS.map((checkpoint) => (
                        <button
                          key={checkpoint}
                          type="button"
                          className={`checkpoint-chip ${form.checkpointTime === checkpoint ? "checkpoint-chip-active" : ""}`}
                          onClick={() => setForm({ ...form, checkpointTime: checkpoint })}
                        >
                          {checkpoint}
                        </button>
                      ))}
                    </div>
                    <small className="field-hint">
                      Use any intraday timestamp you want, or snap back to the backtested session checkpoints.
                    </small>
                  </label>
                  <label className="field">
                    <span>SPX regular-session open</span>
                    <NumericInput
                      value={form.spxOpen}
                      onValueChange={(value) => setForm({ ...form, spxOpen: value })}
                    />
                  </label>
                  <label className="field">
                    <span>VIX daily session open</span>
                    <NumericInput
                      value={form.vixOpen}
                      onValueChange={(value) => setForm({ ...form, vixOpen: value })}
                    />
                  </label>
                </div>
              </InputSection>

              <InputSection title="Observed intraday structure" description="Use realized intraday state so the continuation view is grounded in what has actually happened.">
                <div className="form-grid">
                  <label className="field">
                    <span>Current SPX spot</span>
                    <NumericInput
                      value={form.currentSpot}
                      onValueChange={(value) => setForm({ ...form, currentSpot: value })}
                    />
                  </label>
                  <label className="field">
                    <span>High so far</span>
                    <NumericInput
                      value={form.highSoFar}
                      onValueChange={(value) => setForm({ ...form, highSoFar: value })}
                    />
                  </label>
                  <label className="field">
                    <span>Low so far</span>
                    <NumericInput
                      value={form.lowSoFar}
                      onValueChange={(value) => setForm({ ...form, lowSoFar: value })}
                    />
                  </label>
                </div>
              </InputSection>

              <InputSection title="Touch setup" description="This tells the app which realized move to condition the close and vertical analysis on.">
                <div className="toggle-row">
                  <button
                    type="button"
                    className={`toggle-chip ${touchMode === "auto" ? "toggle-chip-active" : ""}`}
                    onClick={() => setTouchMode("auto")}
                  >
                    Auto-detect
                  </button>
                  <button
                    type="button"
                    className={`toggle-chip ${touchMode === "manual" ? "toggle-chip-active" : ""}`}
                    onClick={() => setTouchMode("manual")}
                  >
                    Manual override
                  </button>
                </div>
                <div className="form-grid">
                  <label className="field">
                    <span>Touched side</span>
                    <select
                      value={form.touchSelection.touchedSide}
                      disabled={touchMode === "auto"}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          touchSelection: {
                            ...form.touchSelection,
                            touchedSide: e.target.value as ScenarioRequest["touchSelection"]["touchedSide"]
                          }
                        })
                      }
                    >
                      <option value="upside_touch">Upside touch</option>
                      <option value="downside_touch">Downside touch</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>Touched threshold (%)</span>
                    <NumericInput
                      value={form.touchSelection.touchedThresholdPct}
                      disabled={touchMode === "auto"}
                      onValueChange={(value) =>
                        setForm({
                          ...form,
                          touchSelection: {
                            ...form.touchSelection,
                            touchedThresholdPct: value
                          }
                        })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Vertical width (points)</span>
                    <NumericInput
                      value={form.verticalSelection.widthPoints}
                      onValueChange={(value) =>
                        setForm({
                          ...form,
                          verticalSelection: {
                            widthPoints: value
                          }
                        })
                      }
                    />
                  </label>
                </div>
                <div className="inline-note">
                  Selected focus: <strong>{selectedTouchLabel}</strong>
                </div>
                {autoDetectedTouch ? (
                  <div className="supporting-note">
                    <strong>Auto-detect logic</strong>
                    <span>
                      {touchMode === "auto" ? "Using" : "Suggested"} {autoDetectedTouch.touchedSide === "upside_touch" ? "upside" : "downside"}{" "}
                      {formatPct(autoDetectedTouch.touchedThresholdPct)}.
                    </span>
                    <span>{autoDetectedTouch.reason}</span>
                    {touchMode === "manual" ? (
                      <button
                        type="button"
                        className="inline-action"
                        onClick={() =>
                          setForm({
                            ...form,
                            touchSelection: {
                              touchedSide: autoDetectedTouch.touchedSide,
                              touchedThresholdPct: autoDetectedTouch.touchedThresholdPct
                            }
                          })
                        }
                      >
                        Use detected values
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </InputSection>

              <InputSection title="Value breakpoints" description="These let us classify the best vertical idea as strong value, watch, thin, or pass.">
                <div className="form-grid">
                  <label className="field">
                    <span>Strong value: minimum profit (pts)</span>
                    <NumericInput
                      value={form.valueBreakpoints.strongProfitThreshold}
                      onValueChange={(value) =>
                        setForm({
                          ...form,
                          valueBreakpoints: {
                            ...form.valueBreakpoints,
                            strongProfitThreshold: value
                          }
                        })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Strong value: minimum profit / risk</span>
                    <NumericInput
                      value={form.valueBreakpoints.strongRatioThreshold}
                      onValueChange={(value) =>
                        setForm({
                          ...form,
                          valueBreakpoints: {
                            ...form.valueBreakpoints,
                            strongRatioThreshold: value
                          }
                        })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Watch: minimum profit (pts)</span>
                    <NumericInput
                      value={form.valueBreakpoints.watchProfitThreshold}
                      onValueChange={(value) =>
                        setForm({
                          ...form,
                          valueBreakpoints: {
                            ...form.valueBreakpoints,
                            watchProfitThreshold: value
                          }
                        })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Watch: minimum profit / risk</span>
                    <NumericInput
                      value={form.valueBreakpoints.watchRatioThreshold}
                      onValueChange={(value) =>
                        setForm({
                          ...form,
                          valueBreakpoints: {
                            ...form.valueBreakpoints,
                            watchRatioThreshold: value
                          }
                        })
                      }
                    />
                  </label>
                </div>
              </InputSection>

              {bootstrap.eventNames.length > 0 ? (
                <details className="disclosure">
                  <summary>Event flags</summary>
                  <div className="checkbox-grid">
                    {bootstrap.eventNames.map((eventName) => {
                      const checked = form.selectedEvents.includes(eventName);
                      return (
                        <label className="checkbox" key={eventName}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() =>
                              setForm({
                                ...form,
                                selectedEvents: checked
                                  ? form.selectedEvents.filter((value) => value !== eventName)
                                  : [...form.selectedEvents, eventName]
                              })
                            }
                          />
                          <span>{eventName}</span>
                        </label>
                      );
                    })}
                  </div>
                </details>
              ) : null}

              <button className="primary-button" type="submit" disabled={submitting}>
                {submitting ? "Running scenario..." : "Run scenario"}
              </button>
            </form>
          </section>
        </aside>

        <section className="content">
          {!scenario ? (
            <section className="panel empty-state">
              <p className="eyebrow">Scenario output</p>
              <h2>Run a scenario to populate the planner</h2>
              <p>
                We’ll show the forecast, current regime context, the top touch levels to care about, what similar days
                usually did by the close after a touch, and the best vertical strategy to watch.
              </p>
            </section>
          ) : (
            <>
              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="eyebrow">Forecast</p>
                    <h2>What looks important today</h2>
                  </div>
                  <span className={`status-pill ${scenario.intradayState.touchConfirmed ? "status-pill-good" : "status-pill-watch"}`}>
                    {scenario.intradayState.touchConfirmed ? "Touch confirmed" : "Planning view"}
                  </span>
                </div>

                <div className="metric-grid">
                  <MetricCard
                    label="Predicted upside from open"
                    value={formatPct(scenario.forecast.predictedHighFromOpenPct)}
                    detail={`Projected high near ${formatPrice(scenario.forecast.predictedHighPrice)}`}
                  />
                  <MetricCard
                    label="Predicted downside from open"
                    value={formatPct(scenario.forecast.predictedLowFromOpenPct)}
                    detail={`Projected low near ${formatPrice(scenario.forecast.predictedLowPrice)}`}
                  />
                  <MetricCard
                    label="Move from open so far"
                    value={formatPct(scenario.intradayState.currentMovePct)}
                    detail={`${formatPoints(form.currentSpot - form.spxOpen)} points`}
                  />
                  <MetricCard
                    label="Selected touch level"
                    value={formatPrice(scenario.intradayState.touchPrice)}
                    detail={scenario.intradayState.touchConsistencyLabel}
                  />
                </div>

                <div className="pill-row">
                  <Pill title="Weekday" value={titleCase(scenario.regimeContext.weekday)} />
                  <Pill title="VIX regime" value={titleCase(scenario.regimeContext.vixRegime)} />
                  <Pill title="Prior-day range" value={titleCase(scenario.regimeContext.rangeRegime)} />
                  <Pill title="Gap regime" value={titleCase(scenario.regimeContext.gapRegime)} />
                  <Pill title="Touch source" value={titleCase(scenario.intradayState.touchConfirmationSource)} />
                </div>
              </section>

              <section className="two-col-grid">
                <DecisionCard title="Top upside thresholds" rows={scenario.decisionSummary.upside} direction="upside" />
                <DecisionCard title="Top downside thresholds" rows={scenario.decisionSummary.downside} direction="downside" />
              </section>

              <section className="two-col-grid">
                <PlaybookCard
                  title="Upside playbook"
                  playbook={scenario.featuredPlaybooks.upside}
                  accent="upside-accent"
                />
                <PlaybookCard
                  title="Downside playbook"
                  playbook={scenario.featuredPlaybooks.downside}
                  accent="downside-accent"
                />
              </section>

              <section className="two-col-grid">
                <IntradayStateCard scenario={scenario} form={form} />
                <VerticalWatchCard
                  summary={scenario.verticalStrategy.summary}
                  notes={scenario.verticalStrategy.notes}
                  provenance={scenario.verticalStrategy.pricingProvenance}
                  spxOpen={form.spxOpen}
                  currentSpot={form.currentSpot}
                />
              </section>

              <DetailSection
                title="Upside touch levels"
                description="These are the upside thresholds worth watching today, with model probability and historical hit rate side by side."
                defaultOpen
              >
                <TouchTable rows={scenario.touchTables.upside} direction="upside" />
              </DetailSection>

              <DetailSection
                title="Upside continuation if touched"
                description="If upside levels like these were reached on historically similar days, this is where the close usually finished."
              >
                <ContinuationTable rows={scenario.continuationTables.upside} direction="upside" />
              </DetailSection>

              <DetailSection
                title="Downside touch levels"
                description="Same idea on the downside: this keeps the current day anchored to the most relevant downside thresholds."
              >
                <TouchTable rows={scenario.touchTables.downside} direction="downside" />
              </DetailSection>

              <DetailSection
                title="Downside continuation if touched"
                description="This is the conditional close view after a downside touch, which helps separate clean continuation from bounce risk."
              >
                <ContinuationTable rows={scenario.continuationTables.downside} direction="downside" />
              </DetailSection>

              <DetailSection
                title="Vertical ranking details"
                description="The headline card keeps the best idea visible. This table lets us inspect the nearby alternatives and how they compare."
              >
                <RankedVerticalTable
                  rows={scenario.verticalStrategy.ranked}
                  widthPoints={form.verticalSelection.widthPoints}
                />
              </DetailSection>

              <DetailSection
                title="Model and API details"
                description="This keeps the contract visible for debugging without dominating the screen."
              >
                <div className="mini-grid">
                  <MetricCard
                    label="Train end"
                    value={scenario.modelConfig.trainEndDate}
                    detail={`PCA variance ${scenario.modelConfig.pcaVarianceRatio}`}
                  />
                  <MetricCard
                    label="Threshold grid"
                    value={scenario.modelConfig.thresholdsPct.map((value) => formatPct(value)).join(", ")}
                    detail={`Max lag ${scenario.modelConfig.maxLag}`}
                  />
                </div>
                <details className="disclosure raw-json">
                  <summary>Raw API response</summary>
                  <pre>{JSON.stringify(scenario, null, 2)}</pre>
                </details>
              </DetailSection>
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function InputSection({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="input-section">
      <div className="input-section-header">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      {children}
    </section>
  );
}

function NumericInput({
  value,
  onValueChange,
  disabled = false
}: {
  value: number;
  onValueChange: (value: number) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState(() => formatEditableNumber(value));

  useEffect(() => {
    setText((current) => (isTransientNumericText(current) ? current : formatEditableNumber(value)));
  }, [value]);

  return (
    <input
      type="text"
      inputMode="decimal"
      value={text}
      disabled={disabled}
      onChange={(event) => {
        const next = event.target.value;
        if (!isValidNumericText(next)) return;
        setText(next);
        const parsed = parseCommittedNumericText(next);
        if (parsed !== null) {
          onValueChange(parsed);
        }
      }}
      onBlur={() => {
        const parsed = parseCommittedNumericText(text);
        if (parsed === null) {
          setText(formatEditableNumber(value));
          return;
        }
        onValueChange(parsed);
        setText(formatEditableNumber(parsed));
      }}
      onFocus={selectAllOnFocus}
    />
  );
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

function Pill({ title, value }: { title: string; value: string }) {
  return (
    <div className="pill">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DecisionCard({
  title,
  rows,
  direction
}: {
  title: string;
  rows: DecisionSummaryItem[];
  direction: Direction;
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{direction === "upside" ? "Continuation bias" : "Risk focus"}</p>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Label</th>
              <th>Today p</th>
              <th>Base rate</th>
              <th>Edge</th>
              <th>Regime AUC</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${direction}-${row.threshold_pct}`}>
                <td>{formatPct(row.threshold_pct)}</td>
                <td>
                  <span className={`value-label ${labelToneClass(row.label)}`}>{row.label}</span>
                </td>
                <td>{formatRate(row.today_probability)}</td>
                <td>{formatRate(row.overall_rate)}</td>
                <td>{signedRate(row.edge)}</td>
                <td>{row.support_auc.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PlaybookCard({
  title,
  playbook,
  accent
}: {
  title: string;
  playbook: FeaturedPlaybook;
  accent: string;
}) {
  return (
    <section className={`panel playbook-card ${accent}`}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">{playbook.eyebrow}</p>
          <h2>{title}</h2>
        </div>
        <span className={`status-pill ${labelToneClass(playbook.label, true)}`}>{playbook.label}</span>
      </div>
      <h3>{playbook.title}</h3>
      <p className="playbook-summary">{playbook.summary}</p>
      <p className="playbook-detail">{playbook.detail}</p>
      <div className="supporting-note">
        <strong>Historical basis</strong>
        <span>{playbook.basis}</span>
      </div>
    </section>
  );
}

function IntradayStateCard({ scenario, form }: { scenario: ScenarioResponse; form: ScenarioRequest }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Intraday state</p>
          <h2>What the market has already realized</h2>
        </div>
      </div>
      <div className="metric-grid compact-grid">
        <MetricCard
          label="Current spot"
          value={formatPrice(form.currentSpot)}
          detail={`${formatPct(scenario.intradayState.currentMovePct)} from open`}
        />
        <MetricCard
          label="High so far"
          value={formatPrice(form.highSoFar)}
          detail={`${formatPct(scenario.intradayState.highMovePct)} from open`}
        />
        <MetricCard
          label="Low so far"
          value={formatPrice(form.lowSoFar)}
          detail={`${formatPct(scenario.intradayState.lowMovePct)} from open`}
        />
      </div>
      <p className="intraday-note">{scenario.intradayState.touchConsistencyLabel}</p>
    </section>
  );
}

function VerticalWatchCard({
  summary,
  notes,
  provenance,
  spxOpen,
  currentSpot
}: {
  summary: VerticalSummary | null;
  notes: string[];
  provenance: ScenarioResponse["verticalStrategy"]["pricingProvenance"];
  spxOpen: number;
  currentSpot: number;
}) {
  if (!summary) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Vertical strategy to watch</p>
            <h2>No vertical idea passed the current screen</h2>
          </div>
          <span className="status-pill status-pill-pass">Pass</span>
        </div>
        <p className="playbook-detail">
          This usually means the quote snapshot was too stale, the current spot filtered out the candidates, or the
          estimated value after execution is not attractive enough right now.
        </p>
        <ProvenanceBlock provenance={provenance} notes={notes} />
      </section>
    );
  }

  const lowerFromOpenPct = spxOpen > 0 ? ((summary.lowerStrike - spxOpen) / spxOpen) * 100.0 : 0;
  const upperFromOpenPct = spxOpen > 0 ? ((summary.upperStrike - spxOpen) / spxOpen) * 100.0 : 0;
  const lowerDiff = currentSpot - summary.lowerStrike;
  const upperDiff = currentSpot - summary.upperStrike;

  return (
    <section className="panel vertical-watch-card">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Vertical strategy to watch</p>
          <h2>{humanizeStrategy(summary.strategy)}</h2>
        </div>
        <span className={`status-pill ${valueBucketClass(summary.valueBucket)}`}>{summary.valueBucket}</span>
      </div>
      <p className="playbook-summary">{summary.outlook}</p>
      <div className="vertical-stat-row">
        <div>
          <span>Strike band</span>
          <strong>
            {formatPrice(summary.lowerStrike)} / {formatPrice(summary.upperStrike)}
          </strong>
        </div>
        <div>
          <span>From the open</span>
          <strong>
            {formatPct(lowerFromOpenPct)} / {formatPct(upperFromOpenPct)}
          </strong>
        </div>
      </div>
      <div className="vertical-stat-row">
        <div>
          <span>Spot vs lower strike</span>
          <strong>{describeSpotVsStrike(lowerDiff)}</strong>
        </div>
        <div>
          <span>Spot vs upper strike</span>
          <strong>{describeSpotVsStrike(upperDiff)}</strong>
        </div>
      </div>
      <div className="metric-grid compact-grid">
        <MetricCard
          label="Entry"
          value={formatPoints(summary.entryPrice)}
          detail="Estimated fill proxy"
        />
        <MetricCard
          label="Terminal value proxy"
          value={formatPoints(summary.predictedTerminalValueProxy)}
          detail="Capped by spread width"
        />
        <MetricCard
          label="Profit proxy"
          value={signedPoints(summary.predictedProfitProxy)}
          detail={summary.valueBucketExplanation}
        />
        <MetricCard
          label="Profit / risk"
          value={`${summary.profitToRiskRatioProxy.toFixed(2)}x`}
          detail="Execution-adjusted proxy"
        />
      </div>
      <ProvenanceBlock provenance={provenance} notes={notes} />
    </section>
  );
}

function ProvenanceBlock({
  provenance,
  notes
}: {
  provenance: ScenarioResponse["verticalStrategy"]["pricingProvenance"];
  notes: string[];
}) {
  return (
    <div className="supporting-note">
      <strong>Pricing provenance</strong>
      <span>
        {titleCase(provenance.pricingMode)}; {provenance.filteredCandidateCount} filtered candidate
        {provenance.filteredCandidateCount === 1 ? "" : "s"}; {provenance.sourceSnapshotCount} source snapshot
        {provenance.sourceSnapshotCount === 1 ? "" : "s"}.
      </span>
      {notes.map((note) => (
        <span key={note}>{note}</span>
      ))}
    </div>
  );
}

function DetailSection({
  title,
  description,
  children,
  defaultOpen = false
}: {
  title: string;
  description: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="panel detail-panel" open={defaultOpen}>
      <summary>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </summary>
      <div className="detail-content">{children}</div>
    </details>
  );
}

function TouchTable({ rows, direction }: { rows: TouchTableRow[]; direction: Direction }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Points</th>
            <th>Touch price</th>
            <th>Today p</th>
            <th>Historical hit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${direction}-${row.thresholdPct}`}>
              <td>{formatPct(row.thresholdPct)}</td>
              <td>{formatPoints(row.points)}</td>
              <td>{formatPrice(row.touchPrice)}</td>
              <td>{formatRate(row.todayProbability)}</td>
              <td>{formatRate(row.overallHitRate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ContinuationTable({ rows, direction }: { rows: ContinuationTableRow[]; direction: Direction }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Basis</th>
            <th>Samples</th>
            <th>Avg close</th>
            <th>Continuation band</th>
            <th>Close on side</th>
            <th>Close past touch</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${direction}-${row.thresholdPct}-${row.basis}`}>
              <td>{formatPct(row.thresholdPct)}</td>
              <td>{row.basis}</td>
              <td>{row.samples}</td>
              <td>{formatReturn(row.avgCloseReturn)}</td>
              <td>
                {formatReturn(row.closeReturnQ25)} to {formatReturn(row.closeReturnQ75)}
              </td>
              <td>{formatRate(row.closeOnSideRate)}</td>
              <td>{formatRate(row.closePastTouchRate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RankedVerticalTable({ rows, widthPoints }: { rows: RankedVertical[]; widthPoints: number }) {
  if (rows.length === 0) {
    return <p className="playbook-detail">No vertical alternatives passed the current screen.</p>;
  }

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Strikes</th>
            <th>Entry</th>
            <th>Terminal proxy</th>
            <th>Profit proxy</th>
            <th>Profit / risk</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const strategy = stringCell(row, "strategy");
            const lowerStrike = numberCell(row, "lower_strike");
            const upperStrike = numberCell(row, "upper_strike");
            const entryPrice = numberCell(row, "entry_price");
            const terminalValue = numberCell(row, "predicted_terminal_value_proxy");
            const profit = numberCell(row, "predicted_profit_proxy");
            const ratio = numberCell(row, "profit_to_cost_ratio_proxy");
            const actualWidth = Math.abs(upperStrike - lowerStrike);
            return (
              <tr key={`${strategy}-${index}`}>
                <td>{humanizeStrategy(strategy)}</td>
                <td>
                  {formatPrice(lowerStrike)} / {formatPrice(upperStrike)}
                  <div className="cell-note">
                    {formatPoints(actualWidth)}-point listed width
                    {Math.abs(actualWidth - widthPoints) > 0.01 ? ` (requested ${formatPoints(widthPoints)})` : ""}
                  </div>
                </td>
                <td>{formatPoints(entryPrice)}</td>
                <td>{formatPoints(terminalValue)}</td>
                <td>{signedPoints(profit)}</td>
                <td>{ratio.toFixed(2)}x</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function stringCell(row: RankedVertical, key: string): string {
  const value = row[key];
  return typeof value === "string" ? value : "";
}

function numberCell(row: RankedVertical, key: string): number {
  const value = row[key];
  return typeof value === "number" ? value : 0;
}

function formatPct(value: number): string {
  return `${value.toFixed(2)}%`;
}

function formatPrice(value: number): string {
  return value.toFixed(2);
}

function formatPoints(value: number): string {
  return value.toFixed(1);
}

function formatRate(value: number): string {
  return `${(value * 100.0).toFixed(1)}%`;
}

function formatReturn(value: number): string {
  return `${value >= 0 ? "+" : ""}${(value * 100.0).toFixed(2)}%`;
}

function signedRate(value: number): string {
  return `${value >= 0 ? "+" : ""}${(value * 100.0).toFixed(1)}%`;
}

function signedPoints(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function titleCase(value: string): string {
  return value.replace(/_/g, " ");
}

function labelToneClass(label: string, pill = false): string {
  const normalized = label.toLowerCase();
  if (normalized.includes("focus") || normalized.includes("strong")) {
    return pill ? "status-pill-good" : "value-label-good";
  }
  if (normalized.includes("watch")) {
    return pill ? "status-pill-watch" : "value-label-watch";
  }
  return pill ? "status-pill-neutral" : "value-label-neutral";
}

function valueBucketClass(bucket: string): string {
  const normalized = bucket.toLowerCase();
  if (normalized.includes("strong")) return "status-pill-good";
  if (normalized.includes("watch")) return "status-pill-watch";
  if (normalized.includes("thin")) return "status-pill-neutral";
  return "status-pill-pass";
}

function humanizeStrategy(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function describeSpotVsStrike(diff: number): string {
  const distance = Math.abs(diff);
  if (distance < 0.05) {
    return "At the strike";
  }
  return diff >= 0 ? `${formatPoints(distance)} points above the strike` : `${formatPoints(distance)} points below the strike`;
}

function selectAllOnFocus(event: React.FocusEvent<HTMLInputElement>) {
  event.currentTarget.select();
}

function buildDefaultForm(payload: BootstrapResponse): ScenarioRequest {
  return {
    predictionDate: payload.suggestedTradeDate,
    spxOpen: 0,
    vixOpen: 0,
    currentSpot: 0,
    checkpointTime: "12:00",
    highSoFar: 0,
    lowSoFar: 0,
    selectedEvents: [],
    touchSelection: {
      touchedSide: payload.defaults.touchedSide,
      touchedThresholdPct: payload.defaults.touchedThresholdPct
    },
    verticalSelection: {
      widthPoints: payload.defaults.verticalWidthPoints
    },
    valueBreakpoints: {
      strongProfitThreshold: payload.defaults.strongProfitThreshold,
      strongRatioThreshold: payload.defaults.strongRatioThreshold,
      watchProfitThreshold: payload.defaults.watchProfitThreshold,
      watchRatioThreshold: payload.defaults.watchRatioThreshold
    }
  };
}

function loadStoredForm(payload: BootstrapResponse): ScenarioRequest | null {
  const raw = window.localStorage.getItem(FORM_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ScenarioRequest>;
    const defaults = buildDefaultForm(payload);
    const checkpointTime =
      typeof parsed.checkpointTime === "string" && CHECKPOINT_OPTIONS.includes(parsed.checkpointTime as (typeof CHECKPOINT_OPTIONS)[number])
        ? parsed.checkpointTime
        : defaults.checkpointTime;
    return {
      ...defaults,
      ...parsed,
      predictionDate: typeof parsed.predictionDate === "string" ? parsed.predictionDate : defaults.predictionDate,
      spxOpen: finiteNumber(parsed.spxOpen, defaults.spxOpen),
      vixOpen: finiteNumber(parsed.vixOpen, defaults.vixOpen),
      currentSpot: finiteNumber(parsed.currentSpot, defaults.currentSpot),
      checkpointTime,
      highSoFar: finiteNumber(parsed.highSoFar, defaults.highSoFar),
      lowSoFar: finiteNumber(parsed.lowSoFar, defaults.lowSoFar),
      selectedEvents: Array.isArray(parsed.selectedEvents) ? parsed.selectedEvents.filter((value): value is string => typeof value === "string") : [],
      touchSelection: {
        touchedSide:
          parsed.touchSelection?.touchedSide === "downside_touch" ? "downside_touch" : defaults.touchSelection.touchedSide,
        touchedThresholdPct: finiteNumber(parsed.touchSelection?.touchedThresholdPct, defaults.touchSelection.touchedThresholdPct)
      },
      verticalSelection: {
        widthPoints: finiteNumber(parsed.verticalSelection?.widthPoints, defaults.verticalSelection.widthPoints)
      },
      valueBreakpoints: {
        strongProfitThreshold: finiteNumber(
          parsed.valueBreakpoints?.strongProfitThreshold,
          defaults.valueBreakpoints.strongProfitThreshold
        ),
        strongRatioThreshold: finiteNumber(
          parsed.valueBreakpoints?.strongRatioThreshold,
          defaults.valueBreakpoints.strongRatioThreshold
        ),
        watchProfitThreshold: finiteNumber(
          parsed.valueBreakpoints?.watchProfitThreshold,
          defaults.valueBreakpoints.watchProfitThreshold
        ),
        watchRatioThreshold: finiteNumber(
          parsed.valueBreakpoints?.watchRatioThreshold,
          defaults.valueBreakpoints.watchRatioThreshold
        )
      }
    };
  } catch {
    return null;
  }
}

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nearestCheckpointTime(value: string): string {
  const currentMinutes = parseTimeToMinutes(value);
  if (currentMinutes === null) return "12:00";

  let bestCheckpoint = CHECKPOINT_OPTIONS[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const checkpoint of CHECKPOINT_OPTIONS) {
    const checkpointMinutes = parseTimeToMinutes(checkpoint);
    if (checkpointMinutes === null) continue;
    const distance = Math.abs(checkpointMinutes - currentMinutes);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestCheckpoint = checkpoint;
    }
  }
  return bestCheckpoint;
}

function parseTimeToMinutes(value: string): number | null {
  const match = /^(\d{2}):(\d{2})$/.exec(value);
  if (!match) return null;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  if (!Number.isInteger(hours) || !Number.isInteger(minutes)) return null;
  return hours * 60 + minutes;
}

function formatEditableNumber(value: number): string {
  return Number.isFinite(value) ? String(value) : "0";
}

function isValidNumericText(value: string): boolean {
  return /^-?\d*\.?\d*$/.test(value.trim());
}

function isTransientNumericText(value: string): boolean {
  const trimmed = value.trim();
  return trimmed === "" || trimmed === "-" || trimmed === "." || trimmed === "-." || trimmed.endsWith(".");
}

function parseCommittedNumericText(value: string): number | null {
  const trimmed = value.trim();
  if (trimmed === "") return 0;
  if (!isValidNumericText(trimmed) || isTransientNumericText(trimmed)) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function loadTouchMode(): TouchMode {
  const stored = window.localStorage.getItem(TOUCH_MODE_STORAGE_KEY);
  return stored === "manual" ? "manual" : "auto";
}

function detectTouchSelection(form: ScenarioRequest, thresholdsPct: number[]): TouchSuggestion {
  const spxOpen = form.spxOpen;
  const highMovePct = spxOpen > 0 ? ((form.highSoFar - spxOpen) / spxOpen) * 100.0 : 0;
  const lowMovePct = spxOpen > 0 ? ((spxOpen - form.lowSoFar) / spxOpen) * 100.0 : 0;
  const currentMovePct = spxOpen > 0 ? ((form.currentSpot - spxOpen) / spxOpen) * 100.0 : 0;
  const sortedThresholds = [...thresholdsPct].sort((left, right) => left - right);
  const upsideTouched = largestTouchedThreshold(highMovePct, sortedThresholds);
  const downsideTouched = largestTouchedThreshold(lowMovePct, sortedThresholds);

  if (upsideTouched !== null || downsideTouched !== null) {
    const chooseUpside =
      upsideTouched !== null &&
      (downsideTouched === null ||
        highMovePct > lowMovePct ||
        (Math.abs(highMovePct - lowMovePct) < 0.0001 && currentMovePct >= 0));
    return {
      touchedSide: chooseUpside ? "upside_touch" : "downside_touch",
      touchedThresholdPct: chooseUpside ? upsideTouched ?? sortedThresholds[0] : downsideTouched ?? sortedThresholds[0],
      reason: chooseUpside
        ? `High so far has already reached ${formatPct(highMovePct)} from the open at the ${form.checkpointTime} snapshot, so the app is conditioning on the deepest confirmed upside touch.`
        : `Low so far has already reached ${formatPct(lowMovePct)} below the open at the ${form.checkpointTime} snapshot, so the app is conditioning on the deepest confirmed downside touch.`
    };
  }

  const fallbackUpside = currentMovePct >= 0;
  const referenceMove = Math.abs(currentMovePct);
  const nearestThreshold = nearestThresholdPct(referenceMove, sortedThresholds);
  return {
    touchedSide: fallbackUpside ? "upside_touch" : "downside_touch",
    touchedThresholdPct: nearestThreshold,
    reason: `No modeled threshold has been confirmed by high/low so far at ${form.checkpointTime}, so the app is falling back to the current move from open of ${formatPct(currentMovePct)} and the nearest modeled threshold.`
  };
}

function largestTouchedThreshold(movePct: number, thresholdsPct: number[]): number | null {
  const touched = thresholdsPct.filter((threshold) => movePct >= threshold);
  return touched.length > 0 ? touched[touched.length - 1] : null;
}

function nearestThresholdPct(movePct: number, thresholdsPct: number[]): number {
  let best = thresholdsPct[0] ?? 0.5;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const threshold of thresholdsPct) {
    const distance = Math.abs(threshold - movePct);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = threshold;
    }
  }
  return best;
}
