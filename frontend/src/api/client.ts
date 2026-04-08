export type BootstrapResponse = {
  underlyingLabel: string;
  latestCommonHistoryDate: string | null;
  suggestedTradeDate: string;
  defaultThresholdsPct: number[];
  eventNames: string[];
  defaults: {
    touchedSide: "upside_touch" | "downside_touch";
    touchedThresholdPct: number;
    verticalWidthPoints: number;
    strongProfitThreshold: number;
    strongRatioThreshold: number;
    watchProfitThreshold: number;
    watchRatioThreshold: number;
  };
  dataStatus: {
    verticalInputsAvailable: boolean;
    refreshNote: string | null;
  };
  modelConfig: {
    trainEndDate: string;
    maxLag: number;
    pcaVarianceRatio: number;
    thresholdsPct: number[];
  };
};

export type ScenarioRequest = {
  predictionDate: string;
  spxOpen: number;
  vixOpen: number;
  currentSpot: number;
  checkpointTime: string;
  highSoFar: number;
  lowSoFar: number;
  selectedEvents: string[];
  touchSelection: {
    touchedSide: "upside_touch" | "downside_touch";
    touchedThresholdPct: number;
  };
  verticalSelection: {
    widthPoints: number;
  };
  valueBreakpoints: {
    strongProfitThreshold: number;
    strongRatioThreshold: number;
    watchProfitThreshold: number;
    watchRatioThreshold: number;
  };
};

export type DecisionSummaryItem = {
  threshold_pct: number;
  today_probability: number;
  overall_rate: number;
  edge: number;
  support_auc: number;
  label: string;
};

export type FeaturedPlaybook = {
  eyebrow: string;
  title: string;
  summary: string;
  detail: string;
  basis: string;
  label: string;
};

export type TouchTableRow = {
  thresholdPct: number;
  points: number;
  touchPrice: number;
  todayProbability: number;
  overallHitRate: number;
};

export type ContinuationTableRow = {
  thresholdPct: number;
  basis: string;
  samples: number;
  avgCloseReturn: number;
  closeReturnQ25: number;
  closeReturnQ75: number;
  closeOnSideRate: number;
  closePastTouchRate: number;
};

export type VerticalSummary = {
  strategy: string;
  outlook: string;
  lowerStrike: number;
  upperStrike: number;
  entryPrice: number;
  predictedTerminalValueProxy: number;
  predictedProfitProxy: number;
  profitToRiskRatioProxy: number;
  valueBucket: string;
  valueBucketExplanation: string;
};

export type ScenarioResponse = {
  forecast: {
    predictedHighFromOpenPct: number;
    predictedLowFromOpenPct: number;
    predictedHighPrice: number;
    predictedLowPrice: number;
  };
  intradayState: {
    currentMovePct: number;
    highMovePct: number;
    lowMovePct: number;
    touchPrice: number;
    touchConfirmed: boolean;
    touchConfirmationSource: string;
    touchConsistencyLabel: string;
  };
  regimeContext: {
    weekday: string;
    vixRegime: string;
    rangeRegime: string;
    gapRegime: string;
  };
  decisionSummary: {
    upside: DecisionSummaryItem[];
    downside: DecisionSummaryItem[];
  };
  featuredPlaybooks: {
    upside: FeaturedPlaybook;
    downside: FeaturedPlaybook;
  };
  touchTables: {
    upside: TouchTableRow[];
    downside: TouchTableRow[];
  };
  continuationTables: {
    upside: ContinuationTableRow[];
    downside: ContinuationTableRow[];
  };
  verticalStrategy: {
    mode: string;
    pricingProvenance: {
      pricingMode: string;
      reanchored: boolean;
      filteredCandidateCount: number;
      sourceSnapshotCount: number;
      maxSnapshotGapPct: number | null;
    };
    summary: VerticalSummary | null;
    ranked: Record<string, unknown>[];
    notes: string[];
  };
  modelConfig: BootstrapResponse["modelConfig"];
};

export async function getBootstrap(): Promise<BootstrapResponse> {
  const response = await fetch("/api/v1/bootstrap");
  if (!response.ok) {
    throw new Error(`Bootstrap failed: ${response.status}`);
  }
  return response.json();
}

export async function postScenario(payload: ScenarioRequest): Promise<ScenarioResponse> {
  const response = await fetch("/api/v1/scenario", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`Scenario failed: ${response.status}`);
  }
  return response.json();
}
