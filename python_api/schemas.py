from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PriorDayOverridesModel(BaseModel):
    spxOpen: float | None = None
    spxHigh: float | None = None
    spxLow: float | None = None
    spxClose: float | None = None
    vixOpen: float | None = None
    vixHigh: float | None = None
    vixLow: float | None = None
    vixClose: float | None = None


class TouchSelectionModel(BaseModel):
    touchedSide: Literal["upside_touch", "downside_touch"] = "upside_touch"
    touchedThresholdPct: float = Field(default=0.50, ge=0.0)


class VerticalSelectionModel(BaseModel):
    widthPoints: float = Field(default=10.0, gt=0.0)


class ValueBreakpointsModel(BaseModel):
    strongProfitThreshold: float = 1.50
    strongRatioThreshold: float = 0.75
    watchProfitThreshold: float = 0.75
    watchRatioThreshold: float = 0.35


class ScenarioRequest(BaseModel):
    predictionDate: str
    spxOpen: float
    vixOpen: float
    currentSpot: float
    checkpointTime: str = "12:00"
    highSoFar: float
    lowSoFar: float
    selectedEvents: list[str] = Field(default_factory=list)
    priorDayOverrides: PriorDayOverridesModel = Field(default_factory=PriorDayOverridesModel)
    touchSelection: TouchSelectionModel = Field(default_factory=TouchSelectionModel)
    verticalSelection: VerticalSelectionModel = Field(default_factory=VerticalSelectionModel)
    valueBreakpoints: ValueBreakpointsModel = Field(default_factory=ValueBreakpointsModel)
