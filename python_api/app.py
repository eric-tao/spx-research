from __future__ import annotations

from fastapi import FastAPI

from python_api.schemas import ScenarioRequest
from python_api.service import ApiConfig, build_bootstrap_payload, build_scenario_payload, get_app_state

app = FastAPI(title="SPX 0DTE Planner API", version="0.1.0")
CONFIG = ApiConfig()


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    state = get_app_state(CONFIG)
    return {
        "ok": True,
        "modelLoaded": state is not None,
        "verticalInputsAvailable": bool(state.vertical_inputs_available),
    }


@app.get("/api/v1/bootstrap")
def bootstrap() -> dict[str, object]:
    state = get_app_state(CONFIG)
    return build_bootstrap_payload(state, CONFIG)


@app.post("/api/v1/scenario")
def scenario(request: ScenarioRequest) -> dict[str, object]:
    state = get_app_state(CONFIG)
    return build_scenario_payload(state, CONFIG, request)
