from fastapi.testclient import TestClient

from python_api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "modelLoaded" in payload


def test_bootstrap_endpoint():
    response = client.get("/api/v1/bootstrap")
    assert response.status_code == 200
    payload = response.json()
    assert payload["underlyingLabel"] == "SPX"
    assert "defaults" in payload
    assert "modelConfig" in payload


def test_scenario_endpoint():
    response = client.post(
        "/api/v1/scenario",
        json={
            "predictionDate": "2026-04-03",
            "spxOpen": 5248.35,
            "vixOpen": 21.40,
            "currentSpot": 5249.10,
            "checkpointTime": "13:15",
            "highSoFar": 5278.00,
            "lowSoFar": 5236.00,
            "selectedEvents": [],
            "touchSelection": {"touchedSide": "upside_touch", "touchedThresholdPct": 0.50},
            "verticalSelection": {"widthPoints": 10},
            "valueBreakpoints": {
                "strongProfitThreshold": 2.0,
                "strongRatioThreshold": 0.9,
                "watchProfitThreshold": 1.0,
                "watchRatioThreshold": 0.4,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "forecast" in payload
    assert "intradayState" in payload
    assert payload["intradayState"]["touchConfirmationSource"] in {
        "high_so_far",
        "low_so_far",
        "spot_only",
        "not_confirmed",
    }
    assert "verticalStrategy" in payload
    assert "pricingProvenance" in payload["verticalStrategy"]
