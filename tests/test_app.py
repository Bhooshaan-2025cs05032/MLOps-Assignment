"""
Unit tests for api/app.py — targets 100 % line coverage.

Coverage map
------------
GET  /          : health_check — returns {"status": "API is running"}
GET  /metrics   : Prometheus metrics endpoint exposed by Instrumentator
POST /predict
  success path  : all 13 fields present, predict() returns a result dict,
                  prediction_counter.inc() is called exactly once
  missing field : one field omitted → 400 with "Missing fields" detail,
                  prediction_counter.inc() is NOT called
  all missing   : empty JSON body → 400
  HTTPException : predict() raises HTTPException → re-raised unchanged (95-96)
  generic error : predict() raises RuntimeError → wrapped in 500 (98-100)
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import app


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    return TestClient(app)


_FULL_PAYLOAD = {
    "age": 40.0,
    "sex": 0,
    "cp": 1,
    "trestbps": 110.0,
    "chol": 180.0,
    "fbs": 0,
    "restecg": 0,
    "thalach": 170.0,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 2,
    "ca": 0.0,
    "thal": 2,
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_returns_200_and_status_message(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "API is running"}


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:

    def test_metrics_endpoint_is_exposed(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "model_predictions_total" in response.text


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

class TestPredictEndpoint:

    def test_valid_payload_returns_success_result(self, client):
        with patch("api.app.predict") as mock_predict:
            mock_predict.return_value = {"prediction": 0, "confidence": 0.82}

            response = client.post("/predict", json=_FULL_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"] == {"prediction": 0, "confidence": 0.82}

    def test_prediction_counter_incremented_on_success(self, client):
        """prediction_counter.inc() must be called exactly once per successful prediction."""
        with patch("api.app.predict") as mock_predict, \
             patch("api.app.prediction_counter") as mock_counter:
            mock_predict.return_value = {"prediction": 1, "confidence": 0.91}

            response = client.post("/predict", json=_FULL_PAYLOAD)

        assert response.status_code == 200
        mock_counter.inc.assert_called_once()

    def test_prediction_counter_not_incremented_on_missing_fields(self, client):
        """Counter must NOT be incremented when validation fails (400 path)."""
        payload = {k: v for k, v in _FULL_PAYLOAD.items() if k != "age"}

        with patch("api.app.prediction_counter") as mock_counter:
            response = client.post("/predict", json=payload)

        assert response.status_code == 400
        mock_counter.inc.assert_not_called()

    def test_missing_single_field_returns_400(self, client):
        payload = {k: v for k, v in _FULL_PAYLOAD.items() if k != "age"}

        response = client.post("/predict", json=payload)

        assert response.status_code == 400
        assert "Missing fields" in response.json()["detail"]
        assert "age" in response.json()["detail"]

    def test_empty_body_returns_400(self, client):
        response = client.post("/predict", json={})

        assert response.status_code == 400
        assert "Missing fields" in response.json()["detail"]

    def test_predict_raising_http_exception_is_re_raised(self, client):
        """Covers 'except HTTPException: raise' (lines 95-96)."""
        with patch("api.app.predict") as mock_predict:
            mock_predict.side_effect = HTTPException(
                status_code=422, detail="Upstream validation error"
            )

            response = client.post("/predict", json=_FULL_PAYLOAD)

        assert response.status_code == 422
        assert "Upstream validation error" in response.json()["detail"]

    def test_predict_raising_generic_exception_returns_500(self, client):
        """Covers 'except Exception as e: raise HTTPException(500)' (lines 98-100)."""
        with patch("api.app.predict") as mock_predict:
            mock_predict.side_effect = RuntimeError("Model exploded")

            response = client.post("/predict", json=_FULL_PAYLOAD)

        assert response.status_code == 500
        assert "Model exploded" in response.json()["detail"]

