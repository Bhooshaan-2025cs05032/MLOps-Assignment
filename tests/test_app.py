"""
Unit tests for api/app.py — targets 100 % line coverage.

Coverage map
------------
GET  /          : health_check — returns {"status": "API is running"}
POST /predict
  success path  : all 13 fields present, predict() returns a result dict
  missing field : one field omitted → 400 with "Missing fields" detail
  all missing   : empty JSON body → 400
  HTTPException : predict() raises HTTPException → re-raised unchanged (83-84)
  generic error : predict() raises RuntimeError → wrapped in 500 (86-88)
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
        """Covers 'except HTTPException: raise' (lines 83-84)."""
        with patch("api.app.predict") as mock_predict:
            mock_predict.side_effect = HTTPException(
                status_code=422, detail="Upstream validation error"
            )

            response = client.post("/predict", json=_FULL_PAYLOAD)

        assert response.status_code == 422
        assert "Upstream validation error" in response.json()["detail"]

    def test_predict_raising_generic_exception_returns_500(self, client):
        """Covers 'except Exception as e: raise HTTPException(500)' (lines 86-88)."""
        with patch("api.app.predict") as mock_predict:
            mock_predict.side_effect = RuntimeError("Model exploded")

            response = client.post("/predict", json=_FULL_PAYLOAD)

        assert response.status_code == 500
        assert "Model exploded" in response.json()["detail"]
