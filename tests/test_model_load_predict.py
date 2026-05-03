"""
Unit tests for api/model_load_predict.py — targets 100 % line coverage.

Coverage map
------------
load_model_from_mlflow  : success path; exception → returns None; custom args;
                          MLFLOW_TRACKING_URI env var used when set; local
                          fallback URI used when env var absent
load_model_from_local   : success path (real tmp file); FileNotFoundError path
load_model              : MLflow succeeds → returned directly; MLflow returns
                          None → falls back to local
get_model               : lazy load on first call; cached on subsequent calls;
                          pre-existing singleton returned unchanged
predict                 : model with predict_proba; model without predict_proba;
                          model obtained from singleton; exception propagates;
                          confidence rounding to 4 decimal places
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import api.model_load_predict as mlp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the module-level singleton is cleared before and after each test."""
    mlp._model_singleton = None
    yield
    mlp._model_singleton = None


_SAMPLE_DATA = {
    "age": 40, "sex": 0, "cp": 1, "trestbps": 110, "chol": 180,
    "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
    "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2,
}


# ---------------------------------------------------------------------------
# load_model_from_mlflow
# ---------------------------------------------------------------------------

class TestLoadModelFromMlflow:

    @patch("api.model_load_predict.mlflow")
    def test_returns_model_on_success(self, mock_mlflow):
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model

        result = mlp.load_model_from_mlflow()

        assert result is mock_model
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.sklearn.load_model.assert_called_once()

    @patch("api.model_load_predict.mlflow")
    def test_returns_none_on_exception(self, mock_mlflow):
        mock_mlflow.sklearn.load_model.side_effect = Exception("Registry unavailable")

        result = mlp.load_model_from_mlflow()

        assert result is None

    @patch("api.model_load_predict.mlflow")
    def test_model_uri_contains_custom_name_and_stage(self, mock_mlflow):
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        mlp.load_model_from_mlflow(
            mlruns_path=Path("/custom/mlruns"),
            model_name="custom_model",
            model_stage="Production",
        )

        called_uri = mock_mlflow.sklearn.load_model.call_args[0][0]
        assert "custom_model" in called_uri
        assert "Production" in called_uri

    @patch("api.model_load_predict.mlflow")
    def test_uses_mlflow_tracking_uri_env_var_when_set(self, mock_mlflow):
        """When MLFLOW_TRACKING_URI is set in env, it overrides the local path."""
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://remote.mlflow/test"}):
            mlp.load_model_from_mlflow(mlruns_path=Path("/any/path"))

        called_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert called_uri == "https://remote.mlflow/test"

    @patch("api.model_load_predict.mlflow")
    def test_falls_back_to_local_uri_when_env_var_absent(self, mock_mlflow):
        """When MLFLOW_TRACKING_URI is NOT set, the file:// local URI is used."""
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        env_without_key = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
        with patch.dict(os.environ, env_without_key, clear=True):
            mlp.load_model_from_mlflow(mlruns_path=Path("/my/mlruns"))

        called_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert called_uri.startswith("file:///")
        assert "my/mlruns" in called_uri


# ---------------------------------------------------------------------------
# load_model_from_local
# ---------------------------------------------------------------------------

class TestLoadModelFromLocal:

    @patch("api.model_load_predict.joblib")
    def test_returns_model_when_file_exists(self, mock_joblib, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.touch()           # file must exist for Path.exists() check

        mock_model = MagicMock()
        mock_joblib.load.return_value = mock_model

        result = mlp.load_model_from_local(model_path)

        assert result is mock_model
        mock_joblib.load.assert_called_once_with(model_path)

    def test_raises_file_not_found_when_missing(self):
        missing_path = Path("/does/not/exist/model.pkl")

        with pytest.raises(FileNotFoundError, match="Local model file not found"):
            mlp.load_model_from_local(missing_path)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:

    @patch("api.model_load_predict.load_model_from_mlflow")
    def test_returns_mlflow_model_when_available(self, mock_from_mlflow):
        mock_model = MagicMock()
        mock_from_mlflow.return_value = mock_model

        result = mlp.load_model(
            mlruns_path=Path("/mlruns"),
            local_model_path=Path("/models/m.pkl"),
        )

        assert result is mock_model

    @patch("api.model_load_predict.load_model_from_local")
    @patch("api.model_load_predict.load_model_from_mlflow")
    def test_falls_back_to_local_when_mlflow_returns_none(
        self, mock_from_mlflow, mock_from_local
    ):
        mock_from_mlflow.return_value = None
        local_model = MagicMock()
        mock_from_local.return_value = local_model

        result = mlp.load_model(
            mlruns_path=Path("/mlruns"),
            local_model_path=Path("/models/m.pkl"),
        )

        assert result is local_model
        mock_from_local.assert_called_once()


# ---------------------------------------------------------------------------
# get_model (singleton)
# ---------------------------------------------------------------------------

class TestGetModel:

    @patch("api.model_load_predict.load_model")
    def test_loads_model_on_first_call(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = mlp.get_model()

        assert result is mock_model
        mock_load_model.assert_called_once()

    @patch("api.model_load_predict.load_model")
    def test_returns_same_instance_on_second_call(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        first = mlp.get_model()
        second = mlp.get_model()

        assert first is second
        assert mock_load_model.call_count == 1   # loaded only once

    def test_returns_pre_existing_singleton_without_loading(self):
        existing_model = MagicMock()
        mlp._model_singleton = existing_model

        with patch("api.model_load_predict.load_model") as mock_load_model:
            result = mlp.get_model()

        assert result is existing_model
        mock_load_model.assert_not_called()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_prediction_and_confidence_with_predict_proba(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.15, 0.85]]

        result = mlp.predict(_SAMPLE_DATA, model=mock_model)

        assert result["prediction"] == 1
        assert result["confidence"] == pytest.approx(0.85, abs=1e-4)

    def test_confidence_is_none_when_predict_proba_absent(self):
        mock_model = MagicMock(spec=["predict"])   # no predict_proba attribute
        mock_model.predict.return_value = [0]

        result = mlp.predict(_SAMPLE_DATA, model=mock_model)

        assert result["prediction"] == 0
        assert result["confidence"] is None

    @patch("api.model_load_predict.get_model")
    def test_uses_singleton_when_model_not_provided(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.7, 0.3]]
        mock_get_model.return_value = mock_model

        result = mlp.predict(_SAMPLE_DATA)

        mock_get_model.assert_called_once()
        assert result["prediction"] == 0

    def test_exception_from_model_propagates(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("Unexpected feature shape")

        with pytest.raises(ValueError, match="Unexpected feature shape"):
            mlp.predict(_SAMPLE_DATA, model=mock_model)

    def test_confidence_rounded_to_four_decimal_places(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.123456789, 0.876543211]]

        result = mlp.predict(_SAMPLE_DATA, model=mock_model)

        assert result["confidence"] == pytest.approx(0.8765, abs=1e-4)
