"""
Unit tests for src/train_model_and_log.py — targets 100 % line coverage.

Coverage map
------------
load_data              : reads a real CSV from tmp_path
split_data             : verifies 80/20 proportions and target column removal
build_preprocessor     : checks ColumnTransformer shape and transformer names
evaluate               : mocked model returns deterministic predictions/probs
format_run_name        : pure string formatting, no I/O
train_and_log          : GridSearchCV mocked; two models, both branches of the
                         'mean_train_score present/absent' conditional covered
register_best_model    : three scenarios — happy path, no experiment, no runs;
                         also covers the missing 'model_type' tag default
run_pipeline           : mocks every sub-function to exercise the entry point
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from src.train_model_and_log import (
    CATEGORICAL_COLS,
    build_preprocessor,
    evaluate,
    format_run_name,
    load_data,
    register_best_model,
    run_pipeline,
    split_data,
    train_and_log,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def _make_sample_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    data = {
        "age": rng.integers(30, 70, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(0, 4, n).astype(float),
        "trestbps": rng.integers(90, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(90, 200, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.uniform(0, 5, n),
        "slope": rng.integers(0, 3, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.integers(0, 4, n).astype(float),
        "target": rng.integers(0, 2, n),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:

    def test_reads_csv_correctly(self, tmp_path):
        df = _make_sample_df()
        csv_path = tmp_path / "heart_clean.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_data(csv_path)

        assert loaded.shape == df.shape
        assert list(loaded.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:

    def test_80_20_proportions(self):
        df = _make_sample_df(100)
        X_train, X_test, y_train, y_test = split_data(df)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_target_removed_from_features(self):
        df = _make_sample_df(50)
        X_train, X_test, y_train, y_test = split_data(df)

        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_total_rows_preserved(self):
        df = _make_sample_df(50)
        X_train, X_test, _, _ = split_data(df)
        assert len(X_train) + len(X_test) == 50


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:

    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer

        df = _make_sample_df()
        X = df.drop("target", axis=1)
        preprocessor = build_preprocessor(X)

        assert isinstance(preprocessor, ColumnTransformer)

    def test_transformer_names(self):
        df = _make_sample_df()
        X = df.drop("target", axis=1)
        preprocessor = build_preprocessor(X)

        names = [t[0] for t in preprocessor.transformers]
        assert "num" in names
        assert "cat" in names

    def test_categorical_columns_assigned_correctly(self):
        df = _make_sample_df()
        X = df.drop("target", axis=1)
        preprocessor = build_preprocessor(X)

        cat_transformer = next(t for t in preprocessor.transformers if t[0] == "cat")
        assert cat_transformer[2] == CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_returns_all_metric_keys(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.3, 0.7], [0.8, 0.2]]
        )

        X_test = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y_test = pd.Series([0, 1, 0, 1, 0])

        result = evaluate(mock_model, X_test, y_test)

        assert set(result.keys()) == {
            "accuracy", "precision", "recall", "roc_auc", "confusion_matrix"
        }

    def test_perfect_predictions_give_max_scores(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.3, 0.7], [0.8, 0.2]]
        )

        X_test = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y_test = pd.Series([0, 1, 0, 1, 0])

        result = evaluate(mock_model, X_test, y_test)

        assert result["accuracy"] == pytest.approx(1.0)
        assert result["roc_auc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_run_name
# ---------------------------------------------------------------------------

class TestFormatRunName:

    def test_single_param(self):
        result = format_run_name("logistic_regression", {"model__C": 1.0})
        assert result == "logistic_regression_C=1.0"

    def test_multiple_params_joined_with_underscore(self):
        result = format_run_name(
            "random_forest",
            {"model__n_estimators": 100, "model__max_depth": 5},
        )
        assert result.startswith("random_forest_")
        assert "n_estimators=100" in result
        assert "max_depth=5" in result

    def test_uses_last_part_of_double_underscore_key(self):
        result = format_run_name("m", {"a__b__key": "val"})
        assert "key=val" in result


# ---------------------------------------------------------------------------
# train_and_log
# ---------------------------------------------------------------------------

class TestTrainAndLog:
    """
    GridSearchCV is mocked entirely.

    Two mock grids are provided via side_effect so each model loop iteration
    receives a distinct object:
      - Grid 1 (logistic_regression) : cv_results_ includes 'mean_train_score'
        → covers the True branch of the conditional log
      - Grid 2 (random_forest)        : cv_results_ does NOT include it
        → covers the False branch
    """

    def _build_grid_mock(self, params, include_train_score: bool):
        n = len(params)
        mock_grid = MagicMock()
        cv_results = {
            "params": params,
            "mean_test_score": np.full(n, 0.8),
            "std_test_score": np.full(n, 0.02),
        }
        if include_train_score:
            cv_results["mean_train_score"] = np.full(n, 0.85)
        mock_grid.cv_results_ = cv_results
        mock_grid.best_estimator_ = MagicMock()
        mock_grid.best_params_ = params[0]
        return mock_grid

    @patch("src.train_model_and_log.mlflow")
    @patch("src.train_model_and_log.GridSearchCV")
    @patch("src.train_model_and_log.evaluate")
    def test_logs_cv_and_best_runs_for_both_models(
        self, mock_eval, mock_gscv_cls, mock_mlflow
    ):
        mock_eval.return_value = {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.82,
            "roc_auc": 0.85,
            "confusion_matrix": np.array([[5, 1], [1, 7]]),
        }

        grid1 = self._build_grid_mock(
            [{"model__C": 0.1}], include_train_score=True
        )
        grid2 = self._build_grid_mock(
            [{"model__n_estimators": 100, "model__max_depth": 5}],
            include_train_score=False,
        )
        mock_gscv_cls.side_effect = [grid1, grid2]

        df = _make_sample_df(40)
        X_train, X_test, y_train, y_test = split_data(df)
        preprocessor = build_preprocessor(X_train)

        train_and_log(
            X_train, X_test, y_train, y_test, preprocessor, Path("/tmp/mlruns")
        )

        mock_mlflow.set_tracking_uri.assert_called_once()
        # set_experiment called twice per model (CV + BEST) = 4 times total
        assert mock_mlflow.set_experiment.call_count == 4
        # mlflow.start_run used as context manager — called once per CV param
        # combination + once per best model = 1+1+1+1 = 4 times
        assert mock_mlflow.start_run.call_count == 4

    @patch("src.train_model_and_log.mlflow")
    @patch("src.train_model_and_log.GridSearchCV")
    @patch("src.train_model_and_log.evaluate")
    def test_mean_train_score_not_logged_when_absent(
        self, mock_eval, mock_gscv_cls, mock_mlflow
    ):
        """Cover False branch: mean_train_score absent → no metric logged."""
        mock_eval.return_value = {
            "accuracy": 0.80, "precision": 0.75, "recall": 0.82,
            "roc_auc": 0.85, "confusion_matrix": np.array([[5, 1], [1, 7]]),
        }

        # Both grids omit mean_train_score
        grid = self._build_grid_mock(
            [{"model__C": 1.0}], include_train_score=False
        )
        mock_gscv_cls.return_value = grid

        df = _make_sample_df(40)
        X_train, X_test, y_train, y_test = split_data(df)

        train_and_log(
            X_train, X_test, y_train, y_test,
            build_preprocessor(X_train), Path("/tmp/mlruns"),
        )

        logged_metrics = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        assert "mean_train_score" not in logged_metrics


    @patch("src.train_model_and_log.mlflow")
    @patch("src.train_model_and_log.GridSearchCV")
    @patch("src.train_model_and_log.evaluate")
    def test_uses_mlflow_tracking_uri_env_var_when_set(
        self, mock_eval, mock_gscv_cls, mock_mlflow
    ):
        """When MLFLOW_TRACKING_URI env var is set, it is used as tracking URI."""
        mock_eval.return_value = {
            "accuracy": 0.80, "precision": 0.75, "recall": 0.82,
            "roc_auc": 0.85, "confusion_matrix": np.array([[5, 1], [1, 7]]),
        }
        grid = self._build_grid_mock([{"model__C": 1.0}], include_train_score=False)
        mock_gscv_cls.return_value = grid

        df = _make_sample_df(40)
        X_train, X_test, y_train, y_test = split_data(df)

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://remote.mlflow/train"}):
            train_and_log(
                X_train, X_test, y_train, y_test,
                build_preprocessor(X_train), Path("/tmp/mlruns"),
            )

        called_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert called_uri == "https://remote.mlflow/train"

    @patch("src.train_model_and_log.mlflow")
    @patch("src.train_model_and_log.GridSearchCV")
    @patch("src.train_model_and_log.evaluate")
    def test_falls_back_to_local_uri_when_env_var_absent(
        self, mock_eval, mock_gscv_cls, mock_mlflow
    ):
        """When MLFLOW_TRACKING_URI is NOT set, file:// local URI is used."""
        mock_eval.return_value = {
            "accuracy": 0.80, "precision": 0.75, "recall": 0.82,
            "roc_auc": 0.85, "confusion_matrix": np.array([[5, 1], [1, 7]]),
        }
        grid = self._build_grid_mock([{"model__C": 1.0}], include_train_score=False)
        mock_gscv_cls.return_value = grid

        df = _make_sample_df(40)
        X_train, X_test, y_train, y_test = split_data(df)

        env_without_key = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
        with patch.dict(os.environ, env_without_key, clear=True):
            train_and_log(
                X_train, X_test, y_train, y_test,
                build_preprocessor(X_train), Path("/tmp/mlruns"),
            )

        called_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert called_uri.startswith("file:///")


# ---------------------------------------------------------------------------
# register_best_model
# ---------------------------------------------------------------------------

class TestRegisterBestModel:

    def _make_mock_client(self, tags=None, metrics=None):
        """Return a wired-up MlflowClient mock with one winning run."""
        mock_client = MagicMock()

        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_exp

        mock_run = MagicMock()
        mock_run.info.run_id = "run_abc"
        mock_run.data.tags = tags if tags is not None else {"model_type": "random_forest"}
        mock_run.data.metrics = metrics if metrics is not None else {"test_roc_auc": 0.92}
        mock_run.data.params = {"model__n_estimators": 100}
        mock_client.search_runs.return_value = [mock_run]
        return mock_client

    @patch("src.train_model_and_log.joblib")
    @patch("src.train_model_and_log.MlflowClient")
    @patch("src.train_model_and_log.mlflow")
    def test_happy_path_registers_and_saves(
        self, mock_mlflow, mock_client_cls, mock_joblib, tmp_path
    ):
        mock_client_cls.return_value = self._make_mock_client()
        mock_mv = MagicMock()
        mock_mv.name = "heart_disease_pred_model"
        mock_mv.version = "1"
        mock_mlflow.register_model.return_value = mock_mv
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        register_best_model(tmp_path / "mlruns", tmp_path / "models")

        mock_mlflow.register_model.assert_called_once()
        mock_joblib.dump.assert_called_once()

    @patch("src.train_model_and_log.joblib")
    @patch("src.train_model_and_log.MlflowClient")
    @patch("src.train_model_and_log.mlflow")
    def test_missing_model_type_tag_defaults_to_unknown(
        self, mock_mlflow, mock_client_cls, mock_joblib, tmp_path
    ):
        """Cover winner.data.tags.get(..., 'unknown') default."""
        mock_client_cls.return_value = self._make_mock_client(tags={})
        mock_mv = MagicMock()
        mock_mv.name = "heart_disease_pred_model"
        mock_mv.version = "1"
        mock_mlflow.register_model.return_value = mock_mv
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        register_best_model(tmp_path / "mlruns", tmp_path / "models")

        mock_mlflow.register_model.assert_called_once()

    @patch("src.train_model_and_log.MlflowClient")
    @patch("src.train_model_and_log.mlflow")
    def test_raises_when_experiment_not_found(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None

        with pytest.raises(RuntimeError, match="not found"):
            register_best_model(Path("/tmp/mlruns"), Path("/tmp/models"))

    @patch("src.train_model_and_log.MlflowClient")
    @patch("src.train_model_and_log.mlflow")
    def test_raises_when_no_runs_exist(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_client.search_runs.return_value = []

        with pytest.raises(RuntimeError, match="No runs found"):
            register_best_model(Path("/tmp/mlruns"), Path("/tmp/models"))

    @patch("src.train_model_and_log.joblib")
    @patch("src.train_model_and_log.MlflowClient")
    @patch("src.train_model_and_log.mlflow")
    def test_uses_mlflow_tracking_uri_env_var_when_set(
        self, mock_mlflow, mock_client_cls, mock_joblib, tmp_path
    ):
        """When MLFLOW_TRACKING_URI env var is set it overrides the local path."""
        mock_client_cls.return_value = self._make_mock_client()
        mock_mlflow.register_model.return_value = MagicMock(name="m", version="1")
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://remote.mlflow/reg"}):
            register_best_model(tmp_path / "mlruns", tmp_path / "models")

        called_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert called_uri == "https://remote.mlflow/reg"

    @patch("src.train_model_and_log.joblib")
    @patch("src.train_model_and_log.MlflowClient")
    @patch("src.train_model_and_log.mlflow")
    def test_falls_back_to_local_uri_when_env_var_absent(
        self, mock_mlflow, mock_client_cls, mock_joblib, tmp_path
    ):
        """When MLFLOW_TRACKING_URI is NOT set, file:// local URI is used."""
        mock_client_cls.return_value = self._make_mock_client()
        mock_mlflow.register_model.return_value = MagicMock(name="m", version="1")
        mock_mlflow.sklearn.load_model.return_value = MagicMock()

        env_without_key = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
        with patch.dict(os.environ, env_without_key, clear=True):
            register_best_model(tmp_path / "mlruns", tmp_path / "models")

        called_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert called_uri.startswith("file:///")



# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:

    @patch("src.train_model_and_log.register_best_model")
    @patch("src.train_model_and_log.train_and_log")
    @patch("src.train_model_and_log.build_preprocessor")
    @patch("src.train_model_and_log.split_data")
    @patch("src.train_model_and_log.load_data")
    def test_calls_all_steps(
        self,
        mock_load,
        mock_split,
        mock_build,
        mock_train,
        mock_register,
    ):
        df = _make_sample_df()
        mock_load.return_value = df

        X = df.drop("target", axis=1)
        y = df["target"]
        mock_split.return_value = (X, X.iloc[:5], y, y.iloc[:5])
        mock_build.return_value = MagicMock()

        run_pipeline(
            data_path=Path("/tmp/data.csv"),
            mlruns_path=Path("/tmp/mlruns"),
            models_dir=Path("/tmp/models"),
        )

        mock_load.assert_called_once_with(Path("/tmp/data.csv"))
        mock_split.assert_called_once_with(df)
        mock_build.assert_called_once()
        mock_train.assert_called_once()
        mock_register.assert_called_once()
