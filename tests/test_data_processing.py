"""
Unit tests for src/data_processing.py — targets 100 % line coverage.

Coverage map
------------
load_data        : mocked fetch_ucirepo, checks shape & column names
clean_data       : '?' replacement, null detection (both branches), median
                   imputation, target binarisation
save_data        : tmp_path integration, verifies CSV on disk
run_pipeline     : mocks the three sub-functions and the default-path variant
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_processing import clean_data, load_data, run_pipeline, save_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_numeric_df():
    """Clean numeric DataFrame with both target classes present."""
    return pd.DataFrame(
        {
            "age": [63.0, 40.0, 55.0],
            "sex": [1, 0, 1],
            "cp": [3, 2, 1],
            "trestbps": [145.0, 130.0, 120.0],
            "chol": [233.0, 200.0, 180.0],
            "fbs": [1, 0, 0],
            "restecg": [0, 1, 0],
            "thalach": [150.0, 160.0, 170.0],
            "exang": [0, 0, 1],
            "oldpeak": [2.3, 1.5, 0.0],
            "slope": [0, 1, 2],
            "ca": [0.0, 1.0, 0.0],
            "thal": [1, 2, 3],
            "target": [0, 1, 2],
        }
    )


def _make_df_with_question_marks():
    """DataFrame containing '?' placeholders and a NaN-inducing numeric column."""
    return pd.DataFrame(
        {
            "age": [63, "?", 40],
            "sex": [1, 0, 1],
            "cp": [3, 2, 1],
            "trestbps": [145, 130, "?"],
            "chol": [233, 200, 180],
            "fbs": [1, 0, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 160, 170],
            "exang": [0, 0, 1],
            "oldpeak": [2.3, 1.5, 0.0],
            "slope": [0, 1, 2],
            "ca": [0, 1, 0],
            "thal": [1, 2, 3],
            "target": [0, 2, 1],
        }
    )


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    @patch("src.data_processing.fetch_ucirepo")
    def test_returns_dataframe_with_correct_columns(self, mock_fetch):
        features = pd.DataFrame(
            {
                "age": [63],
                "sex": [1],
                "cp": [3],
                "trestbps": [145],
                "chol": [233],
                "fbs": [1],
                "restecg": [0],
                "thalach": [150],
                "exang": [0],
                "oldpeak": [2.3],
                "slope": [0],
                "ca": [0],
                "thal": [1],
            }
        )
        targets = pd.DataFrame({"target": [0]})

        mock_dataset = MagicMock()
        mock_dataset.data.features = features
        mock_dataset.data.targets = targets
        mock_fetch.return_value = mock_dataset

        df = load_data()

        mock_fetch.assert_called_once_with(id=45)
        assert df.shape == (1, 14)
        expected_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
        ]
        assert list(df.columns) == expected_cols


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

class TestCleanData:

    def test_question_marks_replaced_and_nulls_filled(self):
        df = _make_df_with_question_marks()
        result = clean_data(df)

        # No NaNs should remain after median fill
        assert result.isnull().sum().sum() == 0

    def test_null_branch_logged_when_nulls_present(self):
        """Covers the 'if not cols_with_nulls.empty:' True branch."""
        df = _make_df_with_question_marks()
        result = clean_data(df)
        # The '?' rows were filled with column medians
        assert result["age"].iloc[1] == pytest.approx(51.5, abs=0.1)

    def test_no_question_marks_no_nulls_branch(self):
        """Covers the 'else' (no nulls) branch."""
        df = _make_numeric_df()
        result = clean_data(df)
        assert result.isnull().sum().sum() == 0

    def test_target_binarized_positive_becomes_one(self):
        df = _make_df_with_question_marks()
        result = clean_data(df)
        # Original target was [0, 2, 1] → binarised to [0, 1, 1]
        assert result["target"].tolist() == [0, 1, 1]

    def test_target_binarized_zero_stays_zero(self):
        df = _make_numeric_df()
        result = clean_data(df)
        assert set(result["target"].unique()).issubset({0, 1})

    def test_fills_nan_with_column_median(self):
        df = pd.DataFrame(
            {
                "age": [10.0, np.nan, 30.0],
                "sex": [1, 0, 1],
                "cp": [1, 2, 3],
                "trestbps": [100.0, 110.0, 120.0],
                "chol": [200.0, 210.0, 220.0],
                "fbs": [0, 1, 0],
                "restecg": [0, 0, 1],
                "thalach": [150.0, 155.0, 160.0],
                "exang": [0, 0, 1],
                "oldpeak": [0.0, 1.0, 2.0],
                "slope": [0, 1, 2],
                "ca": [0.0, 1.0, 0.0],
                "thal": [1, 2, 3],
                "target": [0, 0, 1],
            }
        )
        result = clean_data(df)
        assert result.isnull().sum().sum() == 0
        # median of [10.0, 30.0] = 20.0
        assert result["age"].iloc[1] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------

class TestSaveData:

    def test_creates_directory_and_writes_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output_path = str(tmp_path / "subdir" / "output.csv")

        save_data(df, output_path)

        assert os.path.exists(output_path)
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(df, loaded)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:

    @patch("src.data_processing.save_data")
    @patch("src.data_processing.clean_data")
    @patch("src.data_processing.load_data")
    def test_calls_all_steps_with_custom_path(
        self, mock_load, mock_clean, mock_save
    ):
        raw_df = MagicMock()
        clean_df = MagicMock()
        mock_load.return_value = raw_df
        mock_clean.return_value = clean_df

        run_pipeline(output_path="/tmp/test_output.csv")

        mock_load.assert_called_once()
        mock_clean.assert_called_once_with(raw_df)
        mock_save.assert_called_once_with(clean_df, "/tmp/test_output.csv")

    @patch("src.data_processing.save_data")
    @patch("src.data_processing.clean_data")
    @patch("src.data_processing.load_data")
    def test_uses_default_output_path_when_not_specified(
        self, mock_load, mock_clean, mock_save
    ):
        mock_load.return_value = MagicMock()
        mock_clean.return_value = MagicMock()

        run_pipeline()  # default path

        mock_save.assert_called_once()
        actual_path = mock_save.call_args[0][1]
        assert "heart_clean.csv" in actual_path
