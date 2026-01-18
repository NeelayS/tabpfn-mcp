"""Tests for the TabPFN MCP server."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tabpfn_mcp.server import (
    _is_header_row,
    _parse_csv,
    _validate_file_path,
)
from tabpfn_mcp.server import (
    train_and_predict_classification as _classification_tool,
)
from tabpfn_mcp.server import (
    train_and_predict_regression as _regression_tool,
)

# Get underlying functions from MCP tool wrappers
train_and_predict_regression = _regression_tool.fn
train_and_predict_classification = _classification_tool.fn

TEST_DATA_DIR = Path(__file__).parent / "data"
REGRESSION_DIR = TEST_DATA_DIR / "regression"
CLASSIFICATION_DIR = TEST_DATA_DIR / "classification"


def test_is_header_row():
    assert _is_header_row(["feature1", "feature2", "target"]) is True
    assert _is_header_row(["1.5", "2.3", "3.1"]) is False
    assert _is_header_row(["feature1", "2.3", "target"]) is False


def test_parse_csv():
    X, y = _parse_csv(str(REGRESSION_DIR / "train.csv"), has_labels=True)
    assert len(X) == 20 and len(y) == 20 and len(X[0]) == 4

    X, y = _parse_csv(str(REGRESSION_DIR / "test.csv"), has_labels=False)
    assert len(X) == 5 and y is None


def test_validate_file_path():
    assert _validate_file_path(str(REGRESSION_DIR / "train.csv"), "CSV") is None
    assert "not found" in _validate_file_path("/nonexistent/path.csv", "CSV")


@patch("tabpfn_mcp.server.TabPFNRegressor")
def test_regression_with_metrics(mock_regressor_class):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([50.0, 45.0, 55.0, 42.0, 60.0])
    mock_regressor_class.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name

    try:
        result = train_and_predict_regression(
            str(REGRESSION_DIR / "train.csv"),
            output_path,
            str(REGRESSION_DIR / "test_with_target.csv"),
        )

        assert result["status"] == "success"
        assert result["n_train_samples"] == 20
        assert result["n_test_samples"] == 5
        assert all(k in result["metrics"] for k in ["mse", "rmse", "mae", "r2"])
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


@patch("tabpfn_mcp.server.TabPFNClassifier")
def test_classification_with_metrics(mock_classifier_class):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])
    mock_model.predict_proba.return_value = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
            [0.85, 0.1, 0.05],
            [0.15, 0.75, 0.1],
        ]
    )
    mock_classifier_class.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name

    try:
        result = train_and_predict_classification(
            str(CLASSIFICATION_DIR / "train.csv"),
            output_path,
            str(CLASSIFICATION_DIR / "test_with_labels.csv"),
        )

        assert result["status"] == "success"
        assert result["n_classes"] == 3
        assert all(k in result["metrics"] for k in ["accuracy", "precision", "recall", "f1"])
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_nonexistent_file_error():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name

    try:
        result = train_and_predict_regression("/nonexistent/train.csv", output_path)
        assert result["status"] == "failed"
        assert "not found" in result["error"]
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
