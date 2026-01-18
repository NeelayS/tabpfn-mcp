"""MCP Server for TabPFN regression and classification on custom datasets."""

import csv
import os
from typing import Any

import numpy as np
from fastmcp import FastMCP
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from tabpfn_client import TabPFNClassifier, TabPFNRegressor, get_access_token, set_access_token

# Initialize the MCP server
mcp = FastMCP("TabPFN")

# Set up authentication: use environment variable if available, otherwise trigger interactive login
if token := os.environ.get("TABPFN_TOKEN"):
    set_access_token(token)
else:
    # This will use cached credentials or trigger browser-based login if needed
    get_access_token()


def _validate_file_path(file_path: str, description: str) -> str | None:
    """Validate that a file exists and return an error message if not."""
    if not os.path.exists(file_path):
        return f"{description} not found: {file_path}"
    if not os.path.isfile(file_path):
        return f"{description} is not a file: {file_path}"
    return None


def _is_header_row(row: list[str]) -> bool:
    """Check if a row is a header row by verifying all cells are non-numeric strings."""
    for cell in row:
        try:
            float(cell)
            return False
        except ValueError:
            pass
    return True


def _parse_csv(
    csv_path: str, has_labels: bool = True
) -> tuple[list[list[float]], list[float] | None]:
    """
    Parse a CSV file.

    Args:
        csv_path: Path to the CSV file
        has_labels: If True, assumes last column is y values. If False, treats all columns as X features.

    Returns:
        (X, y) where y is None if has_labels=False or if no labels are present
    """
    X = []
    y = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        first_row = next(reader)

        if not _is_header_row(first_row):
            if has_labels and len(first_row) > 1:
                X.append([float(v) for v in first_row[:-1]])
                y.append(float(first_row[-1]))
            else:
                X.append([float(v) for v in first_row])

        for row in reader:
            if len(row) == 0:
                continue

            if has_labels and len(row) > 1:
                X.append([float(v) for v in row[:-1]])
                y.append(float(row[-1]))
            else:
                X.append([float(v) for v in row])

    return X, y if y else None


def _detect_test_has_labels(test_csv_path: str, n_train_features: int) -> bool:
    """
    Automatically determine if test data has labels by comparing column counts.

    Args:
        test_csv_path: Path to the test CSV file
        n_train_features: Number of features in training data

    Returns:
        True if test data has labels, False otherwise
    """
    with open(test_csv_path) as f:
        reader = csv.reader(f)
        first_row = next(reader)
        if _is_header_row(first_row):
            data_row = next(reader, None)
            n_test_cols = len(data_row) if data_row else len(first_row)
        else:
            n_test_cols = len(first_row)

        return n_test_cols != n_train_features


def _to_list(data: Any) -> list:
    """Convert numpy array or iterable to list."""
    if hasattr(data, "tolist"):
        return data.tolist()
    return list(data)


def _write_predictions_csv(
    output_csv_path: str,
    predictions: list,
    y_test: list[float] | None = None,
    probabilities: list | None = None,
    classes: list | None = None,
) -> None:
    """Write predictions to CSV file."""
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["prediction"]
        if probabilities:
            if classes:
                header.extend([f"prob_class_{cls}" for cls in classes])
            else:
                header.extend([f"prob_{i}" for i in range(len(probabilities[0]))])
        if y_test is not None and len(y_test) > 0:
            header.append("true_value")

        writer.writerow(header)

        for i, pred in enumerate(predictions):
            row = [pred]
            if probabilities:
                row.extend(probabilities[i])
            if y_test is not None and len(y_test) > 0:
                row.append(y_test[i])
            writer.writerow(row)


def _parse_training_data(
    training_csv_path: str,
    dtype: type = np.float32,
) -> tuple[np.ndarray, np.ndarray, int] | dict[str, Any]:
    """
    Parse and validate training data.

    Returns:
        Tuple of (X_train, y_train, n_features) on success, or error dict on failure.
    """
    if error := _validate_file_path(training_csv_path, "Training CSV"):
        return {"error": error, "status": "failed"}

    X_train, y_train = _parse_csv(training_csv_path, has_labels=True)

    if not X_train:
        return {"error": "Training CSV is empty", "status": "failed"}

    if y_train is None or len(y_train) == 0:
        return {
            "error": "Training CSV must contain target values in the last column",
            "status": "failed",
        }

    X_train_array = np.array(X_train, dtype=np.float32)
    y_train_array = np.array(y_train, dtype=dtype)

    return X_train_array, y_train_array, X_train_array.shape[1]


def _parse_test_data(
    test_csv_path: str,
    n_train_features: int,
    dtype: type = np.float32,
) -> tuple[np.ndarray, list[float] | None] | dict[str, Any]:
    """
    Parse and validate test data.

    Returns:
        Tuple of (X_test, y_test) on success, or error dict on failure.
    """
    if error := _validate_file_path(test_csv_path, "Test CSV"):
        return {"error": error, "status": "failed"}

    test_has_labels = _detect_test_has_labels(test_csv_path, n_train_features)
    X_test, y_test = _parse_csv(test_csv_path, has_labels=test_has_labels)

    if not X_test:
        return {"error": "Test CSV is empty", "status": "failed"}

    X_test_array = np.array(X_test, dtype=np.float32)

    if y_test is not None:
        y_test = [dtype(v) for v in y_test]

    return X_test_array, y_test


@mcp.tool
def train_and_predict_regression(
    training_csv_path: str,
    output_csv_path: str,
    test_csv_path: str | None = None,
) -> dict[str, Any]:
    """
    Train a TabPFN regressor on training data and make predictions.

    Args:
        training_csv_path: Path to CSV file with training data. Last column should be y values.
        output_csv_path: Path to write predictions to CSV.
        test_csv_path: Optional path to CSV file with test data. The function automatically
                      detects whether test data has labels by comparing the number of columns
                      with the training data.

    Returns:
        Dictionary containing metrics (if test labels available) and metadata
    """
    try:
        # Parse training data
        result = _parse_training_data(training_csv_path, dtype=np.float32)
        if isinstance(result, dict):
            return result
        X_train_array, y_train_array, n_train_features = result

        # Train model
        model = TabPFNRegressor()
        model.fit(X_train_array, y_train_array)

        response = {
            "status": "success",
            "n_train_samples": len(X_train_array),
            "n_features": n_train_features,
            "training_targets_mean": float(np.mean(y_train_array)),
            "training_targets_std": float(np.std(y_train_array)),
        }

        if not test_csv_path:
            response["message"] = "No test data provided. Only training completed."
            return response

        # Parse test data
        result = _parse_test_data(test_csv_path, n_train_features, dtype=np.float32)
        if isinstance(result, dict):
            return result
        X_test_array, y_test = result

        # Make predictions
        predictions = model.predict(X_test_array)
        predictions_list = _to_list(predictions)

        response["n_test_samples"] = len(X_test_array)

        # Compute metrics if test labels available
        if y_test is not None and len(y_test) > 0:
            y_test_array = np.array(y_test, dtype=np.float32)

            mse = float(mean_squared_error(y_test_array, predictions))
            response["metrics"] = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_test_array, predictions)),
                "r2": float(r2_score(y_test_array, predictions)),
            }

        _write_predictions_csv(output_csv_path, predictions_list, y_test)
        response["output_file"] = output_csv_path

        return response

    except Exception as e:
        return {"error": f"Regression failed: {str(e)}", "status": "failed"}


@mcp.tool
def train_and_predict_classification(
    training_csv_path: str,
    output_csv_path: str,
    test_csv_path: str | None = None,
) -> dict[str, Any]:
    """
    Train a TabPFN classifier on training data and make predictions.

    Args:
        training_csv_path: Path to CSV file with training data. Last column should be class labels.
        output_csv_path: Path to write predictions to CSV.
        test_csv_path: Optional path to CSV file with test data. The function automatically
                      detects whether test data has labels by comparing the number of columns
                      with the training data.

    Returns:
        Dictionary containing metrics (if test labels available) and metadata
    """
    try:
        # Parse training data
        result = _parse_training_data(training_csv_path, dtype=np.int32)
        if isinstance(result, dict):
            return result
        X_train_array, y_train_array, n_train_features = result

        unique_classes = np.unique(y_train_array).tolist()
        n_classes = len(unique_classes)

        # Train model
        model = TabPFNClassifier()
        model.fit(X_train_array, y_train_array)

        response = {
            "status": "success",
            "n_train_samples": len(X_train_array),
            "n_features": n_train_features,
            "n_classes": n_classes,
            "classes": unique_classes,
        }

        if not test_csv_path:
            response["message"] = "No test data provided. Only training completed."
            return response

        # Parse test data
        result = _parse_test_data(test_csv_path, n_train_features, dtype=np.int32)
        if isinstance(result, dict):
            return result
        X_test_array, y_test = result

        # Make predictions
        predictions = model.predict(X_test_array)
        predictions_list = _to_list(predictions)
        probabilities = model.predict_proba(X_test_array)
        probabilities_list = _to_list(probabilities)

        response["n_test_samples"] = len(X_test_array)

        # Compute metrics if test labels available
        if y_test is not None and len(y_test) > 0:
            y_test_array = np.array(y_test, dtype=np.int32)

            is_binary = n_classes == 2
            average = "binary" if is_binary else "weighted"

            response["metrics"] = {
                "accuracy": float(accuracy_score(y_test_array, predictions)),
                "log_loss": float(log_loss(y_test_array, probabilities)),
                "precision": float(
                    precision_score(y_test_array, predictions, average=average, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test_array, predictions, average=average, zero_division=0)
                ),
                "f1": float(f1_score(y_test_array, predictions, average=average, zero_division=0)),
            }

        _write_predictions_csv(
            output_csv_path, predictions_list, y_test, probabilities_list, unique_classes
        )
        response["output_file"] = output_csv_path

        return response

    except Exception as e:
        return {"error": f"Classification failed: {str(e)}", "status": "failed"}


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
