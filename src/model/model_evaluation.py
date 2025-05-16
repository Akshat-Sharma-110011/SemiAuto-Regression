"""
Model Evaluation Module.

This module evaluates the performance of trained regression models using various metrics
and stores the results in a YAML file. It also evaluates the optimized model if available.
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Metrics imports
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    max_error
)

# Import custom logger
import logging
from src.logger import section, configure_logger  # Configure logger

with open('intel.yaml', 'r') as f:
    config = yaml.safe_load(f)
    dataset_name = config['dataset_name']

# Configure logger
configure_logger()
logger = logging.getLogger("Model Evaluation")


def load_intel(intel_path: str = "intel.yaml") -> Dict[str, Any]:
    """
    Load the intelligence YAML file containing paths and configurations.

    Args:
        intel_path: Path to the intel YAML file

    Returns:
        Dictionary containing the loaded intel data
    """
    section(f"Loading Intel from {intel_path}", logger)
    try:
        with open(intel_path, "r") as f:
            intel = yaml.safe_load(f)
        logger.info(f"Successfully loaded intel from {intel_path}")
        return intel
    except Exception as e:
        logger.error(f"Failed to load intel file: {e}")
        raise


def load_model(model_path: str) -> Any:
    """
    Load a trained model from the specified path.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded model object
    """
    section(f"Loading Model from {model_path}", logger)
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_test_data(test_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the test dataset and separate features from target.

    Args:
        test_path: Path to the test dataset
        target_column: Name of the target column

    Returns:
        Tuple of (X_test, y_test)
    """
    section(f"Loading Test Data from {test_path}", logger)
    try:
        test_data = pd.read_csv(test_path)
        logger.info(f"Test data shape: {test_data.shape}")

        # Split features and target
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model using various regression metrics.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values

    Returns:
        Dictionary containing metric names and values
    """
    section("Evaluating Model Performance", logger)
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics (explicitly convert numpy values to native Python floats)
        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mean_squared_error": mse,
            "root_mean_squared_error": float(np.sqrt(mse)),
            "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
            "mean_absolute_percentage_error": float(mean_absolute_percentage_error(y_test, y_pred)),
            "explained_variance_score": float(explained_variance_score(y_test, y_pred)),
            "max_error": float(max_error(y_test, y_pred))
        }

        # Log metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")

        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: Dict[str, float], dataset_name: str, filename: str = "performance.yaml") -> str:
    """
    Save metrics to a YAML file in the reports/metrics directory.

    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
        filename: Name of the metrics file (default: performance.yaml)

    Returns:
        Path to the saved metrics file
    """
    section(f"Saving Performance Metrics to {filename}", logger)
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join("reports", "metrics", f"performance_{dataset_name}")
        os.makedirs(metrics_dir, exist_ok=True)

        # Add timestamp to metrics
        metrics["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert numpy values to native Python types to prevent YAML serialization issues
        cleaned_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) or isinstance(value, np.number):
                cleaned_metrics[key] = float(value)
            else:
                cleaned_metrics[key] = value

        # Define metrics file path
        metrics_file_path = os.path.join(metrics_dir, filename)

        # Save metrics to YAML
        with open(metrics_file_path, "w") as f:
            yaml.dump(cleaned_metrics, f, default_flow_style=False)

        logger.info(f"Metrics saved to {metrics_file_path}")
        return metrics_file_path
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


def update_intel(intel: Dict[str, Any], metrics_path: str, intel_path: str = "intel.yaml",
                 is_optimized: bool = False) -> None:
    """
    Update the intel YAML file with the metrics file path.

    Args:
        intel: Dictionary containing intel data
        metrics_path: Path to the saved metrics file
        intel_path: Path to the intel YAML file
        is_optimized: Whether the metrics are for the optimized model
    """
    section("Updating Intel YAML", logger)
    try:
        # Update intel dictionary with appropriate key based on model type
        if is_optimized:
            intel["optimized_performance_metrics_path"] = metrics_path
            intel["optimized_evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with optimized performance metrics path: {metrics_path}")
        else:
            intel["performance_metrics_path"] = metrics_path
            intel["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with performance metrics path: {metrics_path}")

        # Save updated intel to YAML
        with open(intel_path, "w") as f:
            yaml.dump(intel, f, default_flow_style=False)

    except Exception as e:
        logger.error(f"Failed to update intel: {e}")
        raise


def check_optimized_model_exists(dataset_name: str) -> Tuple[bool, str]:
    """
    Check if the optimized model exists.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (exists, path)
    """
    optimized_model_path = os.path.join("model", f"model_{dataset_name}", "optimized_model.pkl")
    exists = os.path.isfile(optimized_model_path)

    if exists:
        logger.info(f"Optimized model found at {optimized_model_path}")
    else:
        logger.info("No optimized model found")

    return exists, optimized_model_path


def main():
    """Main function to orchestrate the model evaluation process."""
    section("Starting Model Evaluation", logger, char="*", length=60)

    try:
        # Load intel
        intel = load_intel()

        # Extract required paths and config
        model_path = intel["model_path"]
        test_path = intel["test_transformed_path"]
        target_column = intel["target_column"]
        dataset_name = intel["dataset_name"]

        # Load test data - do this once for both evaluations
        X_test, y_test = load_test_data(test_path, target_column)

        # --- Evaluate the original model ---
        # Load original model
        model = load_model(model_path)

        # Evaluate original model
        metrics = evaluate_model(model, X_test, y_test)

        # Save original model metrics
        metrics_path = save_metrics(metrics, dataset_name, "performance.yaml")

        # Update intel with original metrics path
        update_intel(intel, metrics_path)

        # --- Check for and evaluate the optimized model if it exists ---
        optimized_exists, optimized_model_path = check_optimized_model_exists(dataset_name)

        if optimized_exists:
            section("Evaluating Optimized Model", logger, char="-", length=50)

            # Load optimized model
            optimized_model = load_model(optimized_model_path)

            # Evaluate optimized model
            optimized_metrics = evaluate_model(optimized_model, X_test, y_test)

            # Save optimized model metrics
            optimized_metrics_path = save_metrics(
                optimized_metrics, dataset_name, "optimized_performance.yaml")

            # Update intel with optimized metrics path
            update_intel(intel, optimized_metrics_path, is_optimized=True)

            logger.info("Optimized model evaluation completed successfully")

        section("Model Evaluation Complete", logger, char="*", length=60)

    except Exception as e:
        logger.critical(f"Model evaluation failed: {e}")
        section("Model Evaluation Failed", logger, level=logging.CRITICAL, char="*", length=60)
        raise


if __name__ == "__main__":
    main()