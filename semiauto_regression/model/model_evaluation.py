"""
Model Evaluation Module.

This module evaluates the performance of trained regression models using various metrics
and stores the results in a YAML file. It also evaluates the optimized model if available.
API-friendly version that can be imported and used in a FastAPI application.
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

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
from semiauto_regression.logger import section, configure_logger  # Configure logger

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
                 is_optimized: bool = False) -> Dict[str, Any]:
    """
    Update the intel YAML file with the metrics file path.

    Args:
        intel: Dictionary containing intel data
        metrics_path: Path to the saved metrics file
        intel_path: Path to the intel YAML file
        is_optimized: Whether the metrics are for the optimized model

    Returns:
        Updated intel dictionary
    """
    section("Updating Intel YAML", logger)
    try:
        # Create a new intel dictionary to avoid modifying the original
        updated_intel = intel.copy()

        # Update intel dictionary with appropriate key based on model type
        if is_optimized:
            updated_intel["optimized_performance_metrics_path"] = metrics_path
            updated_intel["optimized_evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with optimized performance metrics path: {metrics_path}")
        else:
            updated_intel["performance_metrics_path"] = metrics_path
            updated_intel["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with performance metrics path: {metrics_path}")

        # Save updated intel to YAML
        with open(intel_path, "w") as f:
            yaml.dump(updated_intel, f, default_flow_style=False)

        return updated_intel

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


def evaluate_and_save_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                           dataset_name: str, is_optimized: bool = False) -> Dict[str, float]:
    """
    Evaluate a model and save its metrics.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        dataset_name: Name of the dataset
        is_optimized: Whether this is an optimized model

    Returns:
        Dictionary of evaluation metrics
    """
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Determine filename
    filename = "optimized_performance.yaml" if is_optimized else "performance.yaml"

    # Save metrics
    metrics_path = save_metrics(metrics, dataset_name, filename)

    # Return metrics with path
    metrics["metrics_path"] = metrics_path
    return metrics


def run_evaluation(intel_path: str = "intel.yaml") -> Dict[str, Any]:
    """
    Run the complete model evaluation process.

    Args:
        intel_path: Path to the intel YAML file

    Returns:
        Dictionary containing evaluation results
    """
    section("Starting Model Evaluation", logger, char="*", length=60)
    results = {
        "success": False,
        "standard_model": None,
        "optimized_model": None,
        "intel": None,
        "error": None
    }

    try:
        # Load intel
        intel = load_intel(intel_path)
        results["intel"] = intel

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

        # Evaluate and save original model
        standard_metrics = evaluate_and_save_model(model, X_test, y_test, dataset_name, is_optimized=False)
        results["standard_model"] = standard_metrics

        # Update intel with original metrics path
        intel = update_intel(intel, standard_metrics["metrics_path"], intel_path)
        results["intel"] = intel

        # --- Check for and evaluate the optimized model if it exists ---
        optimized_exists, optimized_model_path = check_optimized_model_exists(dataset_name)

        if optimized_exists:
            section("Evaluating Optimized Model", logger, char="-", length=50)

            # Load optimized model
            optimized_model = load_model(optimized_model_path)

            # Evaluate and save optimized model
            optimized_metrics = evaluate_and_save_model(
                optimized_model, X_test, y_test, dataset_name, is_optimized=True)
            results["optimized_model"] = optimized_metrics

            # Update intel with optimized metrics path
            intel = update_intel(intel, optimized_metrics["metrics_path"], intel_path, is_optimized=True)
            results["intel"] = intel

            logger.info("Optimized model evaluation completed successfully")

        section("Model Evaluation Complete", logger, char="*", length=60)
        results["success"] = True
        return results

    except Exception as e:
        error_msg = f"Model evaluation failed: {str(e)}"
        logger.critical(error_msg)
        section("Model Evaluation Failed", logger, level=logging.CRITICAL, char="*", length=60)
        results["error"] = error_msg
        return results


def get_evaluation_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of the evaluation results.

    Args:
        evaluation_results: Results from run_evaluation

    Returns:
        Dictionary with summary information
    """
    summary = {
        "success": evaluation_results["success"],
        "dataset_name": evaluation_results["intel"]["dataset_name"] if evaluation_results["intel"] else None,
        "standard_model": {},
        "optimized_model": {},
        "has_optimized_model": evaluation_results["optimized_model"] is not None
    }

    # Extract key metrics for standard model
    if evaluation_results["standard_model"]:
        metrics = evaluation_results["standard_model"]
        summary["standard_model"] = {
            "r2_score": metrics["r2_score"],
            "rmse": metrics["root_mean_squared_error"],
            "mae": metrics["mean_absolute_error"]
        }

    # Extract key metrics for optimized model if available
    if evaluation_results["optimized_model"]:
        metrics = evaluation_results["optimized_model"]
        summary["optimized_model"] = {
            "r2_score": metrics["r2_score"],
            "rmse": metrics["root_mean_squared_error"],
            "mae": metrics["mean_absolute_error"]
        }

        # Add improvement metrics if both models exist
        if evaluation_results["standard_model"]:
            std_metrics = evaluation_results["standard_model"]
            opt_metrics = evaluation_results["optimized_model"]

            # Calculate improvement percentages
            r2_improvement = (opt_metrics["r2_score"] - std_metrics["r2_score"]) / max(abs(std_metrics["r2_score"]), 1e-10) * 100
            rmse_improvement = (std_metrics["root_mean_squared_error"] - opt_metrics["root_mean_squared_error"]) / std_metrics["root_mean_squared_error"] * 100
            mae_improvement = (std_metrics["mean_absolute_error"] - opt_metrics["mean_absolute_error"]) / std_metrics["mean_absolute_error"] * 100

            summary["improvement"] = {
                "r2_score": r2_improvement,
                "rmse": rmse_improvement,
                "mae": mae_improvement
            }

    return summary


if __name__ == "__main__":
    # This block only runs when the script is executed directly
    results = run_evaluation()
    if results["success"]:
        print("Model evaluation completed successfully.")
        # Print summary of evaluation results
        summary = get_evaluation_summary(results)
        print(f"\nEvaluation Summary for {summary['dataset_name']}:")
        print(f"Standard Model R² Score: {summary['standard_model']['r2_score']:.4f}")
        print(f"Standard Model RMSE: {summary['standard_model']['rmse']:.4f}")

        if summary['has_optimized_model']:
            print(f"\nOptimized Model R² Score: {summary['optimized_model']['r2_score']:.4f}")
            print(f"Optimized Model RMSE: {summary['optimized_model']['rmse']:.4f}")

            if 'improvement' in summary:
                print(f"\nImprovements:")
                print(f"R² Score: {summary['improvement']['r2_score']:.2f}%")
                print(f"RMSE: {summary['improvement']['rmse']:.2f}%")
    else:
        print(f"Model evaluation failed: {results['error']}")