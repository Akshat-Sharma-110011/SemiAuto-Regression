#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script handles model selection, training, and storing for regression problems.
It provides a selection of regression models including advanced ensemble models,
allows for custom hyperparameter tuning, and stores the trained model.
API-friendly version for integration with FastAPI.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# Import regression models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    BayesianRidge, HuberRegressor, RANSACRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Import advanced ensemble models
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Import the custom logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from semiauto_regression.logger import section, configure_logger  # Configure logger

# Configure logger
configure_logger()
logger = logging.getLogger("Model Building")


class ModelBuilder:
    """
    A class to build, tune, and save regression models for the AutoML pipeline.
    API-friendly version for integration with FastAPI.
    """

    def __init__(self, intel_path: str = "intel.yaml"):
        """
        Initialize ModelBuilder with paths from intel.yaml

        Args:
            intel_path: Path to the intel.yaml file
        """
        section(f"Initializing ModelBuilder with intel file: {intel_path}", logger)
        self.intel_path = intel_path
        self.intel = self._load_intel()
        self.dataset_name = self.intel.get('dataset_name')
        self.target_column = self.intel.get('target_column')

        # Load data paths
        self.train_data_path = self.intel.get('train_transformed_path')
        self.test_data_path = self.intel.get('test_transformed_path')

        # Setup model directory
        self.model_dir = Path(f"model/model_{self.dataset_name}")
        self.model_path = self.model_dir / "model.pkl"

        # Available models dictionary with their default parameters
        self.available_models = self._get_available_models()

        logger.info(f"ModelBuilder initialized for dataset: {self.dataset_name}")
        logger.info(f"Target column: {self.target_column}")

    def _load_intel(self) -> Dict[str, Any]:
        """Load the intel.yaml file"""
        try:
            with open(self.intel_path, 'r') as file:
                intel = yaml.safe_load(file)
            logger.info(f"Successfully loaded intel from {self.intel_path}")
            return intel
        except Exception as e:
            logger.error(f"Failed to load intel file: {e}")
            raise

    def _get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available regression models with their default parameters

        Returns:
            Dictionary of model names and their class/default parameters
        """
        models = {
            # Basic models
            "Linear Regression": {
                "class": LinearRegression,
                "params": {"fit_intercept": True, "n_jobs": -1},
                "description": "Standard linear regression model"
            },
            "Ridge Regression": {
                "class": Ridge,
                "params": {"alpha": 1.0, "fit_intercept": True, "max_iter": 1000},
                "description": "Linear regression with L2 regularization"
            },
            "Lasso Regression": {
                "class": Lasso,
                "params": {"alpha": 1.0, "fit_intercept": True, "max_iter": 1000},
                "description": "Linear regression with L1 regularization"
            },
            "ElasticNet": {
                "class": ElasticNet,
                "params": {"alpha": 1.0, "l1_ratio": 0.5, "fit_intercept": True, "max_iter": 1000},
                "description": "Linear regression with combined L1 and L2 regularization"
            },
            "SGD Regressor": {
                "class": SGDRegressor,
                "params": {"loss": "squared_error", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000},
                "description": "Linear model fitted by minimizing a regularized loss function with SGD"
            },
            "Bayesian Ridge": {
                "class": BayesianRidge,
                "params": {"n_iter": 300, "alpha_1": 1e-6, "alpha_2": 1e-6},
                "description": "Bayesian ridge regression with ARD prior"
            },
            "Huber Regressor": {
                "class": HuberRegressor,
                "params": {"epsilon": 1.35, "alpha": 0.0001, "max_iter": 100},
                "description": "Regression model robust to outliers"
            },
            "RANSAC Regressor": {
                "class": RANSACRegressor,
                "params": {"min_samples": 0.1, "max_trials": 100},
                "description": "RANSAC (RANdom SAmple Consensus) algorithm for robust regression"
            },
            # Tree-based models
            "Decision Tree": {
                "class": DecisionTreeRegressor,
                "params": {"max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
                "description": "Decision tree regressor"
            },
            "Random Forest": {
                "class": RandomForestRegressor,
                "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "n_jobs": -1},
                "description": "Ensemble of decision trees using bootstrap sampling"
            },
            "Gradient Boosting": {
                "class": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0},
                "description": "Gradient boosting for regression"
            },
            "AdaBoost": {
                "class": AdaBoostRegressor,
                "params": {"n_estimators": 50, "learning_rate": 1.0, "loss": "linear"},
                "description": "AdaBoost regression algorithm"
            },
            "Extra Trees": {
                "class": ExtraTreesRegressor,
                "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "n_jobs": -1},
                "description": "Extremely randomized trees"
            },
            # Other models
            "K-Nearest Neighbors": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "n_jobs": -1},
                "description": "Regression based on k-nearest neighbors"
            },
            "Support Vector Regression": {
                "class": SVR,
                "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"},
                "description": "Support vector regression"
            },
            "MLP Regressor": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (100,), "activation": "relu", "solver": "adam", "max_iter": 200},
                "description": "Multi-layer Perceptron regressor"
            },
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = {
                "class": xgb.XGBRegressor,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "n_jobs": -1
                },
                "description": "XGBoost regression algorithm"
            }

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = {
                "class": lgb.LGBMRegressor,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "regression",
                    "n_jobs": -1
                },
                "description": "LightGBM regression algorithm"
            }

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = {
                "class": cb.CatBoostRegressor,
                "params": {
                    "iterations": 100,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "l2_leaf_reg": 3,
                    "loss_function": "RMSE",
                    "verbose": False
                },
                "description": "CatBoost regression algorithm"
            }

        return models

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load the training and test data

        Returns:
            X_train, y_train, X_test, y_test
        """
        section("Loading Data", logger)

        try:
            # Load train data
            train_data = pd.read_csv(self.train_data_path)
            logger.info(f"Loaded training data from {self.train_data_path}")
            logger.info(f"Training data shape: {train_data.shape}")

            # Load test data
            test_data = pd.read_csv(self.test_data_path)
            logger.info(f"Loaded test data from {self.test_data_path}")
            logger.info(f"Test data shape: {test_data.shape}")

            # Separate features and target
            X_train = train_data.drop(columns=[self.target_column])
            y_train = train_data[self.target_column]
            X_test = test_data.drop(columns=[self.target_column])
            y_test = test_data[self.target_column]

            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_available_models(self) -> Dict[str, Dict]:
        """
        API-friendly method to get available models and their default parameters.

        Returns:
            Dictionary with model names, descriptions and default parameters
        """
        models_info = {}

        for model_name, model_data in self.available_models.items():
            # Don't include the class object, just params and description
            models_info[model_name] = {
                "params": model_data["params"],
                "description": model_data["description"]
            }

        return models_info

    def process_model_request(self, model_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        API-friendly method to process a model request.

        Args:
            model_name: Name of the model to train
            custom_params: Optional dictionary of custom parameters to use

        Returns:
            Dictionary with model information
        """
        section(f"Processing model request for {model_name}", logger)

        # Check if model exists
        if model_name not in self.available_models:
            error_msg = f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get model info and apply custom parameters if provided
        model_info = self.available_models[model_name].copy()
        if custom_params:
            # Apply custom parameters with type checking
            for param_name, param_value in custom_params.items():
                # Check if the parameter exists in the default parameters
                if param_name in model_info['params']:
                    # Get the default value's type
                    default_value = model_info['params'][param_name]
                    # Cast the custom value to the same type as the default
                    try:
                        if isinstance(default_value, bool):
                            # Handle booleans which might come as strings
                            if isinstance(param_value, str):
                                # Convert string 'true' or 'false' to boolean
                                cast_value = param_value.lower() == 'true'
                            else:
                                cast_value = bool(param_value)
                        elif isinstance(default_value, int):
                            # Cast to int, allowing for float strings (e.g., '5' -> 5)
                            cast_value = int(float(param_value)) if isinstance(param_value, str) else int(param_value)
                        elif isinstance(default_value, float):
                            cast_value = float(param_value)
                        else:
                            # For other types (str, etc.), use as is
                            cast_value = param_value
                        model_info['params'][param_name] = cast_value
                    except Exception as e:
                        logger.error(
                            f"Failed to cast parameter '{param_name}' value '{param_value}' to type {type(default_value)}: {e}")
                        raise ValueError(f"Invalid value for parameter '{param_name}': {param_value}")
                else:
                    # Add new parameter not present in defaults
                    model_info['params'][param_name] = param_value
            logger.info(f"Applied custom parameters: {custom_params}")

        try:
            # Load data
            X_train, y_train, X_test, y_test = self.load_data()

            # Train model
            model = self.train_model(model_name, model_info, X_train, y_train)

            # Save model
            model_path = self.save_model(model, model_name)

            # Return result
            result = {
                'model_name': model_name,
                'model_path': model_path,
                'parameters': model_info['params'],
                'status': 'success'
            }

            return result

        except Exception as e:
            error_msg = f"Error processing model request: {str(e)}"
            logger.error(error_msg)
            raise

    def train_model(self, model_name: str, model_info: Dict[str, Any], X_train: pd.DataFrame,
                    y_train: pd.Series) -> Any:
        """
        Train the selected model

        Args:
            model_name: Name of the model
            model_info: Dictionary with model class and parameters
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model object
        """
        section(f"Training {model_name}", logger)

        try:
            # Instantiate model with parameters
            model = model_info['class'](**model_info['params'])

            # Train the model
            logger.info(f"Starting training for {model_name}...")
            model.fit(X_train, y_train)
            logger.info(f"Model training completed successfully")

            return model

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save the trained model using cloudpickle

        Args:
            model: Trained model object
            model_name: Name of the model

        Returns:
            Path to the saved model
        """
        section("Saving Model", logger)

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        try:
            # Save model with cloudpickle
            with open(self.model_path, 'wb') as f:
                cloudpickle.dump(model, f)

            logger.info(f"Model saved to {self.model_path}")

            # Update intel.yaml with model path
            model_path_str = str(self.model_path)
            self.intel['model_path'] = model_path_str
            self.intel['model_name'] = model_name
            self.intel['model_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(self.intel_path, 'w') as f:
                yaml.dump(self.intel, f, default_flow_style=False)

            logger.info(f"Updated intel.yaml with model path: {model_path_str}")

            return model_path_str

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


# For backward compatibility with command-line usage
if __name__ == "__main__":
    print("This script is intended to be used as a module by the FastAPI application.")
    print("For command-line usage, please use the original model_building.py script.")
    sys.exit(1)