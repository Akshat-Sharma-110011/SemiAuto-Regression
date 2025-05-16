#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script handles model selection, training, and storing for regression problems.
It provides a selection of regression models including advanced ensemble models,
allows for custom hyperparameter tuning, and stores the trained model.
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
from src.logger import section, configure_logger  # Configure logger

# Configure logger
configure_logger()
logger = logging.getLogger("Model Building")


class ModelBuilder:
    """
    A class to build, tune, and save regression models for the AutoML pipeline.
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

    def select_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Display available models and let the user select one.

        Returns:
            model_name: Name of the selected model
            model_info: Dictionary with model class and parameters
        """
        section("Model Selection", logger)

        logger.info("Available regression models:")
        for i, (model_name, model_info) in enumerate(self.available_models.items(), 1):
            logger.info(f"{i}. {model_name} - {model_info['description']}")

        valid_selection = False
        model_choice = None

        while not valid_selection:
            try:
                model_idx = int(input("\nSelect a model by entering its number: "))
                if 1 <= model_idx <= len(self.available_models):
                    model_choice = list(self.available_models.keys())[model_idx - 1]
                    valid_selection = True
                else:
                    print(f"Please enter a number between 1 and {len(self.available_models)}")
            except ValueError:
                print("Please enter a valid number")

        model_info = self.available_models[model_choice]
        logger.info(f"Selected model: {model_choice}")

        # Ask if user wants to customize hyperparameters
        customize = input("\nDo you want to customize the model hyperparameters? (y/n): ").lower().strip()

        if customize == 'y':
            logger.info("Default parameters:")
            for param, value in model_info['params'].items():
                print(f"  {param}: {value}")

            print("\nEnter new parameters (press Enter to keep default, enter 'None' to set to None):")
            new_params = {}

            for param, default_value in model_info['params'].items():
                new_value = input(f"  {param} (default: {default_value}): ").strip()

                if new_value:
                    if new_value.lower() == 'none':
                        new_params[param] = None
                    elif isinstance(default_value, bool):
                        new_params[param] = new_value.lower() == 'true'
                    elif isinstance(default_value, int):
                        new_params[param] = int(new_value)
                    elif isinstance(default_value, float):
                        new_params[param] = float(new_value)
                    elif isinstance(default_value, tuple):
                        # Parse tuple of integers like (100, 50, 25)
                        try:
                            # Remove parentheses and split by commas
                            values = new_value.strip('()').split(',')
                            new_params[param] = tuple(int(v.strip()) for v in values)
                        except ValueError:
                            logger.warning(f"Could not parse {new_value} as tuple, keeping default: {default_value}")
                            new_params[param] = default_value
                    else:
                        new_params[param] = new_value

            # Update the parameters
            model_info['params'].update(new_params)
            logger.info("Updated parameters:")
            for param, value in model_info['params'].items():
                logger.info(f"  {param}: {value}")

        return model_choice, model_info

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

    def run(self) -> Dict[str, Any]:
        """
        Run the entire model building process

        Returns:
            Dictionary with model information
        """
        section("MODEL BUILDING PROCESS", logger, char='#', length=80)

        try:
            # Load data
            X_train, y_train, X_test, y_test = self.load_data()

            # Select model
            model_name, model_info = self.select_model()

            # Train model
            model = self.train_model(model_name, model_info, X_train, y_train)

            # Save model
            model_path = self.save_model(model, model_name)

            result = {
                'model_name': model_name,
                'model_path': model_path,
                'model_info': model_info
            }

            section("MODEL BUILDING COMPLETED SUCCESSFULLY", logger, char='#', length=80)

            return result

        except Exception as e:
            logger.error(f"Error in model building process: {e}")
            section("MODEL BUILDING FAILED", logger, level=logging.ERROR, char='#', length=80)
            raise


if __name__ == "__main__":
    # Set up the model builder and run
    try:
        model_builder = ModelBuilder()
        result = model_builder.run()
        logger.info(f"Model built successfully: {result['model_name']}")
    except Exception as e:
        logger.error(f"Model building failed: {str(e)}")
        sys.exit(1)