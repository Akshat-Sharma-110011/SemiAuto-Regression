#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module handles the model selection and training for the AutoML regression pipeline.
It supports various regression models and hyperparameter tuning.
"""

import os
import yaml
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
import sys

# Add parent directory to path to import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import section, configure_logger
import logging

# Import regression models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, SGDRegressor, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.cross_decomposition import PLSRegression

# Import advanced ensemble models
try:
    import xgboost as xgb
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)
configure_logger()


class ModelBuilder:
    """Class to handle model selection, training, and persistence."""

    def __init__(self, intel_path='intel.yaml'):
        """
        Initialize the ModelBuilder with paths from intel.yaml.

        Args:
            intel_path (str): Path to the intel.yaml file.
        """
        # Load intel file
        section("Initializing Model Builder", logger)
        self.intel_path = intel_path
        self.intel = self._load_intel()

        # Extract necessary paths and info
        self.dataset_name = self.intel['dataset_name']
        self.target_column = self.intel['target_column']
        self.train_transformed_path = self.intel['train_transformed_path']
        self.test_transformed_path = self.intel['test_transformed_path']

        # Create model directory
        project_root = Path(__file__).parent.parent.parent
        self.model_dir = project_root / 'models' / f'model_{self.dataset_name}'
        self.model_path = self.model_dir / 'model.pkl'
        os.makedirs(self.model_dir, exist_ok=True)

        # Available models dictionary with default parameters
        self.models_dict = self._get_models_dict()

        logger.info(f"Model builder initialized for dataset: {self.dataset_name}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Model will be saved to: {self.model_path}")

    def _load_intel(self):
        """Load the intel file containing paths and configurations."""
        try:
            with open(self.intel_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load intel file: {e}")
            raise

    def _get_models_dict(self):
        """
        Create a dictionary of available regression models with their default parameters.

        Returns:
            dict: Dictionary with model names as keys and (model_class, default_params) as values.
        """
        models = {
            # Linear models
            'Linear Regression': (LinearRegression, {}),
            'Ridge Regression': (Ridge, {'alpha': 1.0, 'max_iter': 1000}),
            'Lasso Regression': (Lasso, {'alpha': 1.0, 'max_iter': 1000}),
            'ElasticNet': (ElasticNet, {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 1000}),
            'Huber Regressor': (HuberRegressor, {'epsilon': 1.35, 'alpha': 0.0001, 'max_iter': 100}),
            'SGD Regressor': (SGDRegressor,
                              {'loss': 'squared_error', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 1000}),
            'Bayesian Ridge': (BayesianRidge,
                               {'n_iter': 300, 'alpha_1': 1e-6, 'alpha_2': 1e-6, 'lambda_1': 1e-6, 'lambda_2': 1e-6}),

            # Support Vector Machines
            'SVR': (SVR, {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale'}),

            # Nearest Neighbors
            'K-Neighbors Regressor': (KNeighborsRegressor,
                                      {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}),

            # Decision Trees
            'Decision Tree': (DecisionTreeRegressor,
                              {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}),

            # Ensemble Methods
            'Random Forest': (RandomForestRegressor,
                              {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}),
            'Gradient Boosting': (GradientBoostingRegressor,
                                  {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 2,
                                   'min_samples_leaf': 1}),
            'AdaBoost': (AdaBoostRegressor, {'n_estimators': 50, 'learning_rate': 1.0}),
            'Extra Trees': (ExtraTreesRegressor,
                            {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}),

            # Neural Networks
            'MLP Regressor': (MLPRegressor,
                              {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'max_iter': 200}),

            # Gaussian Processes
            'Gaussian Process': (GaussianProcessRegressor,
                                 {'kernel': ConstantKernel(1.0) * RBF(1.0), 'alpha': 1e-10, 'n_restarts_optimizer': 0}),

            # Partial Least Squares
            'PLS Regression': (PLSRegression, {'n_components': 2, 'scale': True, 'max_iter': 500}),
        }

        # Add advanced ensemble models if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = (XGBRegressor, {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            })

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = (LGBMRegressor, {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'regression',
                'random_state': 42
            })

        if CATBOOST_AVAILABLE:
            models['CatBoost'] = (CatBoostRegressor, {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'RMSE',
                'verbose': False,
                'random_state': 42
            })

        return models

    def load_data(self):
        """
        Load and prepare the training and testing data.

        Returns:
            tuple: X_train, y_train, X_test, y_test
        """
        section("Loading Transformed Data", logger)
        try:
            logger.info(f"Loading training data from {self.train_transformed_path}")
            train_df = pd.read_csv(self.train_transformed_path)

            logger.info(f"Loading testing data from {self.test_transformed_path}")
            test_df = pd.read_csv(self.test_transformed_path)

            # Split into features and target
            X_train = train_df.drop(self.target_column, axis=1)
            y_train = train_df[self.target_column]

            X_test = test_df.drop(self.target_column, axis=1)
            y_test = test_df[self.target_column]

            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Testing data shape: {X_test.shape}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def display_available_models(self):
        """
        Display the list of available regression models.

        Returns:
            dict: Dictionary of available models with their indices.
        """
        section("Available Regression Models", logger)
        model_dict = {}
        for i, model_name in enumerate(self.models_dict.keys(), 1):
            model_dict[i] = model_name
            print(f"{i}. {model_name}")

        return model_dict

    def select_model(self):
        """
        Interactive model selection process.

        Returns:
            tuple: Selected model class and default parameters.
        """
        section("Model Selection", logger)
        model_dict = self.display_available_models()

        while True:
            try:
                choice = int(input("\nSelect a model by entering its number: "))
                if choice in model_dict:
                    selected_model_name = model_dict[choice]
                    selected_model_class, default_params = self.models_dict[selected_model_name]
                    logger.info(f"Selected model: {selected_model_name}")
                    return selected_model_name, selected_model_class, default_params
                else:
                    print(f"Invalid choice. Please select a number between 1 and {len(model_dict)}")
            except ValueError:
                print("Please enter a valid number.")

    def tune_hyperparameters(self, model_name, default_params):
        """
        Interactive hyperparameter tuning process.

        Args:
            model_name (str): Name of the selected model.
            default_params (dict): Default parameters for the model.

        Returns:
            dict: Updated parameters for the model.
        """
        section(f"Hyperparameter Tuning for {model_name}", logger)

        print(f"\nDefault parameters for {model_name}:")
        for param, value in default_params.items():
            print(f"  - {param}: {value}")

        tune = input("\nWould you like to tune any hyperparameters? (yes/no): ").lower().strip()

        if tune in ['yes', 'y']:
            params = default_params.copy()
            print("\nEnter new values for parameters you want to tune (leave empty to keep default):")

            for param, default_value in default_params.items():
                while True:
                    user_value = input(f"  {param} (default: {default_value}): ").strip()

                    if not user_value:  # Keep default
                        break

                    try:
                        # Try to interpret the input value based on the default value type
                        if isinstance(default_value, int):
                            params[param] = int(user_value)
                        elif isinstance(default_value, float):
                            params[param] = float(user_value)
                        elif isinstance(default_value, str):
                            params[param] = user_value
                        elif isinstance(default_value, bool):
                            params[param] = user_value.lower() in ['true', 'yes', 'y', '1']
                        elif isinstance(default_value, tuple):
                            # Parse tuple of integers, like hidden_layer_sizes
                            params[param] = tuple(
                                int(x.strip()) for x in user_value.strip('()').split(',') if x.strip())
                        elif default_value is None:
                            if user_value.lower() in ['none', 'null']:
                                params[param] = None
                            else:
                                # Try to guess the type
                                try:
                                    params[param] = int(user_value)
                                except ValueError:
                                    try:
                                        params[param] = float(user_value)
                                    except ValueError:
                                        params[param] = user_value
                        else:
                            # For complex types like kernels in Gaussian Process
                            print(f"Complex parameter type. Using default value for {param}.")

                        break
                    except ValueError:
                        print("Invalid value. Please try again.")

            logger.info(f"Using custom parameters for {model_name}")
            for param, value in params.items():
                if params[param] != default_params[param]:
                    logger.info(f"  - Changed {param}: {default_params[param]} -> {value}")

            return params
        else:
            logger.info(f"Using default parameters for {model_name}")
            return default_params

    def train_model(self, model_class, params, X_train, y_train):
        """
        Train the selected model with specified parameters.

        Args:
            model_class: The scikit-learn model class to train.
            params (dict): Parameters for the model.
            X_train: Training features.
            y_train: Training target.

        Returns:
            object: Trained model.
        """
        section("Model Training", logger)

        try:
            logger.info(f"Initializing model with parameters: {params}")
            model = model_class(**params)

            logger.info("Training model...")
            model.fit(X_train, y_train)

            logger.info("Model training completed successfully")
            return model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def save_model(self, model, model_name, params):
        """
        Save the trained model and metadata to disk.

        Args:
            model: Trained model object.
            model_name (str): Name of the model.
            params (dict): Parameters used for the model.
        """
        section("Saving Model", logger)

        try:
            # Create model info dictionary
            model_info = {
                'model': model,
                'model_name': model_name,
                'parameters': params,
                'dataset_name': self.dataset_name,
                'target_column': self.target_column,
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save model with cloudpickle
            with open(self.model_path, 'wb') as f:
                cloudpickle.dump(model_info, f)

            logger.info(f"Model saved successfully to {self.model_path}")

            # Save model metadata in yaml format for easy reading
            metadata_path = self.model_dir / 'model_metadata.yaml'
            metadata = {
                'model_name': model_name,
                'parameters': {k: str(v) for k, v in params.items()},  # Convert all values to strings for YAML
                'dataset_name': self.dataset_name,
                'target_column': self.target_column,
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

            logger.info(f"Model metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def run(self):
        """
        Run the model building pipeline.
        """
        section("Starting Model Building Pipeline", logger, char='*', length=80)

        try:
            # Load data
            X_train, y_train, X_test, y_test = self.load_data()

            # Select model
            model_name, model_class, default_params = self.select_model()

            # Tune hyperparameters if requested
            params = self.tune_hyperparameters(model_name, default_params)

            # Train model
            model = self.train_model(model_class, params, X_train, y_train)

            # Save model
            self.save_model(model, model_name, params)

            section("Model Building Pipeline Completed Successfully", logger, char='*', length=80)

            return model

        except Exception as e:
            logger.error(f"Model building pipeline failed: {e}")
            section("Model Building Pipeline Failed", logger, level=logging.ERROR, char='*', length=80)
            raise


if __name__ == "__main__":
    builder = ModelBuilder()
    builder.run()