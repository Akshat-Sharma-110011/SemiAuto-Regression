import os
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List, Optional, Union

# Import optimization libraries
from sklearn.model_selection import GridSearchCV
import optuna

# Import metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
)

# Import model libraries and classes
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    BayesianRidge, HuberRegressor
)
from sklearn.linear_model import RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Import custom logger
from src.logger import section, configure_logger

INTEL_PATH = "intel.yaml"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logger
configure_logger()
logger = logging.getLogger("Model Optimization")


class ModelOptimizer:
    def __init__(self, intel_path: str = INTEL_PATH, config_overrides: dict = None):
        self.intel_path = intel_path
        self.intel_config = self._load_intel()
        if config_overrides:
            self.intel_config.update(config_overrides)
        self.dataset_name = self.intel_config.get("dataset_name")
        self.model_name = self.intel_config.get("model_name")
        self.target_column = self.intel_config.get("target_column")

        # Store the logger as an instance attribute
        self.logger = logger

        # Load the data paths
        self.train_path = self.intel_config.get("train_transformed_path")
        self.test_path = self.intel_config.get("test_transformed_path")

        # Set up paths for saving outputs
        self.optimized_model_dir = os.path.join(ROOT_DIR, "model", f"model_{self.dataset_name}")
        self.optimized_model_path = os.path.join(self.optimized_model_dir, "optimized_model.pkl")

        self.best_params_dir = os.path.join(ROOT_DIR, "reports", "metrics", f"best_params_{self.dataset_name}")
        self.best_params_path = os.path.join(self.best_params_dir, "params.json")

        # Create directories if they don't exist
        os.makedirs(self.optimized_model_dir, exist_ok=True)
        os.makedirs(self.best_params_dir, exist_ok=True)

        # Load data
        self.X_train, self.y_train = self._load_data(self.train_path)
        self.X_test, self.y_test = self._load_data(self.test_path)

        # Available models and their hyperparameter spaces
        self.models = self._get_available_models()
        self.param_spaces = self._get_hyperparameter_spaces()

    def _load_intel(self) -> Dict[str, Any]:
        try:
            with open(self.intel_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Successfully loaded configuration from {self.intel_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")

            # Separate features and target
            X = data.drop(columns=[self.target_column], errors='ignore')
            if self.target_column in data.columns:
                y = data[self.target_column]
            else:
                logger.error(f"Target column '{self.target_column}' not found in data")
                raise ValueError(f"Target column '{self.target_column}' not found in data")

            logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _get_available_models(self) -> Dict[str, Any]:
        models = {
            "Linear Regression": LinearRegression,
            "Ridge Regression": Ridge,
            "Lasso Regression": Lasso,
            "ElasticNet": ElasticNet,
            "SGD Regressor": SGDRegressor,
            "Bayesian Ridge": BayesianRidge,
            "Huber Regressor": HuberRegressor,
            "RANSAC Regressor": RANSACRegressor,
            "Decision Tree": DecisionTreeRegressor,
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "AdaBoost": AdaBoostRegressor,
            "Extra Trees": ExtraTreesRegressor,
            "K-Nearest Neighbors": KNeighborsRegressor,
            "Support Vector Regression": SVR,
            "MLP Regressor": MLPRegressor,
            "XGBoost": xgb.XGBRegressor,
            "LightGBM": lgb.LGBMRegressor,
            "CatBoost": cb.CatBoostRegressor
        }

        # Log available models
        section("Available Regression Models", logger)
        for i, (name, _) in enumerate(models.items(), 1):
            descriptions = {
                "Linear Regression": "Standard linear regression model",
                "Ridge Regression": "Linear regression with L2 regularization",
                "Lasso Regression": "Linear regression with L1 regularization",
                "ElasticNet": "Linear regression with combined L1 and L2 regularization",
                "SGD Regressor": "Linear model fitted by minimizing a regularized loss function with SGD",
                "Bayesian Ridge": "Bayesian ridge regression with ARD prior",
                "Huber Regressor": "Regression model robust to outliers",
                "RANSAC Regressor": "RANSAC (RANdom SAmple Consensus) algorithm for robust regression",
                "Decision Tree": "Decision tree regressor",
                "Random Forest": "Ensemble of decision trees using bootstrap sampling",
                "Gradient Boosting": "Gradient boosting for regression",
                "AdaBoost": "AdaBoost regression algorithm",
                "Extra Trees": "Extremely randomized trees",
                "K-Nearest Neighbors": "Regression based on k-nearest neighbors",
                "Support Vector Regression": "Support vector regression",
                "MLP Regressor": "Multi-layer Perceptron regressor",
                "XGBoost": "XGBoost regression algorithm",
                "LightGBM": "LightGBM regression algorithm",
                "CatBoost": "CatBoost regression algorithm"
            }
            logger.info(f"{i}. {name} - {descriptions.get(name, '')}")

        return models

    def _get_hyperparameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        param_spaces = {
            "Linear Regression": {
                "fit_intercept": [True, False],
                "positive": [True, False]
            },
            "Ridge Regression": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            },
            "Lasso Regression": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "selection": ["cyclic", "random"]
            },
            "ElasticNet": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "selection": ["cyclic", "random"]
            },
            "SGD Regressor": {
                "loss": ["squared_error", "huber", "epsilon_insensitive"],
                "penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01],
                "l1_ratio": [0.15, 0.5, 0.85]
            },
            "Bayesian Ridge": {
                "alpha_1": [1e-6, 1e-5, 1e-4],
                "alpha_2": [1e-6, 1e-5, 1e-4],
                "lambda_1": [1e-6, 1e-5, 1e-4],
                "lambda_2": [1e-6, 1e-5, 1e-4]
            },
            "Huber Regressor": {
                "epsilon": [1.1, 1.35, 1.5, 2.0],
                "alpha": [0.0001, 0.001, 0.01]
            },
            "RANSAC Regressor": {
                "min_samples": [0.1, 0.5, 0.9],
                "max_trials": [50, 100, 200]
            },
            "Decision Tree": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2", None]
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2"]
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0]
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
                "loss": ["linear", "square", "exponential"]
            },
            "Extra Trees": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2"]
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2]
            },
            "Support Vector Regression": {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", "auto"]
            },
            "MLP Regressor": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh"],
                "solver": ["adam", "sgd"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "gamma": [1e-5, 0.1, 0.2]
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, -1],
                "num_leaves": [31, 50, 100],
                "min_child_samples": [20, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            },
            "CatBoost": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5, 7],
                "border_count": [32, 64, 128],
                "verbose": [False]
            }
        }
        return param_spaces

    def _get_model_class(self) -> Any:
        for model_key, model_class in self.models.items():
            if model_key.lower().replace(" ", "") == self.model_name.lower().replace(" ", ""):
                logger.info(f"Found model class for '{self.model_name}': {model_key}")
                return model_class

        # Direct mapping if the above fails
        model_mapping = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "RidgeRegression": Ridge,
            "Lasso": Lasso,
            "LassoRegression": Lasso,
            "ElasticNet": ElasticNet,
            "SGDRegressor": SGDRegressor,
            "BayesianRidge": BayesianRidge,
            "HuberRegressor": HuberRegressor,
            "RANSACRegressor": RANSACRegressor,
            "DecisionTree": DecisionTreeRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForest": RandomForestRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoosting": GradientBoostingRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "AdaBoost": AdaBoostRegressor,
            "AdaBoostRegressor": AdaBoostRegressor,
            "ExtraTrees": ExtraTreesRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "KNN": KNeighborsRegressor,
            "KNeighborsRegressor": KNeighborsRegressor,
            "SVR": SVR,
            "SupportVectorRegression": SVR,
            "MLP": MLPRegressor,
            "MLPRegressor": MLPRegressor,
            "XGBoost": xgb.XGBRegressor,
            "XGBRegressor": xgb.XGBRegressor,
            "LightGBM": lgb.LGBMRegressor,
            "LGBMRegressor": lgb.LGBMRegressor,
            "CatBoost": cb.CatBoostRegressor,
            "CatBoostRegressor": cb.CatBoostRegressor
        }

        if self.model_name in model_mapping:
            logger.info(f"Found model class for '{self.model_name}' in direct mapping")
            return model_mapping[self.model_name]

        logger.error(f"Model '{self.model_name}' not found in available models")
        raise ValueError(f"Model '{self.model_name}' not found in available models")

    def _get_param_space(self) -> Dict[str, Any]:
        for model_key, param_space in self.param_spaces.items():
            if model_key.lower().replace(" ", "") == self.model_name.lower().replace(" ", ""):
                return param_space

        # Try additional mappings
        model_mapping = {
            "LinearRegression": "Linear Regression",
            "Ridge": "Ridge Regression",
            "RidgeRegression": "Ridge Regression",
            "Lasso": "Lasso Regression",
            "LassoRegression": "Lasso Regression",
            "ElasticNet": "ElasticNet",
            "SGDRegressor": "SGD Regressor",
            "BayesianRidge": "Bayesian Ridge",
            "HuberRegressor": "Huber Regressor",
            "RANSACRegressor": "RANSAC Regressor",
            "DecisionTree": "Decision Tree",
            "DecisionTreeRegressor": "Decision Tree",
            "RandomForest": "Random Forest",
            "RandomForestRegressor": "Random Forest",
            "GradientBoosting": "Gradient Boosting",
            "GradientBoostingRegressor": "Gradient Boosting",
            "AdaBoost": "AdaBoost",
            "AdaBoostRegressor": "AdaBoost",
            "ExtraTrees": "Extra Trees",
            "ExtraTreesRegressor": "Extra Trees",
            "KNN": "K-Nearest Neighbors",
            "KNeighborsRegressor": "K-Nearest Neighbors",
            "SVR": "Support Vector Regression",
            "SupportVectorRegression": "Support Vector Regression",
            "MLP": "MLP Regressor",
            "MLPRegressor": "MLP Regressor",
            "XGBoost": "XGBoost",
            "XGBRegressor": "XGBoost",
            "LightGBM": "LightGBM",
            "LGBMRegressor": "LightGBM",
            "CatBoost": "CatBoost",
            "CatBoostRegressor": "CatBoost"
        }

        if self.model_name in model_mapping:
            mapped_name = model_mapping[self.model_name]
            if mapped_name in self.param_spaces:
                return self.param_spaces[mapped_name]

        logger.error(f"Parameter space for model '{self.model_name}' not found")
        raise ValueError(f"Parameter space for model '{self.model_name}' not found")

    def _calculate_metric(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            metric_name: str
    ) -> float:
        if metric_name == "mse":
            return mean_squared_error(y_true, y_pred)
        elif metric_name == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric_name == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric_name == "mape":
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif metric_name == "r2_score":
            return r2_score(y_true, y_pred)
        elif metric_name == "explained_variance_score":
            return explained_variance_score(y_true, y_pred)
        elif metric_name == "max_error":
            return max_error(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def optimize_with_grid_search(
            self,
            cv: int = 5,
            scoring: str = "neg_mean_squared_error",
            n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, Any]]:
        section("Grid Search Optimization", logger)

        model_class = self._get_model_class()
        param_grid = self._get_param_space()

        logger.info(f"Starting grid search for {self.model_name}")
        logger.info(f"Hyperparameter grid: {param_grid}")

        model = model_class()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        logger.info("Fitting grid search...")
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"Grid search complete. Best score: {best_score}")
        logger.info(f"Best parameters: {best_params}")

        return best_model, best_params

    def _objective(
            self,
            trial: optuna.Trial,
            model_class: Any,
            param_space: Dict[str, Any],
            metric_name: str,
            maximize: bool
    ) -> float:
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if all(isinstance(val, (int, float)) for val in param_values) and len(param_values) > 1:
                    if all(isinstance(val, int) for val in param_values):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            min(param_values),
                            max(param_values),
                            log=max(param_values) / max(1, min(param_values)) > 100
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            min(param_values),
                            max(param_values),
                            log=max(param_values) / max(1e-10, min(param_values)) > 100
                        )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

        if "hidden_layer_sizes" in param_space:
            params["hidden_layer_sizes"] = trial.suggest_categorical("hidden_layer_sizes",
                                                                     param_space["hidden_layer_sizes"])

        model = model_class(**params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        metric_value = self._calculate_metric(self.y_test, y_pred, metric_name)

        logger.info(
            f"Trial {trial.number} - Params: {params}, "
            f"Metric ({metric_name}): {metric_value:.4f}"
        )

        return metric_value if maximize else -metric_value

    def optimize_with_optuna(
            self,
            n_trials: int,
            metric_name: str = "rmse",
            maximize: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        section("Optuna Optimization", logger)

        model_class = self._get_model_class()
        param_space = self._get_param_space()

        logger.info(f"Starting Optuna optimization for {self.model_name}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Metric to {'maximize' if maximize else 'minimize'}: {metric_name}")

        direction = "maximize" if maximize else "minimize"
        study = optuna.create_study(direction=direction)

        study.optimize(
            lambda trial: self._objective(trial, model_class, param_space, metric_name, maximize),
            n_trials=n_trials
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        if maximize:
            logger.info(f"Optimization complete. Best {metric_name}: {best_value:.4f}")
        else:
            logger.info(f"Optimization complete. Best {metric_name}: {-best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        best_model = model_class(**best_params)
        best_model.fit(self.X_train, self.y_train)

        return best_model, best_params

    def save_optimized_model(self, model: Any, best_params: Dict[str, Any]) -> None:
        section("Saving Optimized Model", logger)

        try:
            joblib.dump(model, self.optimized_model_path)
            logger.info(f"Optimized model saved to {self.optimized_model_path}")
        except Exception as e:
            logger.error(f"Error saving optimized model: {str(e)}")
            raise

        try:
            with open(self.best_params_path, "w") as file:
                yaml.dump(best_params, file)
            logger.info(f"Best parameters saved to {self.best_params_path}")
        except Exception as e:
            logger.error(f"Error saving best parameters: {str(e)}")
            raise

    def update_intel_yaml(self) -> None:
        section("Updating Intel YAML", logger)

        try:
            self.intel_config["optimized_model_path"] = self.optimized_model_path
            self.intel_config["best_params_path"] = self.best_params_path
            self.intel_config["optimization_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.intel_path, "w") as file:
                yaml.dump(self.intel_config, file)

            logger.info(f"Updated intel.yaml with optimized model information")
        except Exception as e:
            logger.error(f"Error updating intel.yaml: {str(e)}")
            raise


def get_available_metrics():
    return {
        "1": ("rmse", False, "Root Mean Squared Error"),
        "2": ("mae", False, "Mean Absolute Error"),
        "3": ("r2_score", True, "R² Score"),
        "4": ("mape", False, "Mean Absolute Percentage Error"),
        "5": ("mse", False, "Mean Squared Error"),
        "6": ("explained_variance_score", True, "Explained Variance Score"),
        "7": ("max_error", False, "Maximum Error")
    }


def get_optimization_methods():
    return {
        "1": "Grid Search",
        "2": "Optuna"
    }


def optimize_model(
        optimize: bool = True,
        method: str = "1",
        n_trials: int = 50,
        metric: str = "1",
        config_overrides: dict = None
) -> dict:
    result = {
        "status": "success",
        "message": "",
        "best_params": None,
        "model_path": None,
        "metrics": {}
    }

    try:
        if not optimize:
            result["message"] = "Optimization skipped by user choice"
            return result

        logger.info("Starting model optimization process")
        optimizer = ModelOptimizer(config_overrides=config_overrides)

        if method == "1":
            optimized_model, best_params = optimizer.optimize_with_grid_search()
        else:
            metric_mapping = get_available_metrics()

            if metric not in metric_mapping:
                logger.warning("Invalid metric choice, defaulting to RMSE")
                metric = "1"

            metric_name, maximize, _ = metric_mapping[metric]
            optimized_model, best_params = optimizer.optimize_with_optuna(
                n_trials=n_trials,
                metric_name=metric_name,
                maximize=maximize
            )

        optimizer.save_optimized_model(optimized_model, best_params)
        optimizer.update_intel_yaml()

        y_pred = optimized_model.predict(optimizer.X_test)
        metrics = {
            "rmse": np.sqrt(mean_squared_error(optimizer.y_test, y_pred)),
            "mae": mean_absolute_error(optimizer.y_test, y_pred),
            "r2": r2_score(optimizer.y_test, y_pred),
            "explained_variance": explained_variance_score(optimizer.y_test, y_pred)
        }

        result.update({
            "best_params": best_params,
            "model_path": optimizer.optimized_model_path,
            "metrics": metrics,
            "message": "Optimization completed successfully"
        })

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        result.update({
            "status": "error",
            "message": str(e)
        })

    return result