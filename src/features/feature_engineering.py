#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering script for regression automl clone.
This module handles automatic feature generation, transformation pipeline creation,
and integration with preprocessing pipeline.
API-friendly version that can be called from FastAPI endpoints.
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys
import yaml
import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

# Add parent directory to path for importing custom logger
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import custom logger
import logging
from src.logger import section, configure_logger


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """A transformer that returns the data unchanged."""

    def __init__(self):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing IdentityTransformer")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FeatureToolsTransformer(BaseEstimator, TransformerMixin):
    """A transformer that uses featuretools to create new features."""

    def __init__(self, target_col: str):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing FeatureToolsTransformer")
        self.target_col = target_col
        self.feature_defs = None
        self.feature_names = None

        try:
            import featuretools as ft
            self.ft = ft
        except ImportError:
            self.logger.error("Featuretools package not found. Please install with: pip install featuretools")
            raise

    def fit(self, X, y=None):
        try:
            es = self.ft.EntitySet(id="features")
            X_copy = X.copy()

            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                index_name = "index"
            else:
                index_name = X_copy.index.name

            es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index=index_name,
                make_index=True,
                time_index=None
            )

            feature_matrix, feature_defs = self.ft.dfs(
                entityset=es,
                target_dataframe_name="data",
                trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric", "subtract_numeric"],
                max_depth=1,
                features_only=False,
                verbose=True
            )

            self.feature_defs = feature_defs
            self.feature_names = [col for col in feature_matrix.columns if self.target_col not in col]
            self.logger.info(f"Generated {len(self.feature_names)} features using featuretools")
            return self

        except Exception as e:
            self.logger.error(f"Error in FeatureToolsTransformer fit: {str(e)}")
            raise

    def transform(self, X):
        try:
            X_copy = X.copy()

            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                index_name = "index"
            else:
                index_name = X_copy.index.name

            es = self.ft.EntitySet(id="features_transform")
            es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index=index_name,
                make_index=True,
                time_index=None
            )

            feature_matrix = self.ft.calculate_feature_matrix(
                features=self.feature_defs,
                entityset=es,
                verbose=True
            )

            feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_matrix = feature_matrix[self.feature_names]

            for col in feature_matrix.columns:
                if not pd.api.types.is_numeric_dtype(feature_matrix[col]):
                    feature_matrix = feature_matrix.drop(columns=[col])

            return feature_matrix

        except Exception as e:
            self.logger.error(f"Error in FeatureToolsTransformer transform: {str(e)}")
            raise


class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    """A transformer that uses SHAP values to select important features."""

    def __init__(self, n_features: int = 20, model=None):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info(f"Initializing SHAPFeatureSelector with n_features={n_features}")
        self.n_features = n_features
        self.selected_features = None
        self.importance_df = None
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42)

        try:
            import shap
            self.shap = shap
        except ImportError:
            self.logger.error("SHAP package not found. Please install with: pip install shap")
            raise

    def fit(self, X, y=None):
        try:
            X_copy = X.copy()
            feature_names = list(X_copy.columns)
            self.n_features = min(self.n_features, len(feature_names))

            if X_copy.shape[0] < 10:
                self.selected_features = feature_names
                return self

            self.model.fit(X_copy, y)

            if hasattr(self.model, 'estimators_'):
                explainer = self.shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_copy)
            else:
                sample_size = min(100, int(X_copy.shape[0] * 0.2))
                background = self.shap.kmeans(X_copy, sample_size)
                explainer = self.shap.KernelExplainer(self.model.predict, background)
                shap_values = explainer.shap_values(X_copy.sample(n=min(200, X_copy.shape[0])))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            feature_importance = np.abs(shap_values).mean(axis=0)
            self.importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            self.selected_features = self.importance_df['feature'].head(self.n_features).tolist()
            return self

        except Exception as e:
            self.logger.error(f"Error in SHAPFeatureSelector fit: {str(e)}")
            self.selected_features = list(X.columns)
            return self

    def transform(self, X):
        try:
            X_copy = X.copy()
            available_features = [f for f in self.selected_features if f in X_copy.columns]
            return X_copy[available_features]
        except Exception as e:
            self.logger.error(f"Error in SHAPFeatureSelector transform: {str(e)}")
            return X


class FeatureEngineer:
    """Main class for feature engineering process, API-friendly version."""

    def __init__(self, config_path: Union[str, Path] = "intel.yaml"):
        """
        Initialize the FeatureEngineer.

        Args:
            config_path: Path to the config file (intel.yaml)
        """
        self.logger = logging.getLogger("Feature Engineering")
        section("FEATURE ENGINEERING INITIALIZATION", self.logger)

        # Configure logger if not already configured
        try:
            if not self.logger.handlers:
                configure_logger()
        except Exception:
            # If logger configuration fails, continue with default logger
            pass

        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent
        self.intel = self._load_intel()

        # Load dataset name from config
        self.dataset_name = self.intel.get("dataset_name")
        if not self.dataset_name:
            raise ValueError("dataset_name not found in intel.yaml")

        self.feature_store = self._load_feature_store()
        self.target_column = self.intel.get("target_column")
        self._setup_paths()

    def _load_intel(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading intel.yaml: {str(e)}")
            raise

    def _load_feature_store(self) -> Dict:
        try:
            feature_store_path = self.project_root / self.intel.get("feature_store_path")
            with open(feature_store_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading feature store: {str(e)}")
            raise

    def _setup_paths(self):
        # Construct absolute paths using project root
        self.transformation_pipeline_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/transformation.pkl"
        self.processor_pipeline_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/processor.pkl"
        self.train_transformed_path = self.project_root / f"data/processed/data_{self.dataset_name}/train_transformed.csv"
        self.test_transformed_path = self.project_root / f"data/processed/data_{self.dataset_name}/test_transformed.csv"

        # Ensure directories exist
        self.transformation_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_transformed_path.parent.mkdir(parents=True, exist_ok=True)

    def _update_intel(self, use_feature_tools: bool, use_shap: bool, n_features: int):
        self.intel.update({
            "transformation_pipeline_path": str(self.transformation_pipeline_path.relative_to(self.project_root)),
            "processor_pipeline_path": str(self.processor_pipeline_path.relative_to(self.project_root)),
            "train_transformed_path": str(self.train_transformed_path.relative_to(self.project_root)),
            "test_transformed_path": str(self.test_transformed_path.relative_to(self.project_root)),
            "feature_engineering_config": {
                "use_feature_tools": use_feature_tools,
                "use_shap_selection": use_shap,
                "n_features_selected": n_features if use_shap else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        with open(self.config_path, 'w') as f:
            yaml.dump(self.intel, f)
        self.logger.info(f"Updated intel.yaml at {self.config_path}")
        return self.intel

    def run(self, use_feature_tools: bool = False, use_shap: bool = False, n_features: int = 20):
        """
        Run the feature engineering process with the specified parameters.
        """
        section("FEATURE ENGINEERING PROCESS", self.logger)
        train_df, test_df = self._load_data()
        X_train, y_train = train_df.drop(columns=[self.target_column]), train_df[self.target_column]
        X_test, y_test = test_df.drop(columns=[self.target_column]), test_df[self.target_column]

        pipeline_steps = []
        if use_feature_tools:
            pipeline_steps.append(('feature_tools', FeatureToolsTransformer(self.target_column)))
        else:
            pipeline_steps.append(('identity', IdentityTransformer()))

        if use_shap:
            pipeline_steps.append(('shap_selector', SHAPFeatureSelector(n_features=n_features)))

        transformation_pipeline = Pipeline(pipeline_steps)
        transformation_pipeline.fit(X_train, y_train)

        try:
            X_train_transformed = transformation_pipeline.transform(X_train).reset_index(drop=True)
            X_test_transformed = transformation_pipeline.transform(X_test).reset_index(drop=True)
            y_train_reset = y_train.reset_index(drop=True)
            y_test_reset = y_test.reset_index(drop=True)

            train_transformed_df = pd.concat([X_train_transformed, y_train_reset], axis=1)
            test_transformed_df = pd.concat([X_test_transformed, y_test_reset], axis=1)

            self._save_data(train_transformed_df, test_transformed_df)
            self._save_pipelines(transformation_pipeline)
            self._log_feature_info(transformation_pipeline, use_feature_tools, use_shap)
            updated_intel = self._update_intel(use_feature_tools, use_shap, n_features)

            return {
                "status": "success",
                "message": "Feature engineering completed successfully",
                "metadata": {
                    "train_shape": train_transformed_df.shape,
                    "test_shape": test_transformed_df.shape,
                    "train_path": str(self.train_transformed_path),
                    "test_path": str(self.test_transformed_path),
                    "pipeline_path": str(self.transformation_pipeline_path),
                    "processor_path": str(self.processor_pipeline_path),
                    "feature_engineering_config": updated_intel.get("feature_engineering_config", {})
                }
            }

        except Exception as e:
            error_msg = f"Error in transformation process: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def _load_data(self):
        try:
            train_path = self.project_root / self.intel.get("train_preprocessed_path")
            test_path = self.project_root / self.intel.get("test_preprocessed_path")
            self.logger.info(f"Loading train data from {train_path}")
            self.logger.info(f"Loading test data from {test_path}")
            return (
                pd.read_csv(train_path),
                pd.read_csv(test_path)
            )
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _save_data(self, train_df, test_df):
        try:
            self.logger.info(f"Saving transformed train data to {self.train_transformed_path}")
            train_df.to_csv(self.train_transformed_path, index=False)
            self.logger.info(f"Saving transformed test data to {self.test_transformed_path}")
            test_df.to_csv(self.test_transformed_path, index=False)
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def _load_cleaning_pipeline(self):
        try:
            cleaning_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/cleaning.pkl"
            self.logger.info(f"Loading cleaning pipeline from {cleaning_path}")
            with open(cleaning_path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cleaning pipeline: {str(e)}")
            raise

    def _load_preprocessing_pipeline(self):
        try:
            preprocessing_path = self.project_root / self.intel.get("preprocessing_pipeline_path")
            self.logger.info(f"Loading preprocessing pipeline from {preprocessing_path}")
            with open(preprocessing_path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading preprocessing pipeline: {str(e)}")
            raise

    def _save_pipelines(self, transformation_pipeline):
        try:
            self.logger.info(f"Saving transformation pipeline to {self.transformation_pipeline_path}")
            with open(self.transformation_pipeline_path, 'wb') as f:
                cloudpickle.dump(transformation_pipeline, f)

            # Load cleaning and preprocessing pipelines
            cleaning_pipeline = self._load_cleaning_pipeline()
            preprocessing_pipeline = self._load_preprocessing_pipeline()

            # Create a combined pipeline with all three components
            processor_pipeline = Pipeline([
                ('cleaning', cleaning_pipeline),
                ('preprocessing', preprocessing_pipeline),
                ('transformation', transformation_pipeline)
            ])

            self.logger.info(f"Saving processor pipeline to {self.processor_pipeline_path}")
            with open(self.processor_pipeline_path, 'wb') as f:
                cloudpickle.dump(processor_pipeline, f)
        except Exception as e:
            self.logger.error(f"Error saving pipelines: {str(e)}")
            raise

    def _log_feature_info(self, pipeline, use_feature_tools, use_shap):
        if use_feature_tools and 'feature_tools' in pipeline.named_steps:
            self.logger.info(f"Generated {len(pipeline.named_steps['feature_tools'].feature_names)} features")

        if use_shap and 'shap_selector' in pipeline.named_steps:
            selector = pipeline.named_steps['shap_selector']
            if selector.importance_df is not None:
                top_features = selector.importance_df.head(10)['feature'].tolist()
                self.logger.info(f"Top 10 features: {top_features}")

    def get_feature_importance(self):
        """
        Return feature importance if SHAP selection was used.

        Returns:
            Dict: Feature importance data or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            if 'shap_selector' in pipeline.named_steps:
                selector = pipeline.named_steps['shap_selector']
                if selector.importance_df is not None:
                    return {
                        "status": "success",
                        "feature_importance": selector.importance_df.to_dict(orient='records')
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Feature importance data not available"
                    }
            else:
                return {
                    "status": "error",
                    "message": "SHAP feature selection was not used"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving feature importance: {str(e)}"
            }

    def get_generated_features(self):
        """
        Return generated features if FeatureTools was used.

        Returns:
            Dict: Generated features data or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            if 'feature_tools' in pipeline.named_steps:
                feature_tools = pipeline.named_steps['feature_tools']
                if feature_tools.feature_names is not None:
                    return {
                        "status": "success",
                        "generated_features": feature_tools.feature_names
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Generated features not available"
                    }
            else:
                return {
                    "status": "error",
                    "message": "FeatureTools was not used"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving generated features: {str(e)}"
            }


# Helper function for API usage
def run_feature_engineering(
        config_path: str = "intel.yaml",
        use_feature_tools: bool = False,
        use_shap: bool = False,
        n_features: int = 20
):
    """
    Run feature engineering with the specified parameters.
    This function can be called from an API endpoint.

    Args:
        config_path: Path to the config file (intel.yaml)
        use_feature_tools: Whether to use FeatureTools for feature generation
        use_shap: Whether to use SHAP for feature selection
        n_features: Number of features to select if using SHAP

    Returns:
        Dict: Result of the feature engineering process
    """
    try:
        engineer = FeatureEngineer(config_path=config_path)
        result = engineer.run(
            use_feature_tools=use_feature_tools,
            use_shap=use_shap,
            n_features=n_features
        )
        return result
    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.error(f"Error in feature engineering: {str(e)}")
        return {
            "status": "error",
            "message": f"Feature engineering failed: {str(e)}"
        }


if __name__ == "__main__":
    # This block is for direct script execution (not API call)
    # It demonstrates how to use the API-friendly version
    try:
        # Get parameters from command line arguments if provided
        import argparse

        parser = argparse.ArgumentParser(description='Run feature engineering process')
        parser.add_argument('--config', default='intel.yaml', help='Path to config file')
        parser.add_argument('--use-feature-tools', action='store_true', help='Use FeatureTools')
        parser.add_argument('--use-shap', action='store_true', help='Use SHAP feature selection')
        parser.add_argument('--n-features', type=int, default=20, help='Number of features to select')
        args = parser.parse_args()

        # Configure logger
        configure_logger()
        logger = logging.getLogger("Feature Engineering")

        # Run feature engineering
        result = run_feature_engineering(
            config_path=args.config,
            use_feature_tools=args.use_feature_tools,
            use_shap=args.use_shap,
            n_features=args.n_features
        )

        if result["status"] == "success":
            logger.info("Feature engineering completed successfully")
        else:
            logger.critical(f"Feature engineering failed: {result['message']}")
            sys.exit(1)

    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.critical(f"Feature engineering failed: {str(e)}")
        sys.exit(1)