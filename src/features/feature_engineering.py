#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering script for regression automl clone.
This module handles automatic feature generation, transformation pipeline creation,
and integration with preprocessing pipeline.
"""

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

# Add parent directory to path for importing custom logger
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import custom logger
import logging
from src.logger import section, configure_logger

# Configure logger
configure_logger()
logger = logging.getLogger("Data Preprocessing")


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that returns the data unchanged.
    Used when no feature engineering is desired.
    """

    def __init__(self):
        logger.info("Initializing IdentityTransformer")

    def fit(self, X, y=None):
        logger.info("Identity transformer fit called - no action required")
        return self

    def transform(self, X):
        logger.info("Identity transformer transform called - returning data unchanged")
        return X


class FeatureToolsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that uses featuretools to create new features.
    """

    def __init__(self, target_col: str):
        logger.info("Initializing FeatureToolsTransformer")
        self.target_col = target_col
        self.feature_defs = None
        self.feature_names = None

        try:
            import featuretools as ft
            self.ft = ft
        except ImportError:
            logger.error("Featuretools package not found. Please install with: pip install featuretools")
            raise ImportError("Featuretools package not found")

    def fit(self, X, y=None):
        """
        Use featuretools to create feature definitions from the training data.

        Args:
            X: DataFrame containing features
            y: Series containing target variable (not used directly)

        Returns:
            Self
        """
        logger.info("FeatureToolsTransformer fit starting")

        try:
            # Create an EntitySet and add the data
            es = self.ft.EntitySet(id="features")

            # Make a copy to avoid modifying the original data
            X_copy = X.copy()

            # Create unique index if needed
            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                index_name = "index"
            else:
                index_name = X_copy.index.name

            # Add the data to the EntitySet
            es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index=index_name,
                make_index=True,
                time_index=None
            )

            # Generate feature definitions
            feature_matrix, feature_defs = self.ft.dfs(
                entityset=es,
                target_dataframe_name="data",
                trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric", "subtract_numeric"],
                max_depth=1,
                features_only=False,
                verbose=True
            )

            # Save the feature definitions for transform
            self.feature_defs = feature_defs

            # Filter out the target column from feature names if it exists
            feature_columns = [col for col in feature_matrix.columns if self.target_col not in col]
            self.feature_names = feature_columns

            logger.info(f"Generated {len(feature_columns)} features using featuretools")
            logger.debug(f"Feature names: {feature_columns[:10]}... (truncated)")

            return self

        except Exception as e:
            logger.error(f"Error in FeatureToolsTransformer fit: {str(e)}")
            raise

    def transform(self, X):
        """
        Transform the data using the feature definitions from fit.

        Args:
            X: DataFrame containing features

        Returns:
            DataFrame with original and generated features
        """
        logger.info("FeatureToolsTransformer transform starting")

        try:
            if self.feature_defs is None:
                logger.error("Transform called before fit. Please call fit first.")
                raise ValueError("Transform called before fit")

            # Make a copy to avoid modifying the original data
            X_copy = X.copy()

            # Create unique index if needed
            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                index_name = "index"
            else:
                index_name = X_copy.index.name

            # Create an EntitySet and add the data
            es = self.ft.EntitySet(id="features_transform")
            es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index=index_name,
                make_index=True,
                time_index=None
            )

            # Calculate the feature matrix using the saved feature definitions
            feature_matrix = self.ft.calculate_feature_matrix(
                features=self.feature_defs,
                entityset=es,
                verbose=True
            )

            # Replace infinite values with NaN then with large numbers
            feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
            feature_matrix = feature_matrix.fillna(0)  # Fill NaN with 0

            # Only keep the columns we identified during fit
            feature_matrix = feature_matrix[self.feature_names]

            # Drop any columns that still contain non-numeric values
            for col in feature_matrix.columns:
                if not pd.api.types.is_numeric_dtype(feature_matrix[col]):
                    logger.warning(f"Dropping non-numeric column: {col}")
                    feature_matrix = feature_matrix.drop(columns=[col])

            logger.info(f"Transformed data shape: {feature_matrix.shape}")
            return feature_matrix

        except Exception as e:
            logger.error(f"Error in FeatureToolsTransformer transform: {str(e)}")
            raise


class FeatureEngineer:
    """
    Main class for feature engineering process.
    """

    def __init__(self):
        """Initialize the feature engineer with configuration from intel.yaml"""
        section("FEATURE ENGINEERING INITIALIZATION", logger)
        logger.info("Initializing Feature Engineering process")

        # Load configuration
        self.intel = self._load_intel()
        self.feature_store = self._load_feature_store()

        # Extract needed paths
        self.dataset_name = self.intel.get("dataset_name")
        self.target_column = self.intel.get("target_column")
        self.train_preprocessed_path = self.intel.get("train_preprocessed_path")
        self.test_preprocessed_path = self.intel.get("test_preprocessed_path")
        self.preprocessing_pipeline_path = self.intel.get("preprocessing_pipeline_path")

        # Define output paths
        self.transformation_pipeline_path = os.path.join(
            os.path.dirname(self.preprocessing_pipeline_path),
            "transformation.pkl"
        )
        self.processor_pipeline_path = os.path.join(
            os.path.dirname(self.preprocessing_pipeline_path),
            "processor.pkl"
        )

        # Define transformed data paths
        base_dir = os.path.abspath(os.path.join(
            os.path.dirname(self.train_preprocessed_path),
            "../..",
            "processed",
            f"data_{self.dataset_name}"
        ))
        os.makedirs(base_dir, exist_ok=True)

        self.train_transformed_path = os.path.join(base_dir, "train_transformed.csv")
        self.test_transformed_path = os.path.join(base_dir, "test_transformed.csv")

        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Preprocessing pipeline path: {self.preprocessing_pipeline_path}")
        logger.info(f"Transformation pipeline will be saved to: {self.transformation_pipeline_path}")
        logger.info(f"Processor pipeline will be saved to: {self.processor_pipeline_path}")
        logger.info(f"Transformed train data will be saved to: {self.train_transformed_path}")
        logger.info(f"Transformed test data will be saved to: {self.test_transformed_path}")

    def _load_intel(self) -> Dict:
        """
        Load intel.yaml containing project configuration.

        Returns:
            Dictionary with project configuration
        """
        logger.info("Loading intel.yaml")

        try:
            # Find the intel.yaml in the main project folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            intel_path = os.path.join(project_root, "intel.yaml")

            with open(intel_path, 'r') as f:
                intel = yaml.safe_load(f)

            logger.info(f"Intel loaded successfully from {intel_path}")
            return intel

        except Exception as e:
            logger.error(f"Error loading intel.yaml: {str(e)}")
            raise

    def _load_feature_store(self) -> Dict:
        """
        Load feature store YAML containing dataset information.

        Returns:
            Dictionary with feature store information
        """
        logger.info("Loading feature store")

        try:
            feature_store_path = self.intel.get("feature_store_path")
            with open(feature_store_path, 'r') as f:
                feature_store = yaml.safe_load(f)

            logger.info(f"Feature store loaded successfully from {feature_store_path}")
            return feature_store

        except Exception as e:
            logger.error(f"Error loading feature store: {str(e)}")
            raise

    def _update_intel(self) -> None:
        """Update intel.yaml with new file paths"""
        try:
            # Find the intel.yaml in the main project folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            intel_path = os.path.join(project_root, "intel.yaml")

            # Update the intel dictionary
            self.intel["transformation_pipeline_path"] = self.transformation_pipeline_path
            self.intel["processor_pipeline_path"] = self.processor_pipeline_path
            self.intel["train_transformed_path"] = self.train_transformed_path
            self.intel["test_transformed_path"] = self.test_transformed_path

            # Write the updated intel dictionary back to the file
            with open(intel_path, 'w') as f:
                yaml.dump(self.intel, f, default_flow_style=False)

            logger.info(f"Updated intel.yaml with new paths")

        except Exception as e:
            logger.error(f"Error updating intel.yaml: {str(e)}")
            raise

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed train and test data.

        Returns:
            Tuple containing train and test DataFrames
        """
        section("LOADING PREPROCESSED DATA", logger)

        try:
            logger.info(f"Loading train data from {self.train_preprocessed_path}")
            train_df = pd.read_csv(self.train_preprocessed_path)

            logger.info(f"Loading test data from {self.test_preprocessed_path}")
            test_df = pd.read_csv(self.test_preprocessed_path)

            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_preprocessing_pipeline(self):
        """
        Load the preprocessing pipeline.

        Returns:
            Preprocessing pipeline object
        """
        logger.info(f"Loading preprocessing pipeline from {self.preprocessing_pipeline_path}")

        try:
            with open(self.preprocessing_pipeline_path, 'rb') as f:
                preprocessing_pipeline = cloudpickle.load(f)

            logger.info("Preprocessing pipeline loaded successfully")
            return preprocessing_pipeline

        except Exception as e:
            logger.error(f"Error loading preprocessing pipeline: {str(e)}")
            raise

    def _save_pipeline(self, pipeline, path: str) -> None:
        """
        Save a pipeline to disk using cloudpickle.

        Args:
            pipeline: Pipeline object to save
            path: Path where to save the pipeline
        """
        logger.info(f"Saving pipeline to {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            with open(path, 'wb') as f:
                cloudpickle.dump(pipeline, f)

            logger.info(f"Pipeline saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving pipeline: {str(e)}")
            raise

    def _save_dataframe(self, df: pd.DataFrame, path: str) -> None:
        """
        Save a DataFrame to disk as CSV.

        Args:
            df: DataFrame to save
            path: Path where to save the DataFrame
        """
        logger.info(f"Saving DataFrame to {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            df.to_csv(path, index=False)
            logger.info(f"DataFrame saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving DataFrame: {str(e)}")
            raise

    def run(self) -> None:
        """
        Run the feature engineering process.
        """
        section("FEATURE ENGINEERING PROCESS", logger)

        # Load data
        train_df, test_df = self._load_data()

        # Separate target column from features
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]
        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")

        # Ask user whether to use feature tools
        use_feature_tools = input(
            "Do you want to use FeatureTools for automatic feature engineering? (yes/no): ").strip().lower()

        # Create transformation pipeline based on user choice
        if use_feature_tools in ["yes", "y"]:
            logger.info("User opted to use FeatureTools for feature engineering")
            transformation_pipeline = Pipeline([
                ('feature_tools', FeatureToolsTransformer(target_col=self.target_column))
            ])
        else:
            logger.info("User opted not to use feature engineering - using identity transformer")
            transformation_pipeline = Pipeline([
                ('identity', IdentityTransformer())
            ])

        # Fit the transformation pipeline on training data
        section("FITTING TRANSFORMATION PIPELINE", logger)
        try:
            transformation_pipeline.fit(X_train, y_train)
            logger.info("Transformation pipeline fit successfully")
        except Exception as e:
            logger.error(f"Error fitting transformation pipeline: {str(e)}")
            raise

        # Transform the data
        section("TRANSFORMING DATA", logger)
        try:
            X_train_transformed = transformation_pipeline.transform(X_train)
            X_test_transformed = transformation_pipeline.transform(X_test)

            logger.info(f"Transformed X_train shape: {X_train_transformed.shape}")
            logger.info(f"Transformed X_test shape: {X_test_transformed.shape}")

            # Convert to DataFrame if needed
            if not isinstance(X_train_transformed, pd.DataFrame):
                logger.info("Converting transformed arrays to DataFrames")
                X_train_transformed = pd.DataFrame(X_train_transformed)
                X_test_transformed = pd.DataFrame(X_test_transformed)

            # Ensure all columns have string names
            X_train_transformed.columns = [str(col) for col in X_train_transformed.columns]
            X_test_transformed.columns = [str(col) for col in X_test_transformed.columns]

            # Add target column back
            train_transformed_df = X_train_transformed.copy()
            train_transformed_df[self.target_column] = y_train.values

            test_transformed_df = X_test_transformed.copy()
            test_transformed_df[self.target_column] = y_test.values

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise

        # Clean data to ensure only numeric values with no infinities or nulls
        section("CLEANING TRANSFORMED DATA", logger)
        try:
            # Check for non-numeric columns and remove them
            for col in train_transformed_df.columns:
                if not pd.api.types.is_numeric_dtype(train_transformed_df[col]):
                    logger.warning(f"Dropping non-numeric column: {col}")
                    if col != self.target_column:  # Keep the target column
                        train_transformed_df = train_transformed_df.drop(columns=[col])
                        test_transformed_df = test_transformed_df.drop(columns=[col])

            # Replace infinities with large numbers
            train_transformed_df = train_transformed_df.replace([np.inf, -np.inf], np.nan)
            test_transformed_df = test_transformed_df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values
            train_transformed_df = train_transformed_df.fillna(0)
            test_transformed_df = test_transformed_df.fillna(0)

            logger.info(f"Final cleaned train data shape: {train_transformed_df.shape}")
            logger.info(f"Final cleaned test data shape: {test_transformed_df.shape}")

        except Exception as e:
            logger.error(f"Error cleaning transformed data: {str(e)}")
            raise

        # Save transformed data
        section("SAVING RESULTS", logger)
        try:
            self._save_dataframe(train_transformed_df, self.train_transformed_path)
            self._save_dataframe(test_transformed_df, self.test_transformed_path)

            # Save transformation pipeline
            self._save_pipeline(transformation_pipeline, self.transformation_pipeline_path)

            # Load preprocessing pipeline
            preprocessing_pipeline = self._load_preprocessing_pipeline()

            # Create processor pipeline (preprocessing + transformation)
            # Check if preprocessing_pipeline has steps attribute or is a custom class
            if hasattr(preprocessing_pipeline, 'steps'):
                # It's a sklearn Pipeline
                processor_steps = []
                for name, transformer in preprocessing_pipeline.steps:
                    processor_steps.append((f"preproc_{name}", transformer))

                for name, transformer in transformation_pipeline.steps:
                    processor_steps.append((f"transform_{name}", transformer))

                processor_pipeline = Pipeline(processor_steps)
            else:
                # It's a custom PreprocessingPipeline class
                # Create a new pipeline that first applies preprocessing then transformation
                processor_pipeline = Pipeline([
                    ('preprocessing', preprocessing_pipeline),
                    ('transformation', transformation_pipeline)
                ])

            # Save processor pipeline
            self._save_pipeline(processor_pipeline, self.processor_pipeline_path)

            # Update intel.yaml
            self._update_intel()

            logger.info("Feature engineering process completed successfully")

        except Exception as e:
            logger.error(f"Error in saving results: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        feature_engineer = FeatureEngineer()
        feature_engineer.run()
    except Exception as e:
        logger.critical(f"Feature engineering process failed: {str(e)}")
        sys.exit(1)