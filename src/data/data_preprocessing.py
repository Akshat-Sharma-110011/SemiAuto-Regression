"""
Data Preprocessing Module for SemiAuto-Regression

This module handles the preprocessing of data for the regression model, including:
- Handling missing values
- Handling duplicate values
- Handling outliers

The preprocessing steps are configured based on information in the feature_store.yaml file,
and the preprocessing pipeline is saved for later use.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import cloudpickle
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

# Set up the logger
from src.logger import section, configure_logger

# Configure logger
configure_logger()
logger = logging.getLogger("DataPreprocessing")


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers using either IQR or Z-Score method.
    """

    def __init__(self, method: str = 'IQR', columns: List[str] = None):
        """
        Initialize the OutlierHandler.

        Args:
            method (str): Method to use for outlier detection ('IQR' or 'Z-Score')
            columns (List[str]): List of columns to handle outliers for
        """
        self.method = method
        self.columns = columns
        self.thresholds = {}

        # Validate method
        if method not in ['IQR', 'Z-Score']:
            raise ValueError("Method must be either 'IQR' or 'Z-Score'")

        logger.info(f"Initialized OutlierHandler with method: {method}")

    def fit(self, X, y=None):
        """
        Calculate the thresholds for outlier detection.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for outlier handling")
            return self

        # Calculate thresholds for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            if self.method == 'IQR':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                self.thresholds[col] = {'lower': lower_bound, 'upper': upper_bound}
                logger.info(f"IQR thresholds for {col}: lower={lower_bound:.4f}, upper={upper_bound:.4f}")

            elif self.method == 'Z-Score':
                mean = X[col].mean()
                std = X[col].std()

                self.thresholds[col] = {'mean': mean, 'std': std}
                logger.info(f"Z-Score parameters for {col}: mean={mean:.4f}, std={std:.4f}")

        return self

    def transform(self, X):
        """
        Handle outliers in the data according to the fitted thresholds.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with outliers handled
        """
        X_transformed = X.copy()

        if not self.columns or not self.thresholds:
            logger.warning("No columns or thresholds available for outlier handling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.thresholds:
                continue

            if self.method == 'IQR':
                lower = self.thresholds[col]['lower']
                upper = self.thresholds[col]['upper']

                # Cap outliers at thresholds
                outliers_lower = X_transformed[col] < lower
                outliers_upper = X_transformed[col] > upper

                if outliers_lower.any() or outliers_upper.any():
                    count_lower = outliers_lower.sum()
                    count_upper = outliers_upper.sum()
                    logger.info(
                        f"Handling outliers in {col}: {count_lower} below lower bound, {count_upper} above upper bound")

                    X_transformed.loc[outliers_lower, col] = lower
                    X_transformed.loc[outliers_upper, col] = upper

            elif self.method == 'Z-Score':
                mean = self.thresholds[col]['mean']
                std = self.thresholds[col]['std']

                # Identify outliers (Z-Score > 3 or < -3)
                z_scores = (X_transformed[col] - mean) / std
                outliers = (z_scores > 3) | (z_scores < -3)

                if outliers.any():
                    count = outliers.sum()
                    logger.info(f"Handling {count} Z-Score outliers in {col}")

                    # Cap outliers at ±3 standard deviations
                    X_transformed.loc[z_scores > 3, col] = mean + 3 * std
                    X_transformed.loc[z_scores < -3, col] = mean - 3 * std

        return X_transformed


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling missing values using specified method.
    """

    def __init__(self, method: str = 'mean', columns: List[str] = None):
        """
        Initialize the MissingValueHandler.

        Args:
            method (str): Method to use for handling missing values ('mean', 'median', 'mode', 'drop')
            columns (List[str]): List of columns to handle missing values for
        """
        self.method = method
        self.columns = columns
        self.fill_values = {}

        # Validate method
        if method not in ['mean', 'median', 'mode', 'drop']:
            raise ValueError("Method must be one of 'mean', 'median', 'mode', or 'drop'")

        logger.info(f"Initialized MissingValueHandler with method: {method}")

    def fit(self, X, y=None):
        """
        Calculate the fill values for missing value handling.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Handle drop method separately (nothing to fit)
        if self.method == 'drop':
            return self

        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for missing value handling")
            return self

        # Calculate fill values for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            if self.method == 'mean':
                self.fill_values[col] = X[col].mean()
                logger.info(f"Mean value for {col}: {self.fill_values[col]:.4f}")

            elif self.method == 'median':
                self.fill_values[col] = X[col].median()
                logger.info(f"Median value for {col}: {self.fill_values[col]:.4f}")

            elif self.method == 'mode':
                # Get most frequent value
                self.fill_values[col] = X[col].mode()[0]
                logger.info(f"Mode value for {col}: {self.fill_values[col]}")

        return self

    def transform(self, X):
        """
        Handle missing values in the data according to the fitted values.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with missing values handled
        """
        X_transformed = X.copy()

        if self.method == 'drop':
            # Drop rows with missing values
            rows_before = len(X_transformed)
            X_transformed = X_transformed.dropna(subset=self.columns)
            rows_after = len(X_transformed)
            rows_dropped = rows_before - rows_after

            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows with missing values")

            return X_transformed

        if not self.columns or not self.fill_values:
            logger.warning("No columns or fill values available for missing value handling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.fill_values:
                continue

            # Count missing values before filling
            missing_count = X_transformed[col].isna().sum()

            if missing_count > 0:
                # Fill missing values
                X_transformed[col] = X_transformed[col].fillna(self.fill_values[col])
                logger.info(f"Filled {missing_count} missing values in {col} with {self.fill_values[col]}")

        return X_transformed


class PreprocessingPipeline:
    """
    Main preprocessing pipeline that orchestrates the preprocessing steps.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PreprocessingPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.dataset_name = config.get('dataset_name')
        self.target_column = config.get('target_col')
        self.feature_store = config.get('feature_store', {})
        self.missing_handler = None
        self.outlier_handler = None

        logger.info(f"Initialized PreprocessingPipeline for dataset: {self.dataset_name}")

    def handle_missing_values(self, method: str = 'mean', columns: List[str] = None):
        """
        Set up the missing value handler.

        Args:
            method (str): Method to use for handling missing values
            columns (List[str]): List of columns to handle missing values for
        """
        self.missing_handler = MissingValueHandler(method=method, columns=columns)
        logger.info(f"Set up missing value handler with method: {method}")

    def handle_outliers(self, method: str = 'IQR', columns: List[str] = None):
        """
        Set up the outlier handler.

        Args:
            method (str): Method to use for outlier detection
            columns (List[str]): List of columns to handle outliers for
        """
        self.outlier_handler = OutlierHandler(method=method, columns=columns)
        logger.info(f"Set up outlier handler with method: {method}")

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the data.

        Args:
            df (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        rows_before = len(df)
        df_no_duplicates = df.drop_duplicates()
        rows_after = len(df_no_duplicates)
        rows_dropped = rows_before - rows_after

        if rows_dropped > 0:
            logger.info(f"Removed {rows_dropped} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        return df_no_duplicates

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the preprocessing pipeline on the training data.

        Args:
            X (pd.DataFrame): Training data
        """
        logger.info("Fitting preprocessing pipeline")

        # Fit the missing value handler if defined
        if self.missing_handler:
            self.missing_handler.fit(X)

        # Fit the outlier handler if defined
        if self.outlier_handler:
            self.outlier_handler.fit(X)

    def transform(self, X: pd.DataFrame, handle_duplicates: bool = True) -> pd.DataFrame:
        """
        Transform the data using the fitted preprocessing pipeline.

        Args:
            X (pd.DataFrame): Input data
            handle_duplicates (bool): Whether to handle duplicate rows

        Returns:
            pd.DataFrame: Transformed data
        """
        logger.info("Transforming data with preprocessing pipeline")
        transformed_data = X.copy()

        # Handle duplicates if requested
        if handle_duplicates:
            transformed_data = self.remove_duplicates(transformed_data)

        # Transform with missing value handler if defined
        if self.missing_handler:
            transformed_data = self.missing_handler.transform(transformed_data)

        # Transform with outlier handler if defined
        if self.outlier_handler:
            transformed_data = self.outlier_handler.transform(transformed_data)

        return transformed_data

    def save(self, path: str) -> None:
        """
        Save the preprocessing pipeline using cloudpickle.

        Args:
            path (str): Path to save the pipeline
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            cloudpickle.dump(self, f)

        logger.info(f"Saved preprocessing pipeline to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load a preprocessing pipeline from a file.

        Args:
            path (str): Path to the pipeline file

        Returns:
            PreprocessingPipeline: Loaded pipeline
        """
        with open(path, 'rb') as f:
            pipeline = cloudpickle.load(f)

        logger.info(f"Loaded preprocessing pipeline from {path}")
        return pipeline


def load_yaml(file_path: str) -> Dict:
    """
    Load YAML file into a dictionary.

    Args:
        file_path (str): Path to YAML file

    Returns:
        Dict: Loaded YAML content
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {str(e)}")
        raise


def update_intel_yaml(intel_path: str, updates: Dict) -> None:
    """
    Update the intel.yaml file with new information.

    Args:
        intel_path (str): Path to intel.yaml file
        updates (Dict): Dictionary of updates to apply
    """
    try:
        # Load existing intel
        intel = load_yaml(intel_path)

        # Update with new information
        intel.update(updates)

        # Add processed timestamp
        intel['processed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write back to file
        with open(intel_path, 'w') as file:
            yaml.dump(intel, file, default_flow_style=False)

        logger.info(f"Updated intel.yaml at {intel_path}")
    except Exception as e:
        logger.error(f"Error updating intel.yaml: {str(e)}")
        raise


def check_for_duplicates(df: pd.DataFrame) -> bool:
    """
    Check if dataframe contains duplicate rows.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        bool: True if duplicates exist, False otherwise
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows")
        return True
    else:
        logger.info("No duplicate rows found")
        return False


def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    try:
        section("DATA PREPROCESSING", logger)
        logger.info("Starting data preprocessing")

        # Load intel.yaml
        intel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'intel.yaml')
        intel = load_yaml(intel_path)
        logger.info(f"Loaded intel from {intel_path}")

        # Extract information from intel.yaml
        dataset_name = intel.get('dataset_name')
        feature_store_path = intel.get('feature_store_path')
        train_path = intel.get('train_path')
        test_path = intel.get('test_path')
        target_column = intel.get('target_column')

        # Load feature store
        feature_store = load_yaml(feature_store_path)
        logger.info(f"Loaded feature store from {feature_store_path}")

        # Check for null columns in feature store
        null_columns = feature_store.get('contains_null', [])
        outlier_columns = feature_store.get('contains_outliers', [])

        # Load training and test data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded training data: {train_df.shape} and test data: {test_df.shape}")

        # Initialize preprocessing pipeline
        pipeline_config = {
            'dataset_name': dataset_name,
            'target_col': target_column,
            'feature_store': feature_store
        }
        pipeline = PreprocessingPipeline(pipeline_config)

        # Check for duplicates in training data
        has_duplicates = check_for_duplicates(train_df)

        # Set up paths for output files
        interim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim',
                                   f'data_{dataset_name}')
        os.makedirs(interim_dir, exist_ok=True)

        train_preprocessed_path = os.path.join(interim_dir, 'train_preprocessed.csv')
        test_preprocessed_path = os.path.join(interim_dir, 'test_preprocessed.csv')

        pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'pipelines',
                                    f'preprocessing_{dataset_name}')
        os.makedirs(pipeline_dir, exist_ok=True)

        pipeline_path = os.path.join(pipeline_dir, 'preprocessing.pkl')

        # Interactive preprocessing
        handle_duplicates = True  # Default is to handle duplicates

        # Handle missing values if any
        if null_columns:
            logger.info(f"Found columns with null values: {null_columns}")
            print(f"Found columns with null values: {null_columns}")
            print("How would you like to handle missing values?")
            print("1. Use mean (for numerical columns)")
            print("2. Use median (for numerical columns)")
            print("3. Use mode (most frequent value)")
            print("4. Drop rows with missing values")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                pipeline.handle_missing_values(method='mean', columns=null_columns)
            elif choice == '2':
                pipeline.handle_missing_values(method='median', columns=null_columns)
            elif choice == '3':
                pipeline.handle_missing_values(method='mode', columns=null_columns)
            elif choice == '4':
                pipeline.handle_missing_values(method='drop', columns=null_columns)
            else:
                logger.warning("Invalid choice, using default (mean)")
                pipeline.handle_missing_values(method='mean', columns=null_columns)

        # Handle duplicates if any
        if has_duplicates:
            print("Found duplicate rows in the training data.")
            print("How would you like to handle duplicates?")
            print("1. Drop duplicates")
            print("2. Keep duplicates")

            choice = input("Enter your choice (1-2): ")

            if choice == '1':
                handle_duplicates = True
            elif choice == '2':
                handle_duplicates = False
            else:
                logger.warning("Invalid choice, using default (drop duplicates)")
                handle_duplicates = True

        # Handle outliers if any
        if outlier_columns:
            logger.info(f"Found columns with outliers: {outlier_columns}")
            print(f"Found columns with outliers: {outlier_columns}")
            print("How would you like to handle outliers?")
            print("1. Use IQR method (cap at Q1 - 1.5 * IQR and Q3 + 1.5 * IQR)")
            print("2. Use Z-Score method (cap at mean ± 3 standard deviations)")

            choice = input("Enter your choice (1-2): ")

            if choice == '1':
                pipeline.handle_outliers(method='IQR', columns=outlier_columns)
            elif choice == '2':
                pipeline.handle_outliers(method='Z-Score', columns=outlier_columns)
            else:
                logger.warning("Invalid choice, using default (IQR)")
                pipeline.handle_outliers(method='IQR', columns=outlier_columns)

        section("FITTING PIPELINE", logger)
        # Fit the pipeline on training data
        pipeline.fit(train_df)

        section("TRANSFORMING DATA", logger)
        # Transform training data
        train_preprocessed = pipeline.transform(train_df, handle_duplicates=handle_duplicates)
        logger.info(f"Transformed training data: {train_preprocessed.shape}")

        # Transform test data (without handling duplicates in test data)
        test_preprocessed = pipeline.transform(test_df, handle_duplicates=False)
        logger.info(f"Transformed test data: {test_preprocessed.shape}")

        # Save preprocessed data
        train_preprocessed.to_csv(train_preprocessed_path, index=False)
        test_preprocessed.to_csv(test_preprocessed_path, index=False)
        logger.info(f"Saved preprocessed data to {train_preprocessed_path} and {test_preprocessed_path}")

        # Save the pipeline
        pipeline.save(pipeline_path)

        # Update intel.yaml with new paths
        updates = {
            'train_preprocessed_path': train_preprocessed_path,
            'test_preprocessed_path': test_preprocessed_path,
            'preprocessing_pipeline_path': pipeline_path
        }
        update_intel_yaml(intel_path, updates)

        section("PREPROCESSING COMPLETE", logger)
        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()