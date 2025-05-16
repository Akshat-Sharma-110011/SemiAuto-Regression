"""
Data Preprocessing Module for SemiAuto-Regression

This module handles the preprocessing of data for the regression model, including:
- Handling missing values
- Handling duplicate values
- Handling outliers
- Handling skewed data
- Scaling numerical features
- Encoding categorical features

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
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
import scipy.stats as stats

# Set up the logger
from src.logger import section, configure_logger

with open('intel.yaml', 'r') as f:
    config = yaml.safe_load(f)
    dataset_name = config['dataset_name']

# Configure logger
configure_logger()
logger = logging.getLogger("Data Preprocessing")


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


class SkewedDataHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling skewed data using power transformers.
    """

    def __init__(self, method: str = 'yeo-johnson', columns: List[str] = None):
        """
        Initialize the SkewedDataHandler.

        Args:
            method (str): Method to use for handling skewed data ('yeo-johnson' or 'box-cox')
            columns (List[str]): List of columns to handle skewed data for
        """
        self.method = method
        self.columns = columns
        self.transformers = {}

        # Validate method
        if method not in ['yeo-johnson', 'box-cox']:
            raise ValueError("Method must be either 'yeo-johnson' or 'box-cox'")

        logger.info(f"Initialized SkewedDataHandler with method: {method}")

    def fit(self, X, y=None):
        """
        Fit power transformers for skewed data handling.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for skewed data handling")
            return self

        # Fit transformer for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            # Create and fit transformer
            transformer = PowerTransformer(method=self.method, standardize=True)

            # Box-Cox requires positive values
            if self.method == 'box-cox' and X[col].min() <= 0:
                logger.warning(f"Column {col} has values <= 0, which is incompatible with Box-Cox transform.")
                logger.warning(f"Shifting data to positive values for Box-Cox transformation")
                shift_value = abs(X[col].min()) + 1.0  # Add 1 to ensure all values are positive
                self.transformers[col] = {'transformer': transformer, 'shift': shift_value}
                # Fit transformer on shifted data
                transformer.fit(X[col].add(shift_value).values.reshape(-1, 1))
            else:
                self.transformers[col] = {'transformer': transformer, 'shift': 0.0}
                # Fit transformer
                transformer.fit(X[col].values.reshape(-1, 1))

            logger.info(f"Fitted power transformer for {col}")

        return self

    def transform(self, X):
        """
        Transform skewed data using fitted power transformers.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with skewed data handled
        """
        X_transformed = X.copy()

        if not self.columns or not self.transformers:
            logger.warning("No columns or transformers available for skewed data handling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.transformers:
                continue

            # Get transformer and shift value
            transformer = self.transformers[col]['transformer']
            shift_value = self.transformers[col]['shift']

            # Apply transformation
            if shift_value > 0:
                # Apply shift before transformation
                transformed_values = transformer.transform(X_transformed[col].add(shift_value).values.reshape(-1, 1))
            else:
                transformed_values = transformer.transform(X_transformed[col].values.reshape(-1, 1))

            # Replace original values with transformed values
            X_transformed[col] = transformed_values.flatten()
            logger.info(f"Transformed skewed data in {col}")

        return X_transformed


class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for scaling numerical features.
    """

    def __init__(self, method: str = 'standard', columns: List[str] = None):
        """
        Initialize the NumericalScaler.

        Args:
            method (str): Method to use for scaling ('standard', 'robust', 'minmax')
            columns (List[str]): List of columns to scale
        """
        self.method = method
        self.columns = columns
        self.scalers = {}

        # Validate method
        if method not in ['standard', 'robust', 'minmax']:
            raise ValueError("Method must be one of 'standard', 'robust', or 'minmax'")

        logger.info(f"Initialized NumericalScaler with method: {method}")

    def fit(self, X, y=None):
        """
        Fit scalers for numerical features.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for scaling")
            return self

        # Initialize and fit scaler for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            # Create and fit appropriate scaler
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()

            # Fit scaler
            scaler.fit(X[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
            logger.info(f"Fitted {self.method} scaler for {col}")

        return self

    def transform(self, X):
        """
        Scale numerical features using fitted scalers.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with scaled numerical features
        """
        X_transformed = X.copy()

        if not self.columns or not self.scalers:
            logger.warning("No columns or scalers available for scaling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.scalers:
                continue

            # Get scaler for this column
            scaler = self.scalers[col]

            # Apply scaling
            scaled_values = scaler.transform(X_transformed[col].values.reshape(-1, 1))
            X_transformed[col] = scaled_values.flatten()
            logger.info(f"Scaled numerical features in {col}")

        return X_transformed


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding categorical features.
    """

    def __init__(self, method: str = 'onehot', columns: List[str] = None, drop_first: bool = True):
        """
        Initialize the CategoricalEncoder.

        Args:
            method (str): Method to use for encoding ('onehot', 'label', 'dummies')
            columns (List[str]): List of columns to encode
            drop_first (bool): Whether to drop the first category in one-hot encoding
        """
        self.method = method
        self.columns = columns
        self.drop_first = drop_first
        self.encoders = {}
        self.dummy_columns = {}

        # Validate method
        if method not in ['onehot', 'label', 'dummies']:
            raise ValueError("Method must be one of 'onehot', 'label', or 'dummies'")

        logger.info(f"Initialized CategoricalEncoder with method: {method}")

    def fit(self, X, y=None):
        """
        Fit encoders for categorical features.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for categorical encoding")
            return self

        # Initialize and fit encoder for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            if self.method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, drop='first' if self.drop_first else None)
                encoder.fit(X[col].values.reshape(-1, 1))
                self.encoders[col] = encoder
                logger.info(f"Fitted OneHotEncoder for {col}")

                # Store feature names for later
                feature_names = encoder.get_feature_names_out([col])
                self.dummy_columns[col] = feature_names.tolist()

            elif self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders[col] = encoder
                logger.info(f"Fitted LabelEncoder for {col}")

            elif self.method == 'dummies':
                # For 'dummies' method, we'll just store unique values
                # actual transformation will be done in transform method
                unique_values = X[col].unique()
                self.encoders[col] = unique_values

                # Create dummy column names (will be used during transform)
                dummy_cols = [f"{col}_{val}" for val in unique_values]
                if self.drop_first:
                    dummy_cols = dummy_cols[1:]
                self.dummy_columns[col] = dummy_cols

                logger.info(f"Stored unique values for pd.get_dummies for {col}")

        return self

    def transform(self, X):
        """
        Encode categorical features using fitted encoders.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with encoded categorical features
        """
        X_transformed = X.copy()

        if not self.columns or not self.encoders:
            logger.warning("No columns or encoders available for categorical encoding")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.encoders:
                continue

            if self.method == 'onehot':
                # Apply one-hot encoding
                encoder = self.encoders[col]
                encoded_array = encoder.transform(X_transformed[col].values.reshape(-1, 1))

                # Create DataFrame with encoded values
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.dummy_columns[col],
                    index=X_transformed.index
                )

                # Drop original column and join encoded columns
                X_transformed = X_transformed.drop(columns=[col])
                X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                logger.info(f"Applied OneHotEncoder to {col}, created {len(self.dummy_columns[col])} new columns")

            elif self.method == 'label':
                # Apply label encoding
                encoder = self.encoders[col]
                X_transformed[col] = encoder.transform(X_transformed[col])
                logger.info(f"Applied LabelEncoder to {col}")

            elif self.method == 'dummies':
                # Use pandas.get_dummies
                dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=self.drop_first, dtype=int)

                # Drop original column and join dummy columns
                X_transformed = X_transformed.drop(columns=[col])
                X_transformed = pd.concat([X_transformed, dummies], axis=1)

                # Check for any missing dummy columns that were in the training data
                expected_dummy_cols = set(self.dummy_columns[col])
                actual_dummy_cols = set([col for col in dummies.columns])

                # Add missing dummy columns (with all zeros)
                for missing_col in expected_dummy_cols - actual_dummy_cols:
                    X_transformed[missing_col] = 0

                logger.info(f"Applied pd.get_dummies to {col}, created {dummies.shape[1]} new columns")

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
        self.skewed_handler = None
        self.numerical_scaler = None
        self.categorical_encoder = None

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

    def handle_skewed_data(self, method: str = 'yeo-johnson', columns: List[str] = None):
        """
        Set up the skewed data handler.

        Args:
            method (str): Method to use for handling skewed data
            columns (List[str]): List of columns to handle skewed data for
        """
        self.skewed_handler = SkewedDataHandler(method=method, columns=columns)
        logger.info(f"Set up skewed data handler with method: {method}")

    def scale_numerical_features(self, method: str = 'standard', columns: List[str] = None):
        """
        Set up the numerical scaler.

        Args:
            method (str): Method to use for scaling numerical features
            columns (List[str]): List of columns to scale
        """
        self.numerical_scaler = NumericalScaler(method=method, columns=columns)
        logger.info(f"Set up numerical scaler with method: {method}")

    def encode_categorical_features(self, method: str = 'onehot', columns: List[str] = None, drop_first: bool = True):
        """
        Set up the categorical encoder.

        Args:
            method (str): Method to use for encoding categorical features
            columns (List[str]): List of columns to encode
            drop_first (bool): Whether to drop the first category in one-hot encoding
        """
        self.categorical_encoder = CategoricalEncoder(method=method, columns=columns, drop_first=drop_first)
        logger.info(f"Set up categorical encoder with method: {method}")

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

        # Fit the skewed data handler if defined
        if self.skewed_handler:
            self.skewed_handler.fit(X)

        # Fit the numerical scaler if defined
        if self.numerical_scaler:
            self.numerical_scaler.fit(X)

        # Fit the categorical encoder if defined
        if self.categorical_encoder:
            self.categorical_encoder.fit(X)

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

        # Transform with skewed data handler if defined
        if self.skewed_handler:
            transformed_data = self.skewed_handler.transform(transformed_data)

        # Transform with numerical scaler if defined
        if self.numerical_scaler:
            transformed_data = self.numerical_scaler.transform(transformed_data)

        # Transform with categorical encoder if defined
        if self.categorical_encoder:
            transformed_data = self.categorical_encoder.transform(transformed_data)

        if self.target_column and self.target_column in transformed_data.columns:
            # Extract and reinsert the target column to the end
            target_data = transformed_data.pop(self.target_column)
            transformed_data[self.target_column] = target_data
            logger.info(f"Target column '{self.target_column}' moved to the last position")

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


def check_for_skewness(df: pd.DataFrame, columns: List[str], threshold: float = 0.5) -> Dict[str, float]:
    """
    Check for skewness in the specified columns.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for skewness
        threshold (float): Skewness threshold (abs value) to consider a column skewed

    Returns:
        Dict[str, float]: Dictionary with column names as keys and skewness values as values
    """
    skewed_columns = {}

    for col in columns:
        if col not in df.columns:
            continue

        skewness = df[col].skew()
        if abs(skewness) > threshold:
            skewed_columns[col] = skewness
            logger.info(f"Column {col} is skewed with skewness value: {skewness:.4f}")

    return skewed_columns


def get_numerical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get list of numerical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude (List[str]): Columns to exclude

    Returns:
        List[str]: List of numerical columns
    """
    if exclude is None:
        exclude = []

    # Get columns with numeric dtype
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude specified columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get list of categorical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude (List[str]): Columns to exclude

    Returns:
        List[str]: List of categorical columns
    """
    if exclude is None:
        exclude = []

    # Get columns with object or category dtype
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude specified columns
    cat_cols = [col for col in cat_cols if col not in exclude]

    return cat_cols


def recommend_skewness_transformer(df: pd.DataFrame, column: str) -> str:
    """
    Recommend the best transformer for a skewed column.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check

    Returns:
        str: Recommended transformer ('yeo-johnson' or 'box-cox')
    """
    # Check if column contains negative or zero values
    if df[column].min() <= 0:
        logger.info(f"Column {column} contains negative or zero values, recommending Yeo-Johnson transformation")
        return 'yeo-johnson'

    # Check the skewness after both transformations
    # Create a sample to test transformations (for speed)
    sample = df[column].sample(min(1000, len(df))).copy()

    # Test Yeo-Johnson
    try:
        yj_transformer = PowerTransformer(method='yeo-johnson')
        yj_transformed = yj_transformer.fit_transform(sample.values.reshape(-1, 1)).flatten()
        yj_skewness = stats.skew(yj_transformed)
    except Exception as e:
        logger.warning(f"Error testing Yeo-Johnson transformation: {str(e)}")
        yj_skewness = float('inf')

    # Test Box-Cox
    try:
        bc_transformer = PowerTransformer(method='box-cox')
        bc_transformed = bc_transformer.fit_transform(sample.values.reshape(-1, 1)).flatten()
        bc_skewness = stats.skew(bc_transformed)
    except Exception as e:
        logger.warning(f"Error testing Box-Cox transformation: {str(e)}")
        bc_skewness = float('inf')

    # Compare and recommend
    if abs(bc_skewness) <= abs(yj_skewness):
        logger.info(f"Box-Cox transformation recommended for {column} (skewness: {bc_skewness:.4f} vs {yj_skewness:.4f})")
        return 'box-cox'
    else:
        logger.info(f"Yeo-Johnson transformation recommended for {column} (skewness: {yj_skewness:.4f} vs {bc_skewness:.4f})")
        return 'yeo-johnson'


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

        # Check for special columns in feature store and exclude target column
        null_columns = [col for col in feature_store.get('contains_null', []) if col != target_column]
        outlier_columns = [col for col in feature_store.get('contains_outliers', []) if col != target_column]
        skewed_columns = [col for col in feature_store.get('skewed_cols', []) if col != target_column]
        categorical_columns = [col for col in feature_store.get('categorical_cols', []) if col != target_column]

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

        # If categorical columns not specified in feature store, detect them automatically
        if not categorical_columns:
            categorical_columns = get_categorical_columns(train_df, exclude=[target_column])
            logger.info(f"Auto-detected categorical columns: {categorical_columns}")

        # If skewed columns not specified in feature store, detect them automatically
        if not skewed_columns:
            numerical_columns = get_numerical_columns(train_df, exclude=[target_column])
            skewness_dict = check_for_skewness(train_df, numerical_columns)
            skewed_columns = list(skewness_dict.keys())
            logger.info(f"Auto-detected skewed columns: {skewed_columns}")

        # Set up paths for output files
        interim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim',
                                   f'data_{dataset_name}')
        os.makedirs(interim_dir, exist_ok=True)

        train_preprocessed_path = os.path.join(interim_dir, 'train_preprocessed.csv')
        test_preprocessed_path = os.path.join(interim_dir, 'test_preprocessed.csv')

        pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'pipelines',
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

        # Handle skewed data if any
        if skewed_columns:
            logger.info(f"Found columns with skewed distributions: {skewed_columns}")
            print(f"Found columns with skewed distributions: {skewed_columns}")

            # Show recommended transformer for each skewed column
            print("\nRecommended transformers for each skewed column:")
            recommended_transformers = {}
            for col in skewed_columns:
                recommended = recommend_skewness_transformer(train_df, col)
                recommended_transformers[col] = recommended
                print(f"  - {col}: {recommended}")

            print("\nHow would you like to handle skewed data?")
            print("1. Use Yeo-Johnson transformation (works with negative values)")
            print("2. Use Box-Cox transformation (requires positive values)")
            print("3. Use recommended transformer for each column")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)
            elif choice == '2':
                pipeline.handle_skewed_data(method='box-cox', columns=skewed_columns)
            elif choice == '3':
                # We'll use the recommended transformer
                counts = {'yeo-johnson': 0, 'box-cox': 0}
                for col, transformer in recommended_transformers.items():
                    counts[transformer] += 1

                # Use the most recommended transformer
                if counts['box-cox'] > counts['yeo-johnson']:
                    pipeline.handle_skewed_data(method='box-cox', columns=skewed_columns)
                else:
                    pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)
            else:
                logger.warning("Invalid choice, using default (Yeo-Johnson)")
                pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)

        # Scale numerical features
        numerical_columns = get_numerical_columns(train_df, exclude=[target_column])
        if numerical_columns:
            logger.info(f"Found numerical columns: {numerical_columns}")
            print(f"\nFound numerical columns: {numerical_columns}")
            print("Would you like to scale these numerical features?")
            print("1. Yes, use StandardScaler (mean=0, std=1)")
            print("2. Yes, use RobustScaler (median=0, IQR=1, robust to outliers)")
            print("3. Yes, use MinMaxScaler (scale to range [0,1])")
            print("4. No, do not scale numerical features")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                pipeline.scale_numerical_features(method='standard', columns=numerical_columns)
            elif choice == '2':
                pipeline.scale_numerical_features(method='robust', columns=numerical_columns)
            elif choice == '3':
                pipeline.scale_numerical_features(method='minmax', columns=numerical_columns)
            elif choice == '4':
                logger.info("Skipping numerical feature scaling")
            else:
                logger.warning("Invalid choice, skipping numerical feature scaling")

        # Encode categorical features
        if categorical_columns:
            logger.info(f"Found categorical columns: {categorical_columns}")
            print(f"\nFound categorical columns: {categorical_columns}")
            print("How would you like to encode these categorical features?")
            print("1. Use OneHotEncoder (sklearn)")
            print("2. Use pd.get_dummies (pandas)")
            print("3. Use LabelEncoder (convert to integers)")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                # Ask about dropping first category
                drop_first = input("Drop first category to avoid multicollinearity? (y/n): ").lower() == 'y'
                pipeline.encode_categorical_features(method='onehot', columns=categorical_columns, drop_first=drop_first)
            elif choice == '2':
                # Ask about dropping first category
                drop_first = input("Drop first category to avoid multicollinearity? (y/n): ").lower() == 'y'
                pipeline.encode_categorical_features(method='dummies', columns=categorical_columns, drop_first=drop_first)
            elif choice == '3':
                pipeline.encode_categorical_features(method='label', columns=categorical_columns)
            else:
                logger.warning("Invalid choice, using default (OneHotEncoder)")
                pipeline.encode_categorical_features(method='onehot', columns=categorical_columns)

        # Collect preprocessing configuration from the pipeline
        preprocessing_config = {}

        # Missing values handling
        if pipeline.missing_handler:
            preprocessing_config['missing_values'] = {
                'method': pipeline.missing_handler.method,
                'columns': pipeline.missing_handler.columns
            }

        # Outlier handling
        if pipeline.outlier_handler:
            preprocessing_config['outliers'] = {
                'method': pipeline.outlier_handler.method,
                'columns': pipeline.outlier_handler.columns
            }

        # Skewed data handling
        if pipeline.skewed_handler:
            preprocessing_config['skewed_data'] = {
                'method': pipeline.skewed_handler.method,
                'columns': pipeline.skewed_handler.columns
            }

        # Numerical scaling
        if pipeline.numerical_scaler:
            preprocessing_config['numerical_scaling'] = {
                'method': pipeline.numerical_scaler.method,
                'columns': pipeline.numerical_scaler.columns
            }

        # Categorical encoding
        if pipeline.categorical_encoder:
            preprocessing_config['categorical_encoding'] = {
                'method': pipeline.categorical_encoder.method,
                'columns': pipeline.categorical_encoder.columns,
                'drop_first': pipeline.categorical_encoder.drop_first
            }

        # Duplicates handling
        preprocessing_config['handle_duplicates'] = handle_duplicates

        section("FITTING PIPELINE", logger)
        # Fit the pipeline on training data
        pipeline.fit(train_df)

        section("TRANSFORMING DATA", logger)
        # Transform training data
        train_preprocessed = pipeline.transform(train_df, handle_duplicates=handle_duplicates)
        # Ensure target column is the last column
        if target_column in train_preprocessed.columns:
            cols = [col for col in train_preprocessed.columns if col != target_column] + [target_column]
            train_preprocessed = train_preprocessed[cols]
        logger.info(f"Transformed training data: {train_preprocessed.shape}")

        # Transform test data (without handling duplicates in test data)
        test_preprocessed = pipeline.transform(test_df, handle_duplicates=False)
        # Ensure target column is the last column
        if target_column in test_preprocessed.columns:
            cols = [col for col in test_preprocessed.columns if col != target_column] + [target_column]
            test_preprocessed = test_preprocessed[cols]
        logger.info(f"Transformed test data: {test_preprocessed.shape}")

        train_preprocessed.to_csv(train_preprocessed_path, index=False)
        test_preprocessed.to_csv(test_preprocessed_path, index=False)

        # Save the pipeline
        pipeline.save(pipeline_path)

        # Update intel.yaml with new paths and preprocessing configuration
        updates = {
            'train_preprocessed_path': train_preprocessed_path,
            'test_preprocessed_path': test_preprocessed_path,
            'preprocessing_pipeline_path': pipeline_path,
            'preprocessing_config': preprocessing_config
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