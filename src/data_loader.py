# ==============================================================================
# Script: data_loader.py
# Purpose: Loads vehicle dataset, validates features, applies OneHotEncoder
#          and StandardScaler via ColumnTransformer, and creates train/test splits.
# ==============================================================================

# Import operating system library for resolving paths across Windows/Linux/Mac
import os

# Import system module to configure Windows console encoding
import sys

# Ensure UTF-8 output encoding for Windows PowerShell consoles so emojis and symbols display correctly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import Pandas library for tabular data processing, DataFrame manipulation, and CSV loading
import pandas as pd

# Import NumPy library for numerical arrays and matrix operations
import numpy as np

# Import train_test_split from scikit-learn to divide data into training and evaluation sets
from sklearn.model_selection import train_test_split

# Import StandardScaler to standardize numerical continuous features (Mean=0, StdDev=1)
from sklearn.preprocessing import StandardScaler

# Import OneHotEncoder to convert text categories into binary dummy indicator vectors
from sklearn.preprocessing import OneHotEncoder

# Import ColumnTransformer to apply different preprocessing pipelines to specific column subsets
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------------------------
# File Path Configurations
# ------------------------------------------------------------------------------

# Define the absolute path pointing directly to 'data/vehicle_fuel_dataset.csv'
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", 
    "vehicle_fuel_dataset.csv"
)

# ------------------------------------------------------------------------------
# Feature Columns & Target Variable Definition
# ------------------------------------------------------------------------------

# List of 6 numerical physical features used to calculate vehicle fuel consumption
NUMERICAL_COLS = [
    'Engine Size',      # Engine displacement volume in Liters (L)
    'Cylinders',        # Number of cylinders (3, 4, 6, 8, 12)
    'Horsepower',       # Engine brake horsepower output (HP)
    'Vehicle Weight',   # Vehicle curb mass in kilograms (kg)
    'Acceleration',     # 0 to 100 km/h acceleration time in seconds (s)
    'Model Year'        # Vehicle manufacturing year (2012 - 2024)
]

# List of 2 categorical features representing powertrain configuration
CATEGORICAL_COLS = [
    'Fuel Type',        # Petrol, Diesel, Hybrid, Plug-in Hybrid
    'Transmission'      # Automatic, Manual
]

# Master list of all 8 input features expected by the Machine Learning model
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# The continuous target variable we want to predict (Fuel consumption in Liters / 100 km)
TARGET_COL = 'Fuel Consumption'


# ------------------------------------------------------------------------------
# Function: load_raw_data
# ------------------------------------------------------------------------------
def load_raw_data(data_path=DATA_PATH):
    """
    Loads the raw vehicle dataset from CSV file on disk.
    If the file is missing, automatically invokes download_dataset() to generate or fetch it.
    """
    # Check if the dataset CSV file exists at the given path
    if not os.path.exists(data_path):
        print(f"⚠️ Dataset not found at '{data_path}'. Triggering automated data acquisition...")
        try:
            # Attempt to import and run download_dataset from src.download_data
            from src.download_data import download_dataset
            df = download_dataset()
            return df
        except ImportError:
            # Fallback if executing directly inside src directory
            from download_data import download_dataset
            df = download_dataset()
            return df

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(data_path)

    # Return the loaded DataFrame
    return df


# ------------------------------------------------------------------------------
# Function: preprocess_data
# ------------------------------------------------------------------------------
def preprocess_data(df):
    """
    Validates and separates the input feature matrix X and continuous target vector y:
    1. Validates that all required feature columns exist.
    2. Separates independent features (X) from the dependent target (y).
    3. Handles missing values if present (imputes medians/modes).
    """
    # Make a clean defensive copy to avoid mutating the original DataFrame in memory
    clean_df = df.copy()

    # Verify that the target column exists in the dataset
    if TARGET_COL not in clean_df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset columns: {list(clean_df.columns)}")

    # Check for and handle any missing values in numerical columns using median imputation
    for col in NUMERICAL_COLS:
        if col in clean_df.columns and clean_df[col].isnull().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # Check for and handle any missing values in categorical columns using mode imputation
    for col in CATEGORICAL_COLS:
        if col in clean_df.columns and clean_df[col].isnull().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])

    # Extract the independent feature matrix X containing 8 automotive parameters
    X = clean_df[FEATURE_COLS]

    # Extract the dependent target vector y containing continuous fuel consumption values
    y = clean_df[TARGET_COL]

    # Return the feature matrix X, target vector y, and the cleaned DataFrame
    return X, y, clean_df


# ------------------------------------------------------------------------------
# Function: build_preprocessor
# ------------------------------------------------------------------------------
def build_preprocessor():
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline:
    - Applies StandardScaler to numerical columns: z = (x - mean) / std
    - Applies OneHotEncoder(drop='first') to categorical columns to avoid multicollinearity.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Standardize numerical features so no single large-scale feature (e.g., Weight: 2000)
            # dominates smaller-scale features (e.g., Engine Size: 2.0)
            ('num', StandardScaler(), NUMERICAL_COLS),
            # One-Hot Encode categories (drop='first' prevents dummy variable trap / collinearity)
            ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    return preprocessor


# ------------------------------------------------------------------------------
# Function: get_train_test_data
# ------------------------------------------------------------------------------
def get_train_test_data(test_size=0.2, random_state=42):
    """
    Loads, cleans, and splits the dataset into training and evaluation sets.
    Applies StandardScaler and OneHotEncoder without data leakage.

    Returns:
      - X_train, X_test: Original unscaled feature DataFrames (useful for tree inspection)
      - X_train_proc, X_test_proc: Transformed numerical NumPy matrices ready for ML modeling
      - y_train, y_test: Ground truth fuel consumption target splits (L/100 km)
      - preprocessor: The fitted ColumnTransformer pipeline
      - all_feature_names: List of all feature column names including encoded categories
    """
    # Step 1: Load raw dataset from disk
    df = load_raw_data()

    # Step 2: Separate features X and target y
    X, y, clean_df = preprocess_data(df)

    # Step 3: Split into 80% Training set (for learning) and 20% Test set (for final evaluation).
    # random_state=42 ensures perfectly reproducible splits every time.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 4: Build ColumnTransformer preprocessing pipeline
    preprocessor = build_preprocessor()

    # Step 5: Fit preprocessor ONLY on training data (learns means, variances, category categories).
    # This strictly prevents Data Leakage from the test set into the model!
    X_train_proc = preprocessor.fit_transform(X_train)

    # Step 6: Transform test data using the parameters learned from the training data
    X_test_proc = preprocessor.transform(X_test)

    # Step 7: Extract the exact feature names after One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = NUMERICAL_COLS + encoded_cat_cols

    # Return all split matrices, processed data, fitted preprocessor, and feature names
    return X_train, X_test, X_train_proc, X_test_proc, y_train, y_test, preprocessor, all_feature_names


# ------------------------------------------------------------------------------
# Standalone Execution Verification Block
# ------------------------------------------------------------------------------
# Run this test block only when executing data_loader.py directly in the console
if __name__ == "__main__":
    # Load raw dataset
    df = load_raw_data()
    print(f"✅ Raw Data Loaded Successfully. Shape: {df.shape}")

    # Preprocess and split
    X_train, X_test, X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names = get_train_test_data()

    print("\n📦 Data Split and Preprocessing Summary:")
    print(f"   • Training Feature Matrix (X_train_proc): {X_train_proc.shape[0]} samples, {X_train_proc.shape[1]} features")
    print(f"   • Testing Feature Matrix  (X_test_proc) : {X_test_proc.shape[0]} samples, {X_test_proc.shape[1]} features")
    print(f"   • Training Target Vector  (y_train)     : {y_train.shape[0]} samples")
    print(f"   • Testing Target Vector   (y_test)      : {y_test.shape[0]} samples")

    print("\n🏷️ Encoded Feature Names (Total:", len(feature_names), "):")
    for idx, fname in enumerate(feature_names, 1):
        print(f"   {idx}. {fname}")

    print("\n🔍 First Transformed Training Sample (Normalized Vector):")
    print(np.round(X_train_proc[0], 3))
