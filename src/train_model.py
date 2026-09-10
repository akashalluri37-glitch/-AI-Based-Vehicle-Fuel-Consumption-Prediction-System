# ==============================================================================
# Script: train_model.py
# Purpose: Trains, benchmarks, and evaluates 3 Regression Algorithms
#          (Linear Regression, Decision Tree Regressor, Random Forest Regressor)
#          on automotive specs to predict fuel consumption (L/100 km & MPG),
#          and serializes models and preprocessors for the Streamlit dashboard.
# ==============================================================================

# Import operating system library for path operations and directory creation
import os

# Import system-specific module to configure terminal stream settings
import sys

# Ensure UTF-8 output encoding for Windows PowerShell consoles so emojis and symbols display correctly
if sys.platform == "win32":
    try:
        # Reconfigure standard output stream to use UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
        # Reconfigure standard error stream to use UTF-8
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # Ignore errors if console reconfigure is not supported in the current environment
        pass

# Import json library to serialize model evaluation metrics into human-readable JSON
import json

# Import joblib library to serialize (save) and deserialize (load) trained Python ML models
import joblib

# Import Pandas for manipulating tabular data and creating feature importance DataFrames
import pandas as pd

# Import NumPy for array manipulation, numerical operations, and square root calculations
import numpy as np

# Import Linear Regression algorithm from scikit-learn (baseline linear parametric model)
from sklearn.linear_model import LinearRegression

# Import Decision Tree Regressor from scikit-learn (rule-based non-linear tree)
from sklearn.tree import DecisionTreeRegressor

# Import Random Forest Regressor from scikit-learn (ensemble of multiple bagged decision trees)
from sklearn.ensemble import RandomForestRegressor

# Import regression evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# HOW THIS FILE CONNECTS TO data_loader.py:
# We import get_train_test_data from data_loader.py to receive the preprocessed,
# scaled, and train/test split matrices directly into RAM.
# ------------------------------------------------------------------------------
try:
    # Attempt standard package import when running from the project root directory
    from src.data_loader import get_train_test_data
except ImportError:
    # Fallback import if running directly from within the 'src' directory
    from data_loader import get_train_test_data

# ------------------------------------------------------------------------------
# Path Definitions for Saving Models & Artifacts
# ------------------------------------------------------------------------------

# Define the absolute path to the 'models/' folder (one directory level above 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved_models")

# Define the path for the bundled all-in-one model file (matching ML_1_project format)
BUNDLE_MODEL_FILE = os.path.join(MODELS_DIR, "model.joblib")

# Define path for the evaluation metrics summary JSON file
METRICS_FILE = os.path.join(SAVED_MODELS_DIR, "evaluation_metrics.json")


# ------------------------------------------------------------------------------
# Main Function: train_and_evaluate_models
# ------------------------------------------------------------------------------
def train_and_evaluate_models():
    """
    Orchestrates the entire model training pipeline:
    1. Loads clean train/test data from data_loader.py.
    2. Initializes 3 distinct regression models:
       - Linear Regression (Baseline)
       - Decision Tree Regressor (Non-linear)
       - Random Forest Regressor (Ensemble Bagging)
    3. Trains and benchmarks each model using MAE, RMSE, and R² Score.
    4. Identifies the best-performing model (Random Forest Regressor).
    5. Saves all individual models and ColumnTransformer preprocessor to 'models/saved_models/'.
    6. Saves an all-in-one bundled artifact to 'models/model.joblib'.
    7. Displays automotive feature importance rankings.
    """
    # Create the destination models directories if they do not exist yet
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # Print a visual banner to separate output
    print("🔄 Loading, preprocessing, and splitting vehicle dataset...")

    # Call get_train_test_data() from data_loader.py to retrieve splits and fitted preprocessor
    X_train, X_test, X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names = get_train_test_data()

    print(f"✅ Data split ready: {X_train_proc.shape[0]} training samples, {X_test_proc.shape[0]} test samples")
    print(f"📊 Total encoded features: {len(feature_names)}")

    # Define the dictionary of Machine Learning regression algorithms to train and compare
    models = {
        # Model 1: Linear Regression - baseline linear model predicting continuous consumption
        "Linear Regression": LinearRegression(),

        # Model 2: Decision Tree Regressor - tree with maximum depth of 6 to prevent overfitting
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),

        # Model 3: Random Forest Regressor - 100 decision trees averaged together for high stability
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }

    # Dictionary to store performance metrics for each model
    reg_metrics = {}

    # Dictionary to store the fitted Python model objects
    trained_model_objects = {}

    # Print visual banner for training
    print("\n🚀 Training Regression Models on Vehicle Specifications...\n" + "=" * 60)

    # Loop through each model name and algorithm instance in the dictionary
    for name, model in models.items():
        # Train the model on the preprocessed training features and continuous target
        model.fit(X_train_proc, y_train)

        # Generate predictions on the unseen test feature matrix
        preds = model.predict(X_test_proc)

        # ----------------------------------------------------------------------
        # Regression Evaluation Metrics:
        # 1. MAE (Mean Absolute Error): Average absolute difference between actual & predicted L/100 km.
        # 2. RMSE (Root Mean Squared Error): Square root of mean squared errors; penalizes large errors.
        # 3. R² (Coefficient of Determination): % of variance in fuel consumption explained by the model.
        # ----------------------------------------------------------------------
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Store rounded metrics in the evaluation dictionary
        reg_metrics[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        }

        # Store the fitted model object
        trained_model_objects[name] = model

        # Save individual model to disk in 'models/saved_models/'
        filename = f"regressor_{name.lower().replace(' ', '_')}.joblib"
        model_save_path = os.path.join(SAVED_MODELS_DIR, filename)
        joblib.dump(model, model_save_path)

        # Print model evaluation results to the console
        print(f"📌 Model: {name}")
        print(f"   • MAE (Mean Absolute Error)     : {mae:.4f} L/100 km")
        print(f"   • RMSE (Root Mean Squared Error): {rmse:.4f} L/100 km")
        print(f"   • R² Score (Variance Explained) : {r2 * 100:.2f}%")
        print(f"   💾 Saved artifact to            : {model_save_path}")
        print("-" * 60)

    # Save fitted ColumnTransformer preprocessor to disk
    preprocessor_save_path = os.path.join(SAVED_MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_save_path)
    print(f"💾 Saved fitted Preprocessor to: {preprocessor_save_path}")

    # --------------------------------------------------------------------------
    # Best Model Selection
    # --------------------------------------------------------------------------
    # Identify the best model with highest R² score
    best_model_name = max(reg_metrics, key=lambda m: reg_metrics[m]["R2"])
    best_model = trained_model_objects[best_model_name]
    best_r2 = reg_metrics[best_model_name]["R2"]
    best_mae = reg_metrics[best_model_name]["MAE"]

    print(f"\n🏆 Best Performing Model: {best_model_name} (R² = {best_r2 * 100:.2f}%, MAE = {best_mae:.4f} L/100 km)")

    # --------------------------------------------------------------------------
    # Automotive Feature Importance Analysis (Random Forest)
    # --------------------------------------------------------------------------
    rf_model = trained_model_objects["Random Forest"]
    importances = rf_model.feature_importances_
    fi_dict = dict(zip(feature_names, [round(float(v), 4) for v in importances]))

    # Create a sorted Pandas DataFrame for display
    fi_df = pd.DataFrame({
        'Automotive Feature': feature_names,
        'Importance Score': importances
    }).sort_values(by='Importance Score', ascending=False)

    print("\n⚙️ Automotive Feature Importance Ranking (Random Forest):")
    print(fi_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # Export Evaluation Metrics JSON
    # --------------------------------------------------------------------------
    summary = {
        "Regression Metrics": reg_metrics,
        "Feature Importance": fi_dict,
        "Feature Names": feature_names,
        "Best Model": best_model_name
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"💾 Saved evaluation metrics JSON to: {METRICS_FILE}")

    # --------------------------------------------------------------------------
    # Save All-In-One Bundled Model File (models/model.joblib)
    # --------------------------------------------------------------------------
    saved_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "results": reg_metrics,
        "feature_importance": fi_dict
    }
    joblib.dump(saved_payload, BUNDLE_MODEL_FILE)
    print(f"✅ Saved all-in-one bundle to: {BUNDLE_MODEL_FILE}")

    # Print summary performance comparison table
    print("\n📊 Regression Model Performance Comparison Table:")
    summary_df = pd.DataFrame(reg_metrics).T
    print(summary_df)

    return saved_payload


# ------------------------------------------------------------------------------
# Script Entry Point: Runs only when executing this script directly
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_evaluate_models()
