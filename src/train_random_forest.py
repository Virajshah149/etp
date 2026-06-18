"""
train_random_forest.py — Random Forest Regressor Training
==========================================================
This module trains a Random Forest model to analyze how external
factors (weather, holidays, events, time-of-day) influence passenger demand.

Why Random Forest?
  While LSTM excels at learning sequential time patterns, Random Forest
  is better at learning non-linear relationships between categorical/
  numerical features and the target variable. It complements the LSTM
  in the hybrid prediction by capturing external context.

Features used:
  - Time features: hour, day_of_week, month, is_weekend, is_peak_hour
  - External factors: is_holiday, event_weight
  - Weather: temperature_c, rainfall_mm, weather_severity_score
  - Historical demand: lag_1h, lag_2h, lag_3h, rolling_avg_3h, rolling_avg_6h

OUTPUT:
  models/rf_model.pkl
  outputs/graphs/rf_feature_importance.png
  outputs/graphs/rf_predictions_vs_actual.png
  outputs/reports/model_metrics.json (merged with LSTM metrics)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    FEATURED_DATA_PATH, RF_MODEL_PATH, METRICS_PATH,
    GRAPHS_DIR, REPORTS_DIR,
    TARGET_COL, RF_FEATURES,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    TRAIN_TEST_SPLIT, ensure_dirs, setup_logger,
    save_metrics, load_metrics
)

logger = setup_logger(__name__)


def load_featured_data(path=FEATURED_DATA_PATH) -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    logger.info(f"Loaded featured dataset: {df.shape}")
    return df


def prepare_rf_data(df: pd.DataFrame):
    """
    Prepare data for Random Forest training.

    Steps:
      1. Select only the RF input features and target
      2. Drop any remaining NaN rows
      3. 80/20 random split (RF doesn't require chronological ordering)

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Keep only features defined in utils.RF_FEATURES that are present in df
    feature_names = [f for f in RF_FEATURES if f in df.columns]
    missing_feats = set(RF_FEATURES) - set(feature_names)
    if missing_feats:
        logger.warning(f"Missing RF features (will skip): {missing_feats}")

    # Prepare feature matrix and target vector
    X = df[feature_names].copy()
    y = df[TARGET_COL].copy()

    # Drop rows with any NaN in features or target
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"RF data prepared: {X.shape[0]:,} samples × {X.shape[1]} features")

    # Random train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=(1 - TRAIN_TEST_SPLIT),
        random_state=RF_RANDOM_STATE,
        shuffle=True
    )

    logger.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, feature_names


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Hyperparameters (chosen for balance between accuracy and speed):
      - n_estimators=100 : 100 decision trees
      - max_depth=15     : Enough depth to capture complex patterns
      - min_samples_leaf=5: Prevents overfitting
      - n_jobs=-1        : Use all CPU cores for faster training

    Returns:
        Fitted RandomForestRegressor
    """
    logger.info(
        f"Training Random Forest | n_estimators={RF_N_ESTIMATORS} | "
        f"max_depth={RF_MAX_DEPTH}"
    )

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",      # Use sqrt(n_features) per tree — standard
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,                # Parallelise across all CPU cores
        oob_score=True            # Out-of-bag error estimate
    )

    rf.fit(X_train, y_train)

    logger.info(f"OOB Score (train set estimate): {rf.oob_score_:.4f}")
    return rf


def evaluate_rf(rf: RandomForestRegressor, X_test, y_test) -> dict:
    """
    Evaluate Random Forest on the test set.

    Returns:
        metrics dict and (y_true, y_pred) arrays
    """
    y_pred = rf.predict(X_test)
    y_true = y_test.values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    metrics = {
        "random_forest": {
            "MAE":      round(mae, 4),
            "RMSE":     round(rmse, 4),
            "R2":       round(r2, 4),
            "OOB_Score":round(rf.oob_score_, 4)
        }
    }

    logger.info(f"RF Evaluation → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    return metrics, y_true, y_pred


def save_rf_model(rf: RandomForestRegressor, path=RF_MODEL_PATH):
    """Save the trained Random Forest model using joblib."""
    ensure_dirs()
    joblib.dump(rf, path)
    logger.info(f"Random Forest model saved → {path}")


def plot_feature_importance(rf: RandomForestRegressor, feature_names: list):
    """
    Plot and save the top-N feature importances from the trained RF model.
    Helps explain which factors most influence passenger demand.
    """
    importances = rf.feature_importances_
    feature_df  = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    # Plot top 15
    top_n = min(15, len(feature_df))
    top_features = feature_df.head(top_n)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1],
        color=colors
    )
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title("Random Forest — Feature Importance\n(Eco-Transit Pulse)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, top_features["Importance"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    save_path = GRAPHS_DIR / "rf_feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved → {save_path}")
    return feature_df


def plot_rf_predictions(y_true, y_pred, n_samples=300):
    """Scatter plot of actual vs predicted values for Random Forest."""
    # Sample for scatter readability
    idx = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    sample_true = y_true[idx]
    sample_pred = y_pred[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(sample_true, sample_pred, alpha=0.4,
                    s=15, color="#2196F3", edgecolors="none")
    min_val = min(sample_true.min(), sample_pred.min())
    max_val = max(sample_true.max(), sample_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val],
                 "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Passenger Count")
    axes[0].set_ylabel("Predicted Passenger Count")
    axes[0].set_title("RF: Actual vs Predicted (Scatter)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residual plot
    residuals = sample_pred - sample_true
    axes[1].scatter(sample_true, residuals, alpha=0.4,
                    s=15, color="#FF5722", edgecolors="none")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Actual Passenger Count")
    axes[1].set_ylabel("Residual (Predicted − Actual)")
    axes[1].set_title("RF: Residual Plot")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Random Forest Model Evaluation — Eco-Transit Pulse",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = GRAPHS_DIR / "rf_predictions_vs_actual.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"RF prediction plot saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Eco-Transit Pulse — Random Forest Training")
    logger.info("=" * 55)

    df = load_featured_data()

    X_train, X_test, y_train, y_test, feature_names = prepare_rf_data(df)

    rf = train_random_forest(X_train, y_train)

    metrics, y_true, y_pred = evaluate_rf(rf, X_test, y_test)

    save_rf_model(rf)

    # Save/merge metrics with LSTM metrics
    existing = load_metrics()
    existing.update(metrics)
    save_metrics(existing)
    logger.info(f"Metrics saved → {METRICS_PATH}")

    ensure_dirs()
    feature_df = plot_feature_importance(rf, feature_names)
    plot_rf_predictions(y_true, y_pred)

    logger.info("\n=== Random Forest Training Summary ===")
    logger.info(f"MAE       : {metrics['random_forest']['MAE']}")
    logger.info(f"RMSE      : {metrics['random_forest']['RMSE']}")
    logger.info(f"R²        : {metrics['random_forest']['R2']}")
    logger.info(f"OOB Score : {metrics['random_forest']['OOB_Score']}")

    print("\nTop 10 Most Important Features:")
    print(feature_df.head(10).to_string(index=False))
