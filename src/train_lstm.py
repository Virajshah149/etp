"""
train_lstm.py — LSTM Model for Passenger Demand Forecasting
===========================================================
This module trains a Long Short-Term Memory (LSTM) neural network
to forecast hourly passenger demand using time-series patterns.

Why LSTM?
  LSTM networks are well-suited for sequential data because they can
  remember long-term dependencies. Here, the model learns from 24
  consecutive hours of multi-variate demand data to predict the next hour.

Model Architecture:
  Input Layer → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)

Training Strategy:
  - EarlyStopping on validation loss (patience=8)
  - ModelCheckpoint to save best model
  - 80/20 chronological train/validation split

OUTPUT:
  models/lstm_model.keras
  models/scaler_X.pkl     (MinMaxScaler for input features)
  models/scaler_y.pkl     (MinMaxScaler for target)
  outputs/graphs/lstm_training_history.png
  outputs/graphs/lstm_predictions_vs_actual.png
  outputs/reports/model_metrics.json  (merged with RF metrics later)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    FEATURED_DATA_PATH, LSTM_MODEL_PATH,
    SCALER_X_PATH, SCALER_Y_PATH,
    GRAPHS_DIR, REPORTS_DIR, METRICS_PATH,
    TARGET_COL, LSTM_WINDOW_SIZE,
    LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_UNITS_1, LSTM_UNITS_2,
    TRAIN_TEST_SPLIT, ensure_dirs, setup_logger,
    save_metrics, load_metrics
)

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# FEATURES USED FOR LSTM
# These are the columns fed into each time step
# ─────────────────────────────────────────────
LSTM_INPUT_FEATURES = [
    "passenger_count",       # Target (normalized) — auto-regressive
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "event_weight",
    "temperature_c",
    "rainfall_mm",
    "weather_severity_score",
    "is_peak_hour",
]


def prepare_lstm_data(df: pd.DataFrame):
    """
    Prepare data for LSTM training.

    Strategy:
      1. Aggregate data per (datetime, route_id) — use mean for multi-stop routes
      2. Sort chronologically per route
      3. Normalize all features using MinMaxScaler
      4. Create sliding window sequences (window = LSTM_WINDOW_SIZE)
      5. Chronological 80/20 split (no shuffling — preserves time order)

    Args:
        df: Feature-engineered DataFrame

    Returns:
        X_train, X_val, y_train, y_val: numpy arrays
        scaler_X, scaler_y: fitted scalers
    """
    logger.info("Preparing LSTM data sequences...")

    # Use Route R01 (highest demand route) as the primary training series.
    # We train on aggregate city-level demand per hour for simplicity.
    # This makes the model generalize across routes.
    available_features = [f for f in LSTM_INPUT_FEATURES if f in df.columns]

    # Aggregate to hourly city-level demand (mean across all routes/stops)
    hourly_df = df.groupby("datetime")[available_features].mean().reset_index()
    hourly_df = hourly_df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Hourly aggregated series: {len(hourly_df)} time steps")

    # Separate feature and target scalers
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    feature_data = hourly_df[available_features].values
    target_data  = hourly_df[[TARGET_COL]].values

    # Fit scalers on ALL data first (we'll use the time-split after)
    scaler_X.fit(feature_data)
    scaler_y.fit(target_data)

    scaled_features = scaler_X.transform(feature_data)
    scaled_target   = scaler_y.transform(target_data)

    # Save scalers for later use in prediction
    ensure_dirs()
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    logger.info(f"Scalers saved → {SCALER_X_PATH.parent}")

    # ── Create sliding window sequences ──
    X, y = [], []
    window = LSTM_WINDOW_SIZE

    for i in range(window, len(scaled_features)):
        X.append(scaled_features[i - window : i])   # window time steps of features
        y.append(scaled_target[i, 0])                # next hour's demand

    X = np.array(X)  # shape: (samples, window, n_features)
    y = np.array(y)  # shape: (samples,)

    # ── Chronological train/validation split ──
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
    logger.info(f"y_train: {y_train.shape} | y_val: {y_val.shape}")

    return X_train, X_val, y_train, y_val, scaler_X, scaler_y


def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Build the LSTM model architecture.

    Architecture:
      Input → LSTM(64, return_sequences=True) → Dropout(0.2)
            → LSTM(32) → Dropout(0.2) → Dense(1)

    The two-layer LSTM allows the first layer to learn short-term
    patterns while the second layer learns higher-level temporal patterns.

    Args:
        input_shape: (window_size, n_features)

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(LSTM_UNITS_1, return_sequences=True,
             name="lstm_layer_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(LSTM_UNITS_2, return_sequences=False,
             name="lstm_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(1, name="output_layer")
    ], name="EcoTransit_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    logger.info("LSTM model built:")
    model.summary(print_fn=logger.info)
    return model


def train_lstm(model: tf.keras.Model,
               X_train, y_train, X_val, y_val) -> tf.keras.callbacks.History:
    """
    Train the LSTM model with callbacks:
      - EarlyStopping: stop when val_loss stops improving (patience=8)
      - ModelCheckpoint: save the best model weights
      - ReduceLROnPlateau: halve LR when val_loss plateaus (patience=4)

    Returns:
        Training history object
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(LSTM_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    logger.info(f"Training LSTM | Epochs={LSTM_EPOCHS} | Batch={LSTM_BATCH_SIZE}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    logger.info(f"Training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
    return history


def evaluate_lstm(model, X_val, y_val, scaler_y) -> dict:
    """
    Evaluate LSTM model on validation set.
    Inverse-transforms predictions back to original passenger count scale.

    Returns:
        dict with MAE, RMSE, R² metrics
    """
    y_pred_scaled = model.predict(X_val, verbose=0).flatten()

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    metrics = {
        "lstm": {
            "MAE":  round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2":   round(r2, 4)
        }
    }

    logger.info(f"LSTM Evaluation → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    return metrics, y_true, y_pred


def plot_training_history(history: tf.keras.callbacks.History):
    """Plot and save the training vs validation loss curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history.history["loss"],     label="Train Loss", color="#2196F3")
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#FF5722")
    axes[0].set_title("LSTM Training & Validation Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # MAE plot
    axes[1].plot(history.history["mae"],     label="Train MAE", color="#4CAF50")
    axes[1].plot(history.history["val_mae"], label="Val MAE",   color="#FF9800")
    axes[1].set_title("LSTM Training & Validation MAE", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Eco-Transit Pulse — LSTM Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = GRAPHS_DIR / "lstm_training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history plot saved → {save_path}")


def plot_predictions(y_true, y_pred, n_samples=200):
    """Plot actual vs predicted passenger counts for visual validation."""
    # Use first n_samples for clarity
    indices = range(min(n_samples, len(y_true)))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(indices, y_true[:n_samples], label="Actual",    color="#2196F3", linewidth=1.5)
    ax.plot(indices, y_pred[:n_samples], label="Predicted", color="#FF5722",
            linewidth=1.5, linestyle="--", alpha=0.85)
    ax.fill_between(indices, y_true[:n_samples], y_pred[:n_samples],
                    alpha=0.15, color="#FF9800")
    ax.set_title("LSTM: Actual vs Predicted Passenger Demand", fontsize=13)
    ax.set_xlabel("Time Steps (hours)")
    ax.set_ylabel("Passenger Count")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = GRAPHS_DIR / "lstm_predictions_vs_actual.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Prediction comparison plot saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Eco-Transit Pulse — LSTM Model Training")
    logger.info("=" * 55)

    # Load feature-engineered dataset
    logger.info(f"Loading featured dataset from: {FEATURED_DATA_PATH}")
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["datetime"])

    # Prepare sequences
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_lstm_data(df)

    # Build model
    n_features   = X_train.shape[2]
    input_shape  = (LSTM_WINDOW_SIZE, n_features)
    model        = build_lstm_model(input_shape)

    # Train
    history = train_lstm(model, X_train, y_train, X_val, y_val)

    # Evaluate
    metrics, y_true, y_pred = evaluate_lstm(model, X_val, y_val, scaler_y)

    # Save metrics (will be merged with RF metrics later)
    existing_metrics = load_metrics()
    existing_metrics.update(metrics)
    save_metrics(existing_metrics)
    logger.info(f"Metrics saved → {METRICS_PATH}")

    # Save plots
    ensure_dirs()
    plot_training_history(history)
    plot_predictions(y_true, y_pred)

    logger.info("\n=== LSTM Training Summary ===")
    logger.info(f"MAE  : {metrics['lstm']['MAE']}")
    logger.info(f"RMSE : {metrics['lstm']['RMSE']}")
    logger.info(f"R²   : {metrics['lstm']['R2']}")
    logger.info(f"Model saved → {LSTM_MODEL_PATH}")
