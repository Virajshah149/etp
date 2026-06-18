"""
utils.py — Shared utilities, constants, and helper functions
=============================================================
This module is the single source of truth for:
  - All file/directory paths used across the project
  - Shared column name definitions
  - Logging configuration
  - Common helper functions

Every other module imports from here to ensure consistency.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# ROOT PATH (auto-detected relative to this file)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# DIRECTORY PATHS
# ─────────────────────────────────────────────
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR         = PROJECT_ROOT / "models"
OUTPUTS_DIR        = PROJECT_ROOT / "outputs"
GRAPHS_DIR         = OUTPUTS_DIR / "graphs"
HEATMAPS_DIR       = OUTPUTS_DIR / "heatmaps"
REPORTS_DIR        = OUTPUTS_DIR / "reports"

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
RAW_DATA_PATH       = DATA_RAW_DIR / "transit_demand_raw.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "transit_demand_processed.csv"
FEATURED_DATA_PATH  = DATA_PROCESSED_DIR / "transit_demand_featured.csv"

LSTM_MODEL_PATH     = MODELS_DIR / "lstm_model.keras"
RF_MODEL_PATH       = MODELS_DIR / "rf_model.pkl"
SCALER_X_PATH       = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_PATH       = MODELS_DIR / "scaler_y.pkl"
KMEANS_MODEL_PATH   = MODELS_DIR / "kmeans_model.pkl"

METRICS_PATH        = REPORTS_DIR / "model_metrics.json"
HOTSPOT_MAP_PATH    = HEATMAPS_DIR / "ghost_hotspots_map.html"
CLUSTER_INFO_PATH   = REPORTS_DIR / "cluster_info.json"

# ─────────────────────────────────────────────
# DATASET SCHEMA — SHARED COLUMN NAMES
# ─────────────────────────────────────────────

# Raw data columns (output of data_collection.py)
RAW_COLS = [
    "datetime", "date", "hour", "route_id", "route_name",
    "stop_id", "stop_name", "latitude", "longitude",
    "passenger_count", "temperature_c", "weather_condition",
    "rainfall_mm", "is_holiday", "holiday_name",
    "event_name", "event_weight"
]

# Columns added by preprocessing.py
PREPROCESS_COLS = [
    "day_of_week", "month", "is_weekend",
    "weather_encoded", "season"
]

# Columns added by feature_engineering.py
FEATURE_COLS = [
    "lag_1h", "lag_2h", "lag_3h",
    "rolling_avg_3h", "rolling_avg_6h", "rolling_avg_12h",
    "is_peak_hour", "weather_severity_score"
]

# Random Forest input features
RF_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "event_weight", "temperature_c", "rainfall_mm",
    "weather_severity_score", "is_peak_hour",
    "lag_1h", "lag_2h", "lag_3h",
    "rolling_avg_3h", "rolling_avg_6h"
]

# LSTM input features (one feature per timestep, multivariate)
LSTM_FEATURES = [
    "passenger_count_norm", "hour_norm", "day_of_week_norm",
    "is_weekend", "is_holiday", "event_weight_norm",
    "temperature_norm", "rainfall_norm", "weather_severity_score"
]

# Target column
TARGET_COL = "passenger_count"

# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────
LSTM_WINDOW_SIZE = 24       # Use 24 hours of history to predict next hour
LSTM_EPOCHS      = 50       # Max epochs (early stopping kicks in earlier)
LSTM_BATCH_SIZE  = 64
LSTM_UNITS_1     = 64       # First LSTM layer units
LSTM_UNITS_2     = 32       # Second LSTM layer units

RF_N_ESTIMATORS  = 100
RF_MAX_DEPTH     = 15
RF_RANDOM_STATE  = 42

KMEANS_N_CLUSTERS = 5       # Determined via elbow method
KMEANS_RANDOM_STATE = 42

# Hybrid prediction weights
HYBRID_LSTM_WEIGHT = 0.60   # LSTM captures temporal patterns better
HYBRID_RF_WEIGHT   = 0.40   # RF captures external factor effects

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.80

# ─────────────────────────────────────────────
# DEMAND CLASSIFICATION THRESHOLDS
# ─────────────────────────────────────────────
DEMAND_THRESHOLDS = {
    "Low":       (0,   50),
    "Medium":    (50,  120),
    "High":      (120, 200),
    "Very High": (200, float("inf"))
}

# ─────────────────────────────────────────────
# CITY SIMULATION PARAMETERS (Bengaluru-inspired)
# ─────────────────────────────────────────────
CITY_NAME      = "Bengaluru"
CITY_CENTER    = (12.9716, 77.5946)   # lat, lon
SIMULATION_NOTE = (
    "Dataset is a hybrid of real GTFS structural schema and "
    "realistically simulated demand, weather, and event data. "
    "Route topology is inspired by Bengaluru's BMTC bus network. "
    "Passenger counts, weather, and events are synthetically generated "
    "using statistically realistic distributions and validated patterns."
)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Create a named logger with a consistent format.
    Each module calls this at the top with its own __name__.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────
def ensure_dirs():
    """Create all required project directories if they don't exist."""
    dirs = [
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
        GRAPHS_DIR, HEATMAPS_DIR, REPORTS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────
def save_metrics(metrics: dict, path=METRICS_PATH):
    """Save model evaluation metrics to a JSON file."""
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def load_metrics(path=METRICS_PATH) -> dict:
    """Load saved model metrics from JSON."""
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_cluster_info(info: list, path=CLUSTER_INFO_PATH):
    """Save K-Means cluster info to JSON."""
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def load_cluster_info(path=CLUSTER_INFO_PATH) -> list:
    """Load saved cluster info from JSON."""
    if not Path(path).exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# DEMAND CLASSIFICATION
# ─────────────────────────────────────────────
def classify_demand(count: float) -> str:
    """
    Classify a passenger count into a demand category.

    Args:
        count: Predicted or actual passenger count

    Returns:
        String label: 'Low', 'Medium', 'High', or 'Very High'
    """
    for label, (lo, hi) in DEMAND_THRESHOLDS.items():
        if lo <= count < hi:
            return label
    return "Very High"


# ─────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────
def format_metric(value: float, decimals: int = 2) -> str:
    """Format a float metric for display (e.g., '94.32')."""
    return f"{value:.{decimals}f}"


def get_season(month: int) -> str:
    """Map a month number to an Indian season name."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


if __name__ == "__main__":
    # Quick sanity check — print key paths
    print("=== Eco-Transit Pulse: Path Configuration ===")
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"Raw Data       : {RAW_DATA_PATH}")
    print(f"Processed Data : {PROCESSED_DATA_PATH}")
    print(f"Featured Data  : {FEATURED_DATA_PATH}")
    print(f"LSTM Model     : {LSTM_MODEL_PATH}")
    print(f"RF Model       : {RF_MODEL_PATH}")
    print(f"KMeans Model   : {KMEANS_MODEL_PATH}")
    ensure_dirs()
    print("\nAll directories verified/created successfully.")
