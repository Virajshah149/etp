"""
feature_engineering.py — Feature Engineering Pipeline
======================================================
This module transforms the preprocessed dataset into ML-ready features.

Features engineered:
  Temporal flags:
    - is_peak_hour     : 1 if hour in {7,8,9,17,18,19}
    - weather_severity_score : 0-3 scale from weather_encoded

  Lag features (time-series memory):
    - lag_1h, lag_2h, lag_3h : passenger_count 1/2/3 hours ago (per route+stop)

  Rolling window features (smoothed demand signal):
    - rolling_avg_3h  : 3-hour rolling average
    - rolling_avg_6h  : 6-hour rolling average
    - rolling_avg_12h : 12-hour rolling average

OUTPUT:
  data/processed/transit_demand_featured.csv
  outputs/graphs/feature_correlation_heatmap.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    PROCESSED_DATA_PATH, FEATURED_DATA_PATH,
    GRAPHS_DIR, TARGET_COL, ensure_dirs, setup_logger
)

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# PEAK HOUR DEFINITION
# Based on BMTC reported peak ridership hours (morning + evening)
# ─────────────────────────────────────────────
MORNING_PEAK_HOURS = {7, 8, 9}
EVENING_PEAK_HOURS = {17, 18, 19}
PEAK_HOURS = MORNING_PEAK_HOURS | EVENING_PEAK_HOURS

# Weather severity score mapping (simplified from weather_encoded)
WEATHER_SEVERITY = {
    0: 0.0,   # Sunny
    1: 0.0,   # Partly Cloudy
    2: 0.5,   # Overcast
    3: 1.5,   # Drizzle
    4: 2.5,   # Heavy Rain
    5: 3.0    # Thunderstorm
}


def add_peak_hour_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flag: 1 if current hour is a peak travel hour, else 0.
    Peak hours: 7-9am (morning) and 5-7pm (evening).
    """
    df["is_peak_hour"] = df["hour"].apply(lambda h: 1 if h in PEAK_HOURS else 0)
    peak_count = df["is_peak_hour"].sum()
    logger.info(f"Peak hour flag added — {peak_count:,} peak-hour records.")
    return df


def add_weather_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ordinal weather_encoded (0-5) to a 3-point severity score.
    0 = good weather, 3 = severe weather (thunderstorm).
    This makes the impact on demand more interpretable in the model.
    """
    df["weather_severity_score"] = df["weather_encoded"].map(WEATHER_SEVERITY)
    df["weather_severity_score"] = df["weather_severity_score"].fillna(0.0)
    logger.info("Weather severity score (0-3) added.")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for each (route_id, stop_id) group independently.
    Lag features capture recent demand history, crucial for LSTM and RF.

    lag_1h: passenger_count 1 hour ago
    lag_2h: passenger_count 2 hours ago
    lag_3h: passenger_count 3 hours ago

    NaN values at the start of each group are forward-filled with the
    group's mean demand to avoid data leakage from other groups.
    """
    # Sort to ensure chronological order within each group
    df = df.sort_values(["route_id", "stop_id", "datetime"]).reset_index(drop=True)

    # Compute lag features per (route_id, stop_id) group
    for lag in [1, 2, 3]:
        col_name = f"lag_{lag}h"
        df[col_name] = df.groupby(["route_id", "stop_id"])[TARGET_COL].shift(lag)

    # Fill NaN lag values (beginning of each series) with group mean
    for lag in [1, 2, 3]:
        col_name = f"lag_{lag}h"
        group_means = df.groupby(["route_id", "stop_id"])[TARGET_COL].transform("mean")
        df[col_name] = df[col_name].fillna(group_means)

    logger.info("Lag features added: lag_1h, lag_2h, lag_3h")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features per (route_id, stop_id) group.

    rolling_avg_3h  : Mean demand over past 3 hours
    rolling_avg_6h  : Mean demand over past 6 hours
    rolling_avg_12h : Mean demand over past 12 hours

    Uses min_periods=1 so partial windows at the start are still computed.
    """
    for window in [3, 6, 12]:
        col_name = f"rolling_avg_{window}h"
        df[col_name] = df.groupby(["route_id", "stop_id"])[TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        # Fill any remaining NaNs with group mean
        group_means = df.groupby(["route_id", "stop_id"])[TARGET_COL].transform("mean")
        df[col_name] = df[col_name].fillna(group_means)

    logger.info("Rolling average features added: 3h, 6h, 12h windows")
    return df


def save_featured_data(df: pd.DataFrame, path=FEATURED_DATA_PATH):
    """Save the feature-engineered dataset."""
    ensure_dirs()
    df.to_csv(path, index=False)
    logger.info(f"Featured dataset saved → {path} ({len(df):,} rows)")


def plot_feature_correlation(df: pd.DataFrame):
    """
    Generate and save a correlation heatmap of numeric features.
    Helps visualize which features are most correlated with passenger_count.
    """
    numeric_cols = [
        TARGET_COL, "hour", "day_of_week", "month", "is_weekend",
        "is_holiday", "event_weight", "temperature_c", "rainfall_mm",
        "weather_severity_score", "is_peak_hour",
        "lag_1h", "lag_2h", "lag_3h",
        "rolling_avg_3h", "rolling_avg_6h", "rolling_avg_12h"
    ]
    # Keep only columns that exist in df
    existing = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[existing].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, square=True, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix — Eco-Transit Pulse", fontsize=14, pad=15)
    plt.tight_layout()

    save_path = GRAPHS_DIR / "feature_correlation_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation heatmap saved → {save_path}")


def run_feature_engineering(processed_path=PROCESSED_DATA_PATH,
                             featured_path=FEATURED_DATA_PATH) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.

    Returns:
        Feature-enriched DataFrame ready for ML model training.
    """
    logger.info("Starting feature engineering pipeline...")

    df = pd.read_csv(processed_path, parse_dates=["datetime", "date"])
    logger.info(f"Loaded processed data: {df.shape}")

    df = add_peak_hour_flag(df)
    df = add_weather_severity_score(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Final NaN check
    nan_count = df.isnull().sum().sum()
    logger.info(f"Missing values after feature engineering: {nan_count}")

    save_featured_data(df, featured_path)
    plot_feature_correlation(df)

    logger.info("Feature engineering complete!")
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Eco-Transit Pulse — Feature Engineering")
    logger.info("=" * 55)

    df = run_feature_engineering()

    # Summary of newly added features
    new_features = [
        "is_peak_hour", "weather_severity_score",
        "lag_1h", "lag_2h", "lag_3h",
        "rolling_avg_3h", "rolling_avg_6h", "rolling_avg_12h"
    ]
    print("\n--- New Feature Summary ---")
    print(df[new_features].describe().round(2))
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Total feature columns: {len(df.columns)}")
