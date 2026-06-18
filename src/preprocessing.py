"""
preprocessing.py — Data Cleaning & Preprocessing Pipeline
==========================================================
This module takes the raw dataset produced by data_collection.py
and applies a full cleaning and encoding pipeline.

Steps performed:
  1. Load raw CSV
  2. Parse datetime columns
  3. Drop duplicates
  4. Handle missing values (imputation strategy)
  5. Extract temporal features (day_of_week, month, is_weekend, season)
  6. Encode categorical variables (weather_condition → weather_encoded)
  7. Save processed dataset to data/processed/

OUTPUT:
  data/processed/transit_demand_processed.csv
  models/scaler_X.pkl   (fitted MinMaxScaler for ML features)
  models/scaler_y.pkl   (fitted MinMaxScaler for target column)
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    SCALER_X_PATH, SCALER_Y_PATH,
    TARGET_COL, ensure_dirs, setup_logger, get_season
)

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# WEATHER ENCODING MAP
# Ordinal encoding: more severe weather = higher number
# ─────────────────────────────────────────────
WEATHER_ENCODING = {
    "Sunny":        0,
    "Partly Cloudy":1,
    "Overcast":     2,
    "Drizzle":      3,
    "Heavy Rain":   4,
    "Thunderstorm": 5
}


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file and perform initial type checks."""
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    logger.info(f"Removed {removed} duplicate rows ({before} → {after})")
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'datetime' column to proper datetime type.
    Extract date-level temporal features.
    """
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"]     = pd.to_datetime(df["date"])
    logger.info("Datetime columns parsed successfully.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using context-appropriate strategies:
      - Numerical columns → median imputation
      - Categorical columns → mode imputation
      - passenger_count → median per (route_id, hour) group
    """
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        logger.info("No missing values found — skipping imputation.")
        return df

    logger.info(f"Missing values found: {missing_before} total")

    # Passenger count: impute with group median (route + hour combination)
    if df["passenger_count"].isnull().any():
        df["passenger_count"] = df.groupby(["route_id", "hour"])[
            "passenger_count"
        ].transform(lambda x: x.fillna(x.median()))

    # Temperature: impute with monthly median
    if df["temperature_c"].isnull().any():
        df["temperature_c"] = df.groupby(df["date"].dt.month)[
            "temperature_c"
        ].transform(lambda x: x.fillna(x.median()))

    # Rainfall: fill with 0 (no rain if unknown)
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)

    # Categorical: fill with mode
    for col in ["weather_condition", "event_name", "holiday_name"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Boolean-like columns: fill with 0
    for col in ["is_holiday", "event_weight"]:
        df[col] = df[col].fillna(0)

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after imputation: {missing_after}")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the datetime column.

    Features added:
      - day_of_week (0=Mon, 6=Sun)
      - month (1-12)
      - is_weekend (1 if Sat/Sun)
      - season (Winter/Summer/Monsoon/Post-Monsoon)
    """
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].apply(get_season)
    logger.info("Temporal features extracted: day_of_week, month, is_weekend, season")
    return df


def encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode weather_condition → weather_encoded (0-5).
    Keeps original weather_condition column for display purposes.
    """
    df["weather_encoded"] = df["weather_condition"].map(WEATHER_ENCODING)
    # Fallback for any unseen weather values
    df["weather_encoded"] = df["weather_encoded"].fillna(1).astype(int)
    logger.info("Weather condition encoded (ordinal 0-5).")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data validation rules:
      - passenger_count must be >= 0
      - temperature_c must be in [10, 45]
      - rainfall_mm must be >= 0
      - event_weight must be in [0, 1]
    """
    # Clip to valid ranges
    df["passenger_count"] = df["passenger_count"].clip(lower=0)
    df["temperature_c"]   = df["temperature_c"].clip(lower=10, upper=45)
    df["rainfall_mm"]     = df["rainfall_mm"].clip(lower=0)
    df["event_weight"]    = df["event_weight"].clip(lower=0, upper=1)
    logger.info("Data validation applied — values clipped to valid ranges.")
    return df


def save_processed_data(df: pd.DataFrame, path=PROCESSED_DATA_PATH):
    """Save the processed DataFrame to disk."""
    ensure_dirs()
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved → {path} ({len(df):,} rows)")


def run_preprocessing_pipeline(raw_path=RAW_DATA_PATH,
                                out_path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline in sequence.

    Returns:
        Cleaned and enriched DataFrame
    """
    logger.info("Starting preprocessing pipeline...")

    df = load_raw_data(raw_path)
    df = drop_duplicates(df)
    df = parse_datetime(df)
    df = handle_missing_values(df)
    df = extract_temporal_features(df)
    df = encode_weather(df)
    df = validate_data(df)

    # ── Sort by datetime, route, stop for consistent ordering ──
    df = df.sort_values(["datetime", "route_id", "stop_id"]).reset_index(drop=True)

    save_processed_data(df, out_path)

    logger.info("Preprocessing pipeline complete!")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Columns    : {list(df.columns)}")

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Eco-Transit Pulse — Preprocessing Pipeline")
    logger.info("=" * 55)

    df = run_preprocessing_pipeline()

    # Display quick summary
    print("\n--- Preprocessing Summary ---")
    print(df[["passenger_count", "temperature_c", "rainfall_mm",
              "day_of_week", "month", "is_weekend", "weather_encoded"]].describe())
    print(f"\nWeather distribution:\n{df['weather_condition'].value_counts()}")
    print(f"\nHoliday records: {df['is_holiday'].sum():,}")
    print(f"Event records  : {(df['event_weight'] > 0).sum():,}")
