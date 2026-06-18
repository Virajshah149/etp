"""
prediction.py — Hybrid LSTM + Random Forest Prediction Engine
=============================================================
This module combines LSTM and Random Forest predictions into a
single hybrid demand forecast using a weighted average approach.

Hybrid Strategy:
  final_prediction = (0.60 × LSTM_prediction) + (0.40 × RF_prediction)

Why these weights?
  - LSTM is weighted higher (0.60) because passenger demand is
    fundamentally a time-series problem; temporal patterns (peak hours,
    daily cycles) account for most of the variance.
  - RF is weighted at 0.40 to incorporate context: weather, holidays,
    events, and other external factors that LSTM cannot easily capture.
  - Weights were validated on the held-out test set.

Usage (from dashboard or other modules):
  >>> from prediction import HybridPredictor
  >>> predictor = HybridPredictor()
  >>> result = predictor.predict(hour=8, day_of_week=1, month=3, ...)
  >>> print(result)  # {'predicted_demand': 134, 'category': 'High', ...}
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    LSTM_MODEL_PATH, RF_MODEL_PATH,
    SCALER_X_PATH, SCALER_Y_PATH,
    FEATURED_DATA_PATH,
    LSTM_WINDOW_SIZE, HYBRID_LSTM_WEIGHT, HYBRID_RF_WEIGHT,
    RF_FEATURES, TARGET_COL,
    setup_logger, classify_demand
)

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# LSTM INPUT FEATURES (must match train_lstm.py)
# ─────────────────────────────────────────────
LSTM_INPUT_FEATURES = [
    "passenger_count",
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

# Weather severity scores (mirrors feature_engineering.py)
WEATHER_SEVERITY_MAP = {
    "Sunny": 0.0, "Partly Cloudy": 0.0, "Overcast": 0.5,
    "Drizzle": 1.5, "Heavy Rain": 2.5, "Thunderstorm": 3.0
}

PEAK_HOURS = {7, 8, 9, 17, 18, 19}


class HybridPredictor:
    """
    Loads trained LSTM and Random Forest models and combines their
    predictions into a final passenger demand forecast.

    The predictor is initialized once and can be called multiple times,
    making it suitable for use in the Streamlit dashboard.
    """

    def __init__(self):
        self.lstm_model  = None
        self.rf_model    = None
        self.scaler_X    = None
        self.scaler_y    = None
        self.history_df  = None   # Recent hourly data for LSTM sequences
        self._loaded     = False

    def load_models(self) -> bool:
        """
        Load all saved models and scalers from disk.

        Returns:
            True if all models loaded successfully, False otherwise.
        """
        try:
            import tensorflow as tf

            # ── Load LSTM ──
            if not LSTM_MODEL_PATH.exists():
                logger.error(f"LSTM model not found at {LSTM_MODEL_PATH}")
                return False
            self.lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
            logger.info("LSTM model loaded.")

            # ── Load Random Forest ──
            if not RF_MODEL_PATH.exists():
                logger.error(f"RF model not found at {RF_MODEL_PATH}")
                return False
            self.rf_model = joblib.load(RF_MODEL_PATH)
            logger.info("Random Forest model loaded.")

            # ── Load Scalers ──
            self.scaler_X = joblib.load(SCALER_X_PATH)
            self.scaler_y = joblib.load(SCALER_Y_PATH)
            logger.info("Scalers loaded.")

            # ── Load historical data for LSTM context window ──
            if FEATURED_DATA_PATH.exists():
                self.history_df = pd.read_csv(
                    FEATURED_DATA_PATH, parse_dates=["datetime"]
                )
                logger.info(f"Historical data loaded: {len(self.history_df):,} rows")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    # Hourly demand factors matching data_collection.py (used to build realistic synthetic windows)
    _HOUR_FACTORS = {
        0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.04,
        5: 0.12, 6: 0.45, 7: 0.90, 8: 1.00, 9: 0.70,
        10: 0.50, 11: 0.45, 12: 0.55, 13: 0.50, 14: 0.45,
        15: 0.55, 16: 0.75, 17: 0.95, 18: 1.00, 19: 0.80,
        20: 0.55, 21: 0.35, 22: 0.20, 23: 0.10
    }

    def _build_lstm_sequence(self, input_features: dict) -> np.ndarray:
        """
        Build a realistic synthetic 24-hour context window for LSTM inference.

        Why synthetic rather than pulling raw history?
          Historical data ends in June 2024.  If a user predicts for a
          January Monday 8am, feeding June tail-data creates a temporal
          mismatch that confuses the LSTM.  Instead we reconstruct a
          plausible 24-hour demand curve for the queried day type using
          the same hourly-factor pattern baked into the training data.

        For each of the 24 time steps (hour-23 … hour-0 … current hour):
          - passenger_count is estimated as:
              avg_demand × factor(past_hour) / factor(current_hour)
          - all other contextual features remain constant (same day/weather)

        Returns:
            np.ndarray of shape (1, LSTM_WINDOW_SIZE, n_features)
        """
        current_hour   = int(input_features.get("hour", 12))
        avg_demand     = float(input_features.get("passenger_count", 50.0))
        
        # The hour factors sum to 11.03 over 24 hours (average = 0.45958).
        # We scale the overall average demand by the past hour's factor relative to this average
        # to generate a realistic absolute demand curve for the past 24 hours.
        average_factor = sum(self._HOUR_FACTORS.values()) / 24.0

        rows = []
        for offset in range(LSTM_WINDOW_SIZE - 1, -1, -1):
            past_hour   = (current_hour - offset) % 24
            past_factor = self._HOUR_FACTORS.get(past_hour, 0.1)
            # Realistic demand for the past hour based on the route's overall average
            past_demand = avg_demand * (past_factor / average_factor)

            row = [
                past_demand,                                        # passenger_count
                past_hour,                                          # hour
                input_features.get("day_of_week", 1),
                input_features.get("is_weekend", 0),
                input_features.get("is_holiday", 0),
                input_features.get("event_weight", 0.0),
                input_features.get("temperature_c", 25.0),
                input_features.get("rainfall_mm", 0.0),
                input_features.get("weather_severity_score", 0.0),
                1 if past_hour in PEAK_HOURS else 0,               # is_peak_hour
            ]
            rows.append(row)

        window_data = np.array(rows, dtype=np.float32)   # (24, 10)

        try:
            window_scaled = self.scaler_X.transform(window_data)
        except Exception as e:
            logger.warning(f"Scaler transform failed ({e}); using raw values.")
            window_scaled = window_data

        return window_scaled.reshape(1, LSTM_WINDOW_SIZE, -1)

    def _get_lstm_prediction(self, sequence: np.ndarray) -> float:
        """Run LSTM inference and inverse-transform the result."""
        try:
            pred_scaled = self.lstm_model.predict(sequence, verbose=0)
            pred_orig   = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
            return float(max(0, pred_orig))
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return 0.0

    def _get_rf_prediction(self, input_features: dict) -> float:
        """Run RF inference using the input feature dict."""
        try:
            rf_input = np.array([
                input_features.get(f, 0.0)
                for f in RF_FEATURES
            ]).reshape(1, -1)
            pred = self.rf_model.predict(rf_input)[0]
            return float(max(0, pred))
        except Exception as e:
            logger.warning(f"RF prediction failed: {e}")
            return 0.0

    def predict(self,
                hour: int,
                day_of_week: int,
                month: int,
                is_holiday: int = 0,
                event_weight: float = 0.0,
                temperature_c: float = 25.0,
                weather_condition: str = "Sunny",
                rainfall_mm: float = 0.0,
                route_id: str = "R01",
                avg_demand_estimate: float = 80.0) -> dict:
        """
        Generate a hybrid passenger demand prediction.

        Args:
            hour              : Hour of day (0-23)
            day_of_week       : Day (0=Mon, 6=Sun)
            month             : Month (1-12)
            is_holiday        : 1 if public holiday, else 0
            event_weight      : Event intensity near this stop (0-1)
            temperature_c     : Temperature in Celsius
            weather_condition : One of the 6 weather conditions
            rainfall_mm       : Rainfall in mm
            route_id          : Route identifier (used for context)
            avg_demand_estimate: Best estimate of current base demand

        Returns:
            dict with keys:
              - lstm_prediction    : float
              - rf_prediction      : float
              - hybrid_prediction  : float
              - predicted_demand   : int (rounded)
              - demand_category    : str ('Low'/'Medium'/'High'/'Very High')
              - confidence_note    : str
        """
        if not self._loaded:
            success = self.load_models()
            if not success:
                return {
                    "error": "Models not loaded. Run training scripts first.",
                    "predicted_demand": 0
                }

        # ── Compute derived features ──
        is_weekend            = 1 if day_of_week >= 5 else 0
        is_peak_hour          = 1 if hour in PEAK_HOURS else 0
        weather_severity_score= WEATHER_SEVERITY_MAP.get(weather_condition, 0.0)

        # Build the input feature dict (used by both models)
        input_features = {
            "passenger_count":        avg_demand_estimate,
            "hour":                   hour,
            "day_of_week":            day_of_week,
            "month":                  month,
            "is_weekend":             is_weekend,
            "is_holiday":             is_holiday,
            "event_weight":           event_weight,
            "temperature_c":          temperature_c,
            "rainfall_mm":            rainfall_mm,
            "weather_severity_score": weather_severity_score,
            "is_peak_hour":           is_peak_hour,
            # Lag and rolling features (approx from avg_demand_estimate)
            "lag_1h":                 avg_demand_estimate * 0.95,
            "lag_2h":                 avg_demand_estimate * 0.90,
            "lag_3h":                 avg_demand_estimate * 0.88,
            "rolling_avg_3h":         avg_demand_estimate * 0.92,
            "rolling_avg_6h":         avg_demand_estimate * 0.90,
        }

        # ── Run models ──
        lstm_sequence = self._build_lstm_sequence(input_features)
        lstm_pred     = self._get_lstm_prediction(lstm_sequence)
        rf_pred       = self._get_rf_prediction(input_features)

        # ── Hybrid combination ──
        hybrid_pred = (HYBRID_LSTM_WEIGHT * lstm_pred) + (HYBRID_RF_WEIGHT * rf_pred)
        hybrid_pred = max(0.0, hybrid_pred)

        category = classify_demand(hybrid_pred)

        # Confidence note for dashboard display
        if abs(lstm_pred - rf_pred) < 20:
            confidence = "High — Both models agree closely."
        elif abs(lstm_pred - rf_pred) < 50:
            confidence = "Moderate — Some divergence between models."
        else:
            confidence = "Low — Models diverge significantly; interpret with caution."

        return {
            "lstm_prediction":   round(lstm_pred, 1),
            "rf_prediction":     round(rf_pred, 1),
            "hybrid_prediction": round(hybrid_pred, 1),
            "predicted_demand":  int(round(hybrid_pred)),
            "demand_category":   category,
            "confidence_note":   confidence,
            "input_summary": {
                "hour": hour, "day_of_week": day_of_week, "month": month,
                "weather": weather_condition, "is_holiday": is_holiday,
                "event_weight": event_weight
            }
        }


def get_predictor() -> HybridPredictor:
    """
    Factory function — returns a loaded HybridPredictor instance.
    Use this in the dashboard to avoid re-loading models on every interaction.
    """
    p = HybridPredictor()
    p.load_models()
    return p


# ─────────────────────────────────────────────
# QUICK TEST (run this module directly to verify)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Testing HybridPredictor...")

    predictor = HybridPredictor()
    success   = predictor.load_models()

    if not success:
        print("\n⚠️  Models not found. Run training scripts first:")
        print("   python src/train_lstm.py")
        print("   python src/train_random_forest.py")
    else:
        # Test prediction: Rush hour, Tuesday, rainy March day
        result = predictor.predict(
            hour=8,
            day_of_week=1,
            month=3,
            is_holiday=0,
            event_weight=0.0,
            temperature_c=24.5,
            weather_condition="Heavy Rain",
            rainfall_mm=12.0,
        )

        print("\n=== Hybrid Prediction Result ===")
        print(f"  LSTM Prediction  : {result['lstm_prediction']}")
        print(f"  RF Prediction    : {result['rf_prediction']}")
        print(f"  Hybrid Prediction: {result['hybrid_prediction']}")
        print(f"  Demand Category  : {result['demand_category']}")
        print(f"  Confidence       : {result['confidence_note']}")
