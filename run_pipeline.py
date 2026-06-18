"""
run_pipeline.py — Full Project Pipeline Orchestrator
=====================================================
Run this single script to execute the complete Eco-Transit Pulse
data + model pipeline from scratch.

Steps:
  1. Generate hybrid dataset (data_collection.py)
  2. Clean & preprocess data (preprocessing.py)
  3. Engineer features (feature_engineering.py)
  4. Train LSTM model (train_lstm.py)
  5. Train Random Forest model (train_random_forest.py)
  6. Run K-Means clustering (clustering.py)

After this, launch the dashboard with:
  streamlit run dashboard/app.py

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-training    # Skip model training (use saved models)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Ensure src/ is on the path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import (
    setup_logger, ensure_dirs,
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURED_DATA_PATH,
    LSTM_MODEL_PATH, RF_MODEL_PATH, KMEANS_MODEL_PATH
)

logger = setup_logger("pipeline")


def run_step(step_name: str, func, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    logger.info(f"\n{'='*55}")
    logger.info(f"  STEP: {step_name}")
    logger.info(f"{'='*55}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"  ✅ {step_name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        logger.error(f"  ❌ {step_name} FAILED: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Eco-Transit Pulse Pipeline")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training if models already exist")
    args = parser.parse_args()

    logger.info("\n" + "="*55)
    logger.info("  🚌 ECO-TRANSIT PULSE — FULL PIPELINE")
    logger.info("="*55)

    # Ensure all directories exist
    ensure_dirs()

    # ── Step 1: Data Collection ──
    from data_collection import generate_dataset, save_raw_data
    df_raw = run_step(
        "Data Collection (Hybrid Dataset Generation)",
        lambda: generate_dataset("2024-01-01", "2024-06-30")
    )
    save_raw_data(df_raw)

    # ── Step 2: Preprocessing ──
    from preprocessing import run_preprocessing_pipeline
    df_processed = run_step(
        "Data Preprocessing",
        run_preprocessing_pipeline
    )

    # ── Step 3: Feature Engineering ──
    from feature_engineering import run_feature_engineering
    df_featured = run_step(
        "Feature Engineering",
        run_feature_engineering
    )

    # ── Step 4: LSTM Training ──
    if args.skip_training and LSTM_MODEL_PATH.exists():
        logger.info("Skipping LSTM training (model exists).")
    else:
        import train_lstm as tl
        import pandas as pd

        def train_lstm_step():
            df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["datetime"])
            X_train, X_val, y_train, y_val, sX, sy = tl.prepare_lstm_data(df)
            n_feats = X_train.shape[2]
            model   = tl.build_lstm_model((tl.LSTM_WINDOW_SIZE, n_feats))
            history = tl.train_lstm(model, X_train, y_train, X_val, y_val)
            metrics, y_true, y_pred = tl.evaluate_lstm(model, X_val, y_val, sy)
            tl.ensure_dirs()
            tl.plot_training_history(history)
            tl.plot_predictions(y_true, y_pred)
            existing = tl.load_metrics()
            existing.update(metrics)
            tl.save_metrics(existing)
            return metrics

        run_step("LSTM Model Training", train_lstm_step)

    # ── Step 5: Random Forest Training ──
    if args.skip_training and RF_MODEL_PATH.exists():
        logger.info("Skipping RF training (model exists).")
    else:
        import train_random_forest as trf
        import pandas as pd

        def train_rf_step():
            df = trf.load_featured_data()
            X_train, X_test, y_train, y_test, feat_names = trf.prepare_rf_data(df)
            rf = trf.train_random_forest(X_train, y_train)
            metrics, y_true, y_pred = trf.evaluate_rf(rf, X_test, y_test)
            trf.save_rf_model(rf)
            existing = trf.load_metrics()
            existing.update(metrics)
            trf.save_metrics(existing)
            trf.ensure_dirs()
            trf.plot_feature_importance(rf, feat_names)
            trf.plot_rf_predictions(y_true, y_pred)
            return metrics

        run_step("Random Forest Training", train_rf_step)

    # ── Step 6: Clustering ──
    from clustering import run_clustering
    run_step("K-Means Ghost Hotspot Detection", run_clustering)

    # ── Done ──
    logger.info("\n" + "="*55)
    logger.info("  🎉 PIPELINE COMPLETE!")
    logger.info("="*55)
    logger.info(f"  Raw data      → {RAW_DATA_PATH}")
    logger.info(f"  Processed     → {PROCESSED_DATA_PATH}")
    logger.info(f"  Featured      → {FEATURED_DATA_PATH}")
    logger.info(f"  LSTM model    → {LSTM_MODEL_PATH}")
    logger.info(f"  RF model      → {RF_MODEL_PATH}")
    logger.info(f"  Cluster model → {KMEANS_MODEL_PATH}")
    logger.info("\n  Launch dashboard:")
    logger.info("    streamlit run dashboard/app.py")
    logger.info("="*55)


if __name__ == "__main__":
    main()
