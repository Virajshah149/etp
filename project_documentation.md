# Eco-Transit Pulse: Project Documentation

## 1. Project Overview
**Eco-Transit Pulse** is a data-driven Urban Mobility Optimizer designed to analyze and predict bus passenger demand. The project leverages historical transit data and contextual factors like weather, holidays, and local events to provide actionable insights for transit authorities.

The primary goals of the project are:
1. To predict hourly passenger demand for various bus routes.
2. To identify "Ghost Hotspots" — bus stops with high passenger demand but insufficient bus route coverage.
3. To provide rule-based recommendations for transit planners to optimize fleet deployment.

## 2. Dataset Information
The dataset used in this project is a hybrid dataset structured around the General Transit Feed Specification (GTFS). 
- **Route Topology:** Inspired by Bengaluru's BMTC bus network.
- **Passenger Demand & Context:** Synthetically generated using statistically realistic distributions to simulate real-world conditions (including rush hours, weather impacts, and weekend dips).
- **Features Engineered:**
  - Temporal features: Hour, day of week, month, is_weekend, is_holiday.
  - Weather features: Temperature, rainfall, weather_score.
  - Historical lags: 1-hour lag, 24-hour lag, 7-day rolling average.

## 3. Machine Learning Architecture
The prediction engine uses a **Hybrid Model Strategy** to balance long-term trends with immediate contextual factors.

### A. LSTM (Long Short-Term Memory) Network
- **Weight:** 60% of final prediction
- **Purpose:** Time-series forecasting. It captures sequential patterns such as the daily morning and evening rush hour curves.
- **Input:** Sequences of historical passenger demand (e.g., the last 24 hours).

### B. Random Forest Regressor
- **Weight:** 40% of final prediction
- **Purpose:** Contextual forecasting. It excels at understanding non-linear relationships between categorical/environmental features (e.g., how heavy rain or a specific holiday instantly impacts demand).
- **Input:** Engineered features like `weather_score`, `is_holiday`, `event_flag`, and `hour`.

### C. K-Means Clustering (Ghost Hotspot Detection)
- **Algorithm:** K-Means (k=5)
- **Purpose:** Spatial analysis of bus stops.
- **Methodology:** Groups bus stops based on their geographic coordinates (latitude, longitude) and average passenger demand. Stops that fall into the 75th percentile for demand but are served by only 1 route are flagged as "Ghost Hotspots."

## 4. Pipeline Modules
The project is organized into modular Python scripts:
- `data_collection.py`: Generates the hybrid GTFS dataset.
- `preprocessing.py`: Cleans the data, handles outliers, and normalizes features.
- `feature_engineering.py`: Creates time-lagged variables and rolling averages.
- `train_lstm.py` / `train_random_forest.py`: Trains the predictive models and saves the artifacts (`.keras` / `.pkl`).
- `clustering.py`: Runs the K-Means algorithm and generates the Folium heatmap.
- `prediction.py`: Contains the `HybridPredictor` class used for live inference on the dashboard.

## 5. Technology Stack
- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** TensorFlow (Keras), Scikit-learn
- **Visualization:** Plotly Express, Folium, Matplotlib
- **Web Dashboard:** Streamlit
