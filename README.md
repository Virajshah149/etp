# 🚌 Eco-Transit Pulse
### A Multi-Modal Urban Mobility Optimizer using Hybrid LSTM-Random Forest Models and Event-Driven Prescriptive Analytics

> **Undergraduate Internship Project** | Computer Science Engineering | 45-Day Program

---

## 📌 Project Overview

**Eco-Transit Pulse** is an intelligent urban transit demand forecasting and optimization system built for Bengaluru's bus network. It addresses a core challenge in public transportation: **routes are either overcrowded or underutilized** because traditional scheduling ignores real-world context like weather, holidays, and local events.

The system:
- **Predicts** hourly passenger demand using a hybrid LSTM + Random Forest model
- **Identifies** Ghost Hotspots — high-demand areas with insufficient transit coverage
- **Recommends** actionable transit improvements through an interactive dashboard

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Hybrid ML Model** | LSTM (60%) + Random Forest (40%) weighted prediction |
| 📊 **Interactive Dashboard** | 6-page Streamlit app with Plotly charts |
| 🗺️ **Ghost Hotspot Maps** | Folium heatmaps + K-Means cluster visualization |
| 🌦️ **Weather-Aware** | IMD climate-normals-based weather simulation |
| 📅 **Holiday-Aware** | Indian public holiday calendar integrated |
| 🎪 **Event-Aware** | Local event demand spikes modelled |
| 💡 **Smart Recommendations** | Rule-based prescriptive analytics |
| ⬇️ **CSV Export** | Download filtered prediction results |

---

## 🗃️ Dataset Strategy

The dataset is a **hybrid of real public data and realistic simulations**:

| Component | Source | Type |
|-----------|--------|------|
| Route topology | BMTC Bengaluru GTFS schema (OpenStreetMap-derived) | **Real** |
| Stop coordinates | Real Bengaluru geographic coordinates | **Real** |
| Holiday calendar | Government of India Gazette 2024 | **Real** |
| Weather patterns | IMD Bengaluru historical climate normals | **Real** (distribution-based) |
| Passenger counts | Multiplicative demand model (peak/weather/event) | **Simulated** |
| Event occurrences | Representative local events (tech, cultural, sports) | **Simulated** |

> **Note:** BMTC does not publish granular hourly passenger count data publicly. Counts are generated using a validated multiplicative demand model that replicates known patterns (morning/evening peaks, weekend drop, rain suppression, event boosts).

---

## 🏗️ Project Structure

```
EcoTransitPulse/
│
├── data/
│   ├── raw/                        # transit_demand_raw.csv
│   └── processed/                  # transit_demand_processed.csv
│                                   # transit_demand_featured.csv
│
├── models/                         # Saved trained models
│   ├── lstm_model.keras
│   ├── rf_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
│
├── src/
│   ├── utils.py                    # Shared constants, paths, helpers
│   ├── data_collection.py          # Hybrid dataset generation
│   ├── preprocessing.py            # Cleaning & encoding pipeline
│   ├── feature_engineering.py      # Lag, rolling, flag features
│   ├── train_lstm.py               # LSTM model training
│   ├── train_random_forest.py      # Random Forest training
│   ├── clustering.py               # K-Means Ghost Hotspot detection
│   └── prediction.py               # Hybrid prediction engine
│
├── dashboard/
│   └── app.py                      # Streamlit multi-page dashboard
│
├── outputs/
│   ├── graphs/                     # Training plots, EDA charts
│   ├── heatmaps/                   # ghost_hotspots_map.html
│   └── reports/                    # model_metrics.json, cluster_info.json
│
├── notebooks/                      # (Optional) Jupyter EDA notebooks
│
├── run_pipeline.py                 # One-command pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Technology Stack

| Category | Library | Version |
|----------|---------|---------|
| Language | Python | 3.10+ |
| Data | Pandas, NumPy | 2.2.2, 1.26.4 |
| Deep Learning | TensorFlow/Keras | 2.16.1 |
| Machine Learning | Scikit-learn | 1.4.2 |
| Dashboard | Streamlit | 1.35.0 |
| Charts | Plotly | 5.22.0 |
| Maps | Folium, streamlit-folium | 0.16.0, 0.20.0 |
| Visualization | Matplotlib, Seaborn | 3.8.4, 0.13.2 |
| Model Persistence | Joblib | 1.4.2 |

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd EcoTransitPulse
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Option A — One command (recommended)
```bash
python run_pipeline.py
```
This runs all 6 pipeline steps automatically, then:
```bash
streamlit run dashboard/app.py
```

### Option B — Step by step
```bash
# Step 1: Generate dataset
python src/data_collection.py

# Step 2: Clean & preprocess
python src/preprocessing.py

# Step 3: Engineer features
python src/feature_engineering.py

# Step 4: Train LSTM (takes 5-15 min depending on hardware)
python src/train_lstm.py

# Step 5: Train Random Forest
python src/train_random_forest.py

# Step 6: Generate Ghost Hotspot map
python src/clustering.py

# Step 7: Launch dashboard
streamlit run dashboard/app.py
```

### Skip training if models already exist
```bash
python run_pipeline.py --skip-training
```

---

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI overview, dataset stats, model performance summary |
| 📂 **Data Explorer** | Filter, search, and download the dataset |
| 📊 **EDA** | Interactive charts: hourly/daily/monthly trends, weather impact, correlations |
| 🔮 **Demand Prediction** | Enter conditions → get hybrid demand forecast with gauge chart |
| 🗺️ **Ghost Hotspots** | Interactive Folium map with heatmap + cluster markers |
| 💡 **Recommendations** | Rule-based prescriptive suggestions by priority |

---

## 🤖 Model Architecture

### LSTM Model
```
Input (24 time steps × 10 features)
    → LSTM(64, return_sequences=True)
    → Dropout(0.2)
    → LSTM(32)
    → Dropout(0.2)
    → Dense(1)
```
- **Window:** 24 hours of historical data
- **Training:** EarlyStopping + ReduceLROnPlateau
- **Metrics:** MAE, RMSE, R²

### Random Forest
- **n_estimators:** 100 trees
- **max_depth:** 15
- **Features:** 15 engineered features (time + weather + lag + rolling)
- **Metrics:** MAE, RMSE, R², OOB Score

### Hybrid Prediction
```
Final = (0.60 × LSTM_prediction) + (0.40 × RF_prediction)
```
LSTM captures temporal patterns; RF captures external factor effects.

### K-Means Clustering (Ghost Hotspots)
- **Features:** [latitude, longitude, avg_demand] (StandardScaler normalised)
- **k = 5** (selected via elbow method)
- **Ghost Hotspot criteria:** demand ≥ 75th percentile AND routes served ≤ 1

---

## 📂 Key Output Files

| File | Description |
|------|-------------|
| `data/raw/transit_demand_raw.csv` | 6-month hourly transit dataset |
| `data/processed/transit_demand_featured.csv` | ML-ready featured dataset |
| `models/lstm_model.keras` | Trained LSTM model |
| `models/rf_model.pkl` | Trained Random Forest |
| `outputs/heatmaps/ghost_hotspots_map.html` | Interactive Folium map |
| `outputs/graphs/*.png` | EDA + training visualization plots |
| `outputs/reports/model_metrics.json` | LSTM + RF evaluation metrics |
| `outputs/reports/cluster_info.json` | K-Means cluster descriptions |

---

## 🔭 Future Scope

- [ ] Integrate live GTFS feeds from BMTC open data portal
- [ ] Connect real-time weather API (OpenWeatherMap)
- [ ] Add ARIMA/Prophet model for comparison
- [ ] Deploy on Streamlit Cloud for public access
- [ ] Mobile-responsive layout improvements
- [ ] Route optimization using linear programming

---

## 👥 Team

| Name | Role |
|------|------|
| Student 1 | Data pipeline, LSTM model, feature engineering |
| Student 2 | Random Forest, clustering, dashboard, documentation |

**Internship Duration:** 45 days  
**Domain:** Intelligent Transportation Systems / Data Science

---

## 📄 License

This project is developed as part of an undergraduate internship program.  
For academic and educational use only.
