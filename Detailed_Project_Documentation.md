# Eco-Transit Pulse: Detailed Project Documentation & Technical Architecture

## 1. Executive Summary
**Eco-Transit Pulse** is a comprehensive, data-driven Urban Mobility Optimizer designed to address the inefficiencies present in modern public transportation networks. Built primarily for the Bengaluru Metropolitan Transport Corporation (BMTC) context, the project leverages advanced Machine Learning (ML) and Deep Learning (DL) techniques to forecast passenger demand and optimize route coverage. 

By utilizing a Hybrid Machine Learning approach combining Long Short-Term Memory (LSTM) neural networks with Random Forest Regressors, the system achieves highly accurate hourly passenger predictions. Furthermore, it incorporates unsupervised spatial clustering (K-Means) to identify critical "Ghost Hotspots"—areas with massive passenger footfall but inadequate bus services.

---

## 2. Problem Statement in Urban Mobility
Public transit systems often operate on rigid, static schedules that do not adapt to dynamic daily conditions. This leads to two major issues:
1. **Resource Wastage (Ghost Buses):** Buses running on off-peak hours or empty routes waste fuel, increase carbon emissions, and incur unnecessary operational costs.
2. **Passenger Bottlenecks:** Sudden changes in weather (e.g., heavy rain) or local events cause sudden spikes in demand, leaving passengers stranded due to a lack of dynamic bus allocation.

**The Objective:** To build an intelligent system that predicts exact hourly ridership demand and visually flags areas suffering from low transit coverage, allowing city planners to deploy buses dynamically.

---

## 3. Data Pipeline Architecture

The success of any Machine Learning model relies entirely on the quality of its data. Because granular, hourly passenger data is rarely made public by transit authorities, this project uses a highly sophisticated **hybrid simulation engine**.

### 3.1 Data Collection & Simulation (`data_collection.py`)
The dataset is rooted in real-world General Transit Feed Specification (GTFS) data. We extracted real bus route topologies and stop coordinates (latitude/longitude) from Bengaluru.
To simulate ridership, the script uses a **Multiplicative Demand Model**:
- **Base Demand:** Every route is assigned a baseline ridership volume.
- **Temporal Multipliers:** The script applies multipliers based on the time of day (e.g., morning rush 8 AM - 10 AM gets a 2.5x multiplier, while 2 AM gets a 0.1x multiplier).
- **Weather Simulation:** We integrate historical climate normals from the Indian Meteorological Department (IMD). If it rains, the model automatically reduces routine travel demand but increases demand for specific sheltered stops.
- **Event Triggers:** Simulated local events (e.g., cricket matches at Chinnaswamy Stadium, tech conferences in Electronic City) apply localized 50% demand spikes to nearby bus stops.

### 3.2 Data Preprocessing (`preprocessing.py`)
Raw data is rarely ready for machine learning. The preprocessing script performs several vital operations:
1. **Missing Value Imputation:** Scans the dataset for `NaN` (Not a Number) values and fills them using forward-fill techniques or median averages to ensure the neural networks do not crash.
2. **Outlier Detection:** Removes mathematically impossible passenger counts (e.g., negative passengers or a bus carrying 5,000 people at once) using Interquartile Range (IQR) bounds.
3. **Encoding:** Machine learning algorithms only understand numbers. Categorical variables (like "Sunny" or "Rainy") are converted into numerical formats.

### 3.3 Feature Engineering (`feature_engineering.py`)
This is where the data is enriched to give the AI context.
- **Rolling Averages:** The script calculates the 7-day moving average of passenger demand for every route. This helps the AI understand long-term baseline trends.
- **Time Lags:** We create a feature called `demand_lag_1h` (demand exactly one hour ago) and `demand_lag_24h` (demand exactly yesterday at this time). This is crucial for the LSTM network to understand sequential flow.
- **Weather Scoring:** Combines temperature and precipitation into a single numerical `weather_score` indicating how favorable the weather is for public transit.

---

## 4. Machine Learning Architecture (Deep Dive)

A major highlight of this project is the **Hybrid Prediction Engine**, which proves a deep understanding of model strengths and weaknesses.

### 4.1 The Deep Learning Model: LSTM (Long Short-Term Memory)
- **Concept:** Standard neural networks suffer from "amnesia"—they evaluate every data point in isolation. LSTMs are a type of Recurrent Neural Network (RNN) that contain internal "memory cells" and "gates" (Forget Gate, Input Gate, Output Gate). These gates decide what information from the past is relevant and should be kept, and what should be forgotten.
- **Implementation in Project:** The LSTM is fed sequences of data (the last 24 hours). It excels at mapping out the smooth, continuous wave of daily ridership (the morning climb, the afternoon lull, the evening peak).
- **Architecture:** The model features multiple LSTM layers intertwined with `Dropout` layers (to prevent overfitting, ensuring the model doesn't just memorize the training data).
- **Weighting:** Contributes **60%** to the final prediction.

### 4.2 The Machine Learning Model: Random Forest Regressor
- **Concept:** Random Forest is an ensemble learning method. It builds hundreds of mathematical "Decision Trees" during training. When asked to make a prediction, every tree votes on the outcome, and the average is taken.
- **Implementation in Project:** While LSTM is great at time patterns, Random Forest is exceptional at handling abrupt, non-linear categorical changes. If the data suddenly indicates `is_holiday = True` and `weather_condition = Heavy Rain`, the Random Forest immediately slices the data to predict a massive drop in demand, reacting faster than an LSTM looking at the past 24 hours.
- **Weighting:** Contributes **40%** to the final prediction.

### 4.3 The Ensemble Strategy (HybridPredictor)
When you ask the dashboard for a prediction, the `prediction.py` script routes the input to *both* models simultaneously. 
The final mathematical output is:
`Final_Demand = (LSTM_Output * 0.60) + (RandomForest_Output * 0.40)`
This yields a highly robust model that rarely over-predicts or under-predicts, achieving excellent R² (R-Squared) scores.

---

## 5. Unsupervised Learning: K-Means Spatial Analytics

Predicting demand is only half the battle. Transit authorities also need to know *where* to put new bus stops. This project uses Unsupervised Learning to map the geographic network.

### 5.1 K-Means Clustering (`clustering.py`)
K-Means groups data points into distinct clusters based on similarity. In this project, the features grouped are:
1. Latitude
2. Longitude
3. Average Passenger Demand

### 5.2 The Elbow Method (Why k=5?)
How do we know how many zones to divide the city into? We used the **Elbow Method**. The algorithm was run multiple times, changing 'k' (the number of clusters) from 2 to 10, calculating the "Inertia" (the sum of squared distances of samples to their closest cluster center) each time. When plotted on a graph, the curve dropped sharply and then flattened out at `k=5`, creating an "elbow" shape. This proved mathematically that Bengaluru's transit stops in our dataset are best categorized into 5 major zones.

### 5.3 Ghost Hotspot Identification
Once the stops are grouped into 5 clusters, the algorithm runs a filtering script.
A stop is flagged as a **Ghost Hotspot** if it meets two strict conditions:
1. **Demand Threshold:** Its average passenger demand falls in the top 75th percentile of the entire city network.
2. **Coverage Threshold:** It is served by strictly `1` bus route.

These stops are highlighted in **red** with warning markers on the Folium map, instantly telling transit planners: *"This area has massive crowds but almost no buses. Send new routes here immediately."*

---

## 6. Front-End Deployment: The Streamlit Dashboard (`app.py`)

The entire backend intelligence is presented via a professional, modular web application built using the Streamlit framework.

- **1. Home:** Displays overall Key Performance Indicators (KPIs) like total records, bus stops, and model evaluation metrics (MAE, RMSE, R²).
- **2. Data Explorer:** Allows the user to view the raw Pandas DataFrame, filter by route, and download the data to CSV.
- **3. Exploratory Data Analysis (EDA):** Uses `Plotly` to render highly interactive charts. It visualizes the impact of temperature on ridership, hourly trend lines, and feature correlation matrices.
- **4. Demand Prediction:** The interactive inference UI. The user adjusts sliders (Time, Temperature, Rain, Holiday status) and clicks "Predict." The inputs are parsed, scaled using the saved `StandardScaler`, and fed into the LSTM and RF models to generate a live prediction displayed on a gauge chart.
- **5. Ghost Hotspots:** Uses `Folium` to render an interactive map of Bengaluru. It layers a demand heatmap over the map and plots custom markers for the 5 clusters and the Ghost Hotspots.
- **6. Recommendations:** A prescriptive analytics module that generates automated, rule-based text recommendations (High, Medium, Low priority) for transit planners based on the data outputs.

---

## 7. Technology Stack & Dependencies

- **Core Programming:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy` (For fast vectorized array operations).
- **Machine Learning / Deep Learning:** 
  - `scikit-learn` (Random Forest, K-Means, StandardScaler, Metrics)
  - `tensorflow` / `keras` (LSTM Neural Network construction, compiling, and training)
- **Data Visualization:**
  - `plotly` (Interactive dashboard charts)
  - `folium` & `streamlit-folium` (Geospatial mapping)
  - `matplotlib` & `seaborn` (Static backend plotting and elbow curves)
- **Web Framework:** `streamlit` (UI rendering and state management)
- **Model Persistence:** `joblib` (Saving and loading the trained Random Forest and Scaler objects without retraining).

---

## 8. Setup & Execution Lifecycle

The project utilizes a unified execution script (`run_pipeline.py`) designed to orchestrate the entire lifecycle:
1. Checks for required directories (`models/`, `outputs/`, `data/`).
2. Executes data generation and engineering consecutively.
3. Triggers model training, saving the weights (`.keras` and `.pkl` files) to the disk.
4. Executes the spatial clustering to generate the HTML maps.

Once the pipeline completes, the user simply runs `streamlit run dashboard/app.py` to mount the local web server and interact with the finalized models.

---

## 9. Conclusion
Eco-Transit Pulse demonstrates a full-stack data science lifecycle. By combining rigorous data engineering, hybrid deep-learning predictive models, and unsupervised spatial clustering, it successfully transitions from theoretical data analysis to a deployable, interactive prescriptive analytics tool designed to solve real-world urban mobility challenges.
