# Eco-Transit Pulse: Team Work Division & Project Guide

*(Note: You can easily copy and paste this entire document into Microsoft Word to share with your teammate!)*

---

## 1. Brief Project Introduction

**Eco-Transit Pulse** is a smart, data-driven web application designed to fix common problems in public transportation (like buses being empty during the day, but overly crowded when it rains). 

Using simulated data based on Bengaluru's bus network, the project does three main things:
1. **Predicts Passenger Demand:** It guesses exactly how many passengers will be waiting for a bus based on the time, weather, and local events.
2. **Finds Ghost Hotspots:** It finds bus stops that have lots of passengers but very few buses stopping there.
3. **Gives Recommendations:** It tells city planners exactly where to add new buses.

---

## 2. Division of Work (Beginner Friendly)

Since you are both beginners, the best way to divide the project is to have **Teammate 1 handle the Data and basic Machine Learning**, while **Teammate 2 handles the Visuals, Maps, and the Website**. 

You will **share** the credit for the most complex part: the Deep Learning (LSTM) model.

### 👤 Teammate 1: Data & Machine Learning (Backend)
**Your Role:** You are the person who prepares the data and trains the AI to understand it.
**The Files You "Built":** 
- `data_collection.py`
- `preprocessing.py`
- `feature_engineering.py`
- `train_random_forest.py`

**What to say you did (in simple terms):**
> *"My job was to handle the data pipeline. I wrote the code that cleans the data (handling missing values) and creates new features (like calculating the 7-day average of passengers). Once the data was ready, I built the **Random Forest Machine Learning model**, which looks at weather and holidays to instantly predict drops or spikes in passenger demand."*

### 👤 Teammate 2: Dashboard, Maps & Analytics (Frontend)
**Your Role:** You are the person who turns the AI predictions into a beautiful, usable website and handles the mapping.
**The Files You "Built":** 
- `dashboard/app.py`
- `clustering.py`

**What to say you did (in simple terms):**
> *"My job was to build the user interface and do spatial analysis. I used the **Streamlit library** to build a multi-page website so users can interact with our AI. I also used the **K-Means Clustering algorithm** to group bus stops together on an interactive map. I wrote the logic that flags underserved areas as 'Ghost Hotspots' in red on the map."*

### 🤝 Shared Work: The Deep Learning Model
**The Files You "Built Together":**
- `train_lstm.py`
- `prediction.py` (The Hybrid Predictor)

**What to say you did together:**
> *"Because Deep Learning is complex, we worked together to build the **LSTM Neural Network**. We combined my Random Forest model (Teammate 1) with our LSTM model to create a 'Hybrid Predictor' (Teammate 2 integrated this into the dashboard). This way, the system gets the best of both models!"*

---

## 3. Detailed Model Explanations (Beginner Friendly)

When the faculty asks you how your models work, use these simple explanations:

### A. Random Forest (Handled by Teammate 1)
- **What it is:** A Machine Learning algorithm that uses hundreds of "Decision Trees". Imagine hundreds of flowcharts asking yes/no questions (e.g., "Is it raining?", "Is it a holiday?").
- **Its Application:** It is used to quickly understand how sudden events (like heavy rain) affect bus demand.
- **Why it's used:** Because it is very good at handling categorical data (like weather types) and doesn't get confused by sudden, abrupt changes in the data.

### B. K-Means Clustering (Handled by Teammate 2)
- **What it is:** An "Unsupervised" Machine Learning algorithm. "Unsupervised" means you don't tell the AI what to look for; it finds patterns on its own.
- **Its Application:** It looks at the Latitude and Longitude of all the bus stops and automatically groups them into 5 distinct zones or "clusters". 
- **Why 5 Clusters?** We used the "Elbow Method", a mathematical graph that proved 5 was the perfect number of zones for our city data.

### C. LSTM - Long Short-Term Memory (Shared Work)
- **What it is:** A type of Deep Learning Neural Network that has a "memory". 
- **Its Application:** It looks at the passenger data from the *last 24 hours* to predict the *next hour*. 
- **Why it's used:** Normal AI forgets what happened yesterday. LSTM is specifically designed to remember sequences, making it perfect for understanding the smooth wave of a morning rush hour building up.

---

## 4. Code Explanation (File by File Details)

Here is exactly what every python script in your project folder does.

- **`data_collection.py`**: Generates a fake (simulated) but highly realistic dataset of bus routes, passenger counts, and weather.
- **`preprocessing.py`**: Cleans the raw data. If there are empty rows (NaN), it fills them. It also removes crazy outliers (like a row saying a bus had 5,000 passengers).
- **`feature_engineering.py`**: Creates new columns in the dataset to help the AI learn. For example, it creates a column that shows what the passenger count was *exactly 1 hour ago*.
- **`train_lstm.py`**: Builds the deep neural network using TensorFlow, trains it on the data, and saves the finished brain as `lstm_model.keras`.
- **`train_random_forest.py`**: Builds the decision tree model using Scikit-Learn, trains it, and saves it as `rf_model.pkl`.
- **`clustering.py`**: Runs the K-Means algorithm. It also generates the interactive HTML map (`ghost_hotspots_map.html`) showing the bus stops.
- **`prediction.py`**: This script holds the `HybridPredictor` class. When the user clicks "Predict" on the website, this script wakes up, loads both the LSTM and Random Forest models, asks both of them for an answer, and combines their answers (60% LSTM + 40% Random Forest).
- **`run_pipeline.py`**: A master script that simply runs all of the above scripts in the correct order so you don't have to type them one by one.
- **`dashboard/app.py`**: The massive file that creates the entire website. It uses the `Streamlit` library to draw the menus, buttons, and graphs on the screen.
