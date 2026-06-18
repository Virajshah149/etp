"""
app.py — Eco-Transit Pulse: Streamlit Dashboard
================================================
Multi-page interactive dashboard with 6 sections:

  1. 🏠 Home             — Project overview + KPI cards
  2. 📂 Data Explorer    — Filterable dataset viewer
  3. 📊 EDA              — Interactive Plotly charts
  4. 🔮 Demand Prediction — Hybrid model inference UI
  5. 🗺️  Ghost Hotspots   — Folium map + cluster table
  6. 💡 Recommendations  — Rule-based smart suggestions

Run with:
  streamlit run dashboard/app.py
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATH SETUP — resolve project root from dashboard/
# ─────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = DASHBOARD_DIR.parent
SRC_DIR       = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import (
    FEATURED_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH,
    HOTSPOT_MAP_PATH, CLUSTER_INFO_PATH, METRICS_PATH,
    GRAPHS_DIR, MODELS_DIR,
    HYBRID_LSTM_WEIGHT, HYBRID_RF_WEIGHT,
    CITY_NAME, SIMULATION_NOTE,
    load_metrics, load_cluster_info, classify_demand
)

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Transit Pulse",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — clean, modern academic style
# ─────────────────────────────────────────────
CUSTOM_CSS = ""




# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the featured dataset, fall back to processed or raw if needed."""
    for path in [FEATURED_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH]:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["datetime", "date"])
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cluster_data() -> list:
    return load_cluster_info()


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    return load_metrics()


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the HybridPredictor once and cache it for the session."""
    try:
        from prediction import HybridPredictor
        p = HybridPredictor()
        p.load_models()
        return p
    except Exception as e:
        return None

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚌 Eco-Transit Pulse")
        st.caption("Urban Mobility Optimizer")

        st.write("---")

        if "page" not in st.session_state:
            st.session_state.page = "Home"

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"

        if st.button("📂 Data Explorer", use_container_width=True):
            st.session_state.page = "Data Explorer"

        if st.button("📊 Exploratory Analysis", use_container_width=True):
            st.session_state.page = "EDA"

        if st.button("🔮 Demand Prediction", use_container_width=True):
            st.session_state.page = "Prediction"

        if st.button("🗺️ Ghost Hotspots", use_container_width=True):
            st.session_state.page = "Hotspots"

        if st.button("💡 Recommendations", use_container_width=True):
            st.session_state.page = "Recommendations"

        st.write("---")
        st.write("**Tech Stack:** Python, Streamlit, TensorFlow, Scikit-learn")
        st.write("**Models:** LSTM + Random Forest")

        return st.session_state.page

# ══════════════════
# ═══════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════
def page_home(df: pd.DataFrame, metrics: dict):
    st.markdown("## 🏠 Eco-Transit Pulse Dashboard", unsafe_allow_html=True)

    st.markdown("""
    > **Eco-Transit Pulse** is an intelligent urban mobility optimizer for **Bengaluru's** bus network.
    > It combines **LSTM deep learning** (for time-series demand forecasting) with **Random Forest regression**
    > (for external factor analysis) to predict passenger demand and identify underserved transit areas —
    > called **Ghost Hotspots** — using K-Means clustering.
    """)

    # ── Data availability warning ──
    if df.empty:
        st.error("⚠️ No dataset found. Please run the data pipeline first:\n"
                 "`python src/data_collection.py` → `preprocessing.py` → `feature_engineering.py`")
        return

    # ── KPI Cards ──
    st.markdown("#### 📈 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)

    n_routes    = df["route_id"].nunique() if "route_id" in df.columns else "—"
    n_stops     = df["stop_id"].nunique() if "stop_id" in df.columns else "—"
    avg_demand  = f"{df['passenger_count'].mean():.0f}" if "passenger_count" in df.columns else "—"
    total_records = f"{len(df):,}"
    date_range  = (f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}"
                   if "date" in df.columns else "—")

    c1.metric("Total Routes", n_routes)
    c2.metric("Bus Stops", n_stops)
    c3.metric("Avg Demand / hr", avg_demand)
    c4.metric("Total Records", total_records)
    c5.metric("Date Range", date_range)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Performance Summary ──
    st.markdown("#### 🤖 Model Performance Summary")

    if metrics:
        mc1, mc2 = st.columns(2)
        lstm_m = metrics.get("lstm", {})
        rf_m   = metrics.get("random_forest", {})

        with mc1:
            st.markdown("**LSTM Model**")
            if lstm_m:
                st.metric("MAE",  f"{lstm_m.get('MAE', '—')}")
                st.metric("RMSE", f"{lstm_m.get('RMSE', '—')}")
                st.metric("R²",   f"{lstm_m.get('R2', '—')}")
            else:
                st.info("Run `train_lstm.py` to see metrics.")

        with mc2:
            st.markdown("**Random Forest Model**")
            if rf_m:
                st.metric("MAE",  f"{rf_m.get('MAE', '—')}")
                st.metric("RMSE", f"{rf_m.get('RMSE', '—')}")
                st.metric("R²",   f"{rf_m.get('R2', '—')}")
            else:
                st.info("Run `train_random_forest.py` to see metrics.")


    else:
        st.info("Train the models first to see performance metrics here.")

    # ── Quick Demand Distribution ──
    st.markdown("#### 📊 Passenger Demand Distribution")
    if not df.empty and "passenger_count" in df.columns:
        fig = px.histogram(
            df.sample(min(5000, len(df))),
            x="passenger_count",
            nbins=50,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"passenger_count": "Passenger Count"},
            title="Distribution of Hourly Passenger Counts"
        )
        fig.update_layout(
            
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)



    # ── Project Workflow ──

def page_data_explorer(df: pd.DataFrame):
    st.markdown("## 📂 Data Explorer", unsafe_allow_html=True)

    if df.empty:
        st.warning("Dataset not found. Run the data pipeline first.")
        return

    # ── Filters in sidebar-style expander ──
    with st.expander("🔽 Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            routes = ["All"] + sorted(df["route_id"].unique().tolist()) if "route_id" in df.columns else ["All"]
            sel_route = st.selectbox("Route", routes)

        with fc2:
            weathers = ["All"] + sorted(df["weather_condition"].unique().tolist()) if "weather_condition" in df.columns else ["All"]
            sel_weather = st.selectbox("Weather", weathers)

        with fc3:
            hours = ["All"] + list(range(24))
            sel_hour = st.selectbox("Hour of Day", hours)

        with fc4:
            holiday_opt = st.selectbox("Holiday", ["All", "Holiday Only", "Non-Holiday"])

    # Apply filters
    filtered = df.copy()
    if sel_route != "All":
        filtered = filtered[filtered["route_id"] == sel_route]
    if sel_weather != "All":
        filtered = filtered[filtered["weather_condition"] == sel_weather]
    if sel_hour != "All":
        filtered = filtered[filtered["hour"] == sel_hour]
    if holiday_opt == "Holiday Only":
        filtered = filtered[filtered["is_holiday"] == 1]
    elif holiday_opt == "Non-Holiday":
        filtered = filtered[filtered["is_holiday"] == 0]

    # ── Stats ──
    st.markdown(f"**Showing {len(filtered):,} records** after filters")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Records",   f"{len(filtered):,}")
    sc2.metric("Avg Demand",      f"{filtered['passenger_count'].mean():.1f}" if len(filtered) else "—")
    sc3.metric("Max Demand",      f"{filtered['passenger_count'].max()}" if len(filtered) else "—")
    sc4.metric("Holiday Records", f"{filtered['is_holiday'].sum():,}" if len(filtered) else "—")

    # ── Display columns ──
    display_cols = [c for c in [
        "datetime", "route_name", "stop_name", "passenger_count",
        "weather_condition", "temperature_c", "rainfall_mm",
        "is_holiday", "event_name", "event_weight"
    ] if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].head(500).reset_index(drop=True),
        use_container_width=True,
        height=380
    )

    # ── Summary Statistics ──
    with st.expander("📐 Summary Statistics"):
        num_cols = filtered.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(filtered[num_cols].describe().round(2), use_container_width=True)

    # ── Download ──
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="etp_filtered_data.csv",
        mime="text/csv"
    )


# ═══════════════════════════════════════════════
# PAGE 3 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════
def page_eda(df: pd.DataFrame):
    st.markdown("## 📊 Exploratory Data Analysis", unsafe_allow_html=True)

    if df.empty:
        st.warning("Dataset not found.")
        return

    # ── Tab layout ──
    tabs = st.tabs([
        "⏰ Temporal Trends",
        "🛣️ Route Analysis",
        "🌦️ Weather & Events",
        "🔥 Correlation"
    ])

    # ── Tab 1: Temporal ──
    with tabs[0]:
        st.markdown("##### Hourly Demand Pattern")
        hourly_avg = df.groupby("hour")["passenger_count"].mean().reset_index()
        fig_hour = px.bar(
            hourly_avg, x="hour", y="passenger_count",
            color="passenger_count",
            color_continuous_scale="Blues",
            labels={"hour": "Hour of Day", "passenger_count": "Avg Passenger Count"},
            title="Average Hourly Passenger Demand"
        )
        fig_hour.update_layout(
            
            coloraxis_showscale=False, height=350
        )
        st.plotly_chart(fig_hour, use_container_width=True)

        st.markdown("##### Daily Demand Pattern")
        if "day_of_week" in df.columns:
            day_labels = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily_avg  = df.groupby("day_of_week")["passenger_count"].mean().reset_index()
            daily_avg["day_name"] = daily_avg["day_of_week"].apply(lambda x: day_labels[x])
            fig_day = px.bar(
                daily_avg, x="day_name", y="passenger_count",
                color="passenger_count",
                color_continuous_scale="Teal",
                labels={"day_name": "Day of Week", "passenger_count": "Avg Demand"},
                title="Average Daily Passenger Demand"
            )
            fig_day.update_layout(
                
                coloraxis_showscale=False, height=350
            )
            st.plotly_chart(fig_day, use_container_width=True)

        st.markdown("##### Monthly Trend")
        if "month" in df.columns:
            month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly = df.groupby("month")["passenger_count"].agg(["mean","max","min"]).reset_index()
            monthly["month_name"] = monthly["month"].map(month_labels)
            fig_month = go.Figure()
            fig_month.add_trace(go.Scatter(
                x=monthly["month_name"], y=monthly["mean"],
                name="Avg Demand", line=dict(color="#3f5efb", width=2.5), mode="lines+markers"
            ))
            fig_month.add_trace(go.Scatter(
                x=monthly["month_name"], y=monthly["max"],
                name="Max Demand", line=dict(color="#e74c3c", width=1.5, dash="dot")
            ))
            fig_month.update_layout(
                title="Monthly Passenger Demand Trend",
                xaxis_title="Month", yaxis_title="Passenger Count",
                 height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_month, use_container_width=True)

    # ── Tab 2: Route Analysis ──
    with tabs[1]:
        st.markdown("##### Route-wise Average Demand")
        if "route_name" in df.columns:
            route_demand = (
                df.groupby(["route_id", "route_name"])["passenger_count"]
                .mean().reset_index()
                .sort_values("passenger_count", ascending=True)
            )
            fig_route = px.bar(
                route_demand, x="passenger_count", y="route_name",
                orientation="h",
                color="passenger_count",
                color_continuous_scale="RdYlGn",
                labels={"passenger_count": "Avg Demand/hr", "route_name": "Route"},
                title="Average Passenger Demand by Route"
            )
            fig_route.update_layout(
                
                height=500, coloraxis_showscale=False
            )
            st.plotly_chart(fig_route, use_container_width=True)

        st.markdown("##### Peak vs Off-Peak Demand by Route")
        if "is_peak_hour" in df.columns and "route_name" in df.columns:
            peak_comp = df.groupby(["route_name", "is_peak_hour"])["passenger_count"].mean().reset_index()
            peak_comp["period"] = peak_comp["is_peak_hour"].map({1: "Peak", 0: "Off-Peak"})
            fig_peak = px.bar(
                peak_comp, x="route_name", y="passenger_count",
                color="period", barmode="group",
                color_discrete_map={"Peak": "#3f5efb", "Off-Peak": "#a0aec0"},
                labels={"passenger_count": "Avg Demand", "route_name": "Route"},
                title="Peak vs Off-Peak Demand by Route"
            )
            fig_peak.update_layout(
                
                xaxis_tickangle=-30, height=400
            )
            st.plotly_chart(fig_peak, use_container_width=True)

    # ── Tab 3: Weather & Events ──
    with tabs[2]:
        st.markdown("##### Demand by Weather Condition")
        if "weather_condition" in df.columns:
            weather_demand = df.groupby("weather_condition")["passenger_count"].mean().reset_index()
            weather_order  = ["Sunny","Partly Cloudy","Overcast","Drizzle","Heavy Rain","Thunderstorm"]
            weather_demand["weather_condition"] = pd.Categorical(
                weather_demand["weather_condition"], categories=weather_order, ordered=True
            )
            weather_demand = weather_demand.sort_values("weather_condition")
            fig_weather = px.bar(
                weather_demand, x="weather_condition", y="passenger_count",
                color="passenger_count",
                color_continuous_scale="RdYlGn_r",
                labels={"weather_condition": "Weather", "passenger_count": "Avg Demand"},
                title="Impact of Weather on Passenger Demand"
            )
            fig_weather.update_layout(
                
                coloraxis_showscale=False, height=350
            )
            st.plotly_chart(fig_weather, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Holiday vs Non-Holiday Demand")
            if "is_holiday" in df.columns:
                hol_avg = df.groupby("is_holiday")["passenger_count"].mean().reset_index()
                hol_avg["Type"] = hol_avg["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"})
                fig_hol = px.pie(
                    hol_avg, values="passenger_count", names="Type",
                    color_discrete_sequence=["#3f5efb", "#e74c3c"],
                    title="Avg Demand: Holiday vs Non-Holiday"
                )
                fig_hol.update_layout(height=320)
                st.plotly_chart(fig_hol, use_container_width=True)

        with col2:
            st.markdown("##### Event Impact on Demand")
            if "event_weight" in df.columns:
                df["has_event"] = (df["event_weight"] > 0).map({True: "Event Day", False: "No Event"})
                event_avg = df.groupby("has_event")["passenger_count"].mean().reset_index()
                fig_ev = px.bar(
                    event_avg, x="has_event", y="passenger_count",
                    color="has_event",
                    color_discrete_map={"Event Day": "#f39c12", "No Event": "#3f5efb"},
                    labels={"passenger_count": "Avg Demand", "has_event": ""},
                    title="Event vs Non-Event Day Demand"
                )
                fig_ev.update_layout(
                    
                    showlegend=False, height=320
                )
                st.plotly_chart(fig_ev, use_container_width=True)

        st.markdown("##### Temperature vs Passenger Demand")
        sample = df.sample(min(3000, len(df)))
        if "temperature_c" in df.columns:
            fig_temp = px.scatter(
                sample, x="temperature_c", y="passenger_count",
                color="weather_condition" if "weather_condition" in df.columns else None,
                opacity=0.5, 
                labels={"temperature_c": "Temperature (°C)", "passenger_count": "Passenger Count"},
                title="Temperature vs Passenger Demand (sample)"
            )
            fig_temp.update_layout(
                 height=380
            )
            st.plotly_chart(fig_temp, use_container_width=True)

    # ── Tab 4: Correlation ──
    with tabs[3]:
        st.markdown("##### Feature Correlation Matrix")
        corr_cols = [c for c in [
            "passenger_count", "hour", "day_of_week", "month",
            "is_weekend", "is_holiday", "event_weight",
            "temperature_c", "rainfall_mm", "weather_severity_score",
            "is_peak_hour", "lag_1h", "lag_2h", "lag_3h",
            "rolling_avg_3h", "rolling_avg_6h"
        ] if c in df.columns]

        if len(corr_cols) > 3:
            corr = df[corr_cols].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
                title="Pearson Correlation Matrix of Features"
            )
            fig_corr.update_layout(height=550)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Run feature_engineering.py to see full correlation matrix.")

        with st.expander("💡 Interpreting the Correlation Matrix"):
            st.markdown("""
            - **+1.0**: Perfect positive correlation (both increase together)
            - **-1.0**: Perfect negative correlation (one increases, other decreases)
            - **~0.0**: No linear relationship
            - **lag_1h / rolling_avg** columns typically have the highest correlation
              with `passenger_count` — confirming that recent demand is the best predictor of future demand.
            """)


# ═══════════════════════════════════════════════
# PAGE 4 — DEMAND PREDICTION
# ═══════════════════════════════════════════════
def page_prediction(df: pd.DataFrame):
    st.markdown("## 🔮 Demand Prediction", unsafe_allow_html=True)

    st.markdown("""
    Enter transit conditions below and the **Hybrid LSTM + Random Forest model**
    will predict the expected passenger demand for that hour.
    """)

    predictor = load_predictor()

    if predictor is None or not predictor._loaded:
        st.warning("""
        ⚠️ **Models not loaded.** Please train the models first:
        ```
        python src/train_lstm.py
        python src/train_random_forest.py
        ```
        """)

    # ── Input Form ──
    st.markdown("#### 🎛️ Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🕐 Time Parameters**")
        sel_hour = st.slider("Hour of Day", 0, 23, 8, help="0=Midnight, 8=Morning rush, 17=Evening rush")
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        sel_day   = st.selectbox("Day of Week", day_names, index=1)
        sel_dow   = day_names.index(sel_day)
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        sel_month_name = st.selectbox("Month", month_names, index=2)
        sel_month = month_names.index(sel_month_name) + 1

    with col2:
        st.markdown("**🌦️ Weather Conditions**")
        weather_options = ["Sunny","Partly Cloudy","Overcast","Drizzle","Heavy Rain","Thunderstorm"]
        sel_weather  = st.selectbox("Weather Condition", weather_options)
        sel_temp     = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, step=0.5)
        sel_rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0, step=0.5)

    with col3:
        st.markdown("**📅 Context**")
        sel_holiday = st.selectbox("Is Holiday?", ["No", "Yes"])
        is_holiday  = 1 if sel_holiday == "Yes" else 0

        route_options = sorted(df["route_id"].unique().tolist()) if not df.empty else [f"R{i:02d}" for i in range(1,16)]
        sel_route = st.selectbox("Route", route_options)

        sel_event_weight = st.slider(
            "Event Weight (0=No event, 1=Major event)", 0.0, 1.0, 0.0, step=0.05,
            help="Set >0 if there's a local event near this stop today"
        )

    # ── Derived info display ──
    is_weekend   = sel_dow >= 5
    is_peak      = sel_hour in {7,8,9,17,18,19}
    avg_est      = df[df["route_id"]==sel_route]["passenger_count"].mean() if not df.empty and sel_route in df["route_id"].values else 80.0

    info_cols = st.columns(4)
    info_cols[0].info(f"**Weekend:** {'Yes' if is_weekend else 'No'}")
    info_cols[1].info(f"**Peak Hour:** {'Yes ⚡' if is_peak else 'No'}")
    info_cols[2].info(f"**Avg Base Demand (Route):** {avg_est:.0f}")
    info_cols[3].info(f"**Season:** {'Monsoon' if sel_month in [6,7,8,9] else 'Summer' if sel_month in [3,4,5] else 'Winter'}")

    # ── Predict button ──
    if st.button("🚀 Predict Passenger Demand", use_container_width=True, type="primary"):
        if predictor is None or not predictor._loaded:
            st.error("Models not loaded. Please run training scripts first.")
        else:
            with st.spinner("Running hybrid prediction..."):
                result = predictor.predict(
                    hour=sel_hour,
                    day_of_week=sel_dow,
                    month=sel_month,
                    is_holiday=is_holiday,
                    event_weight=sel_event_weight,
                    temperature_c=sel_temp,
                    weather_condition=sel_weather,
                    rainfall_mm=sel_rainfall,
                    route_id=sel_route,
                    avg_demand_estimate=float(avg_est)
                )

            if "error" in result:
                st.error(result["error"])
            else:
                # ── Result display ──
                cat_colors = {"Low":"#27ae60","Medium":"#f39c12","High":"#e67e22","Very High":"#e74c3c"}
                cat = result["demand_category"]
                color = cat_colors.get(cat, "#3f5efb")

                st.success(f"**Predicted Passenger Demand:** {result['predicted_demand']} passengers / hour | Category: {cat}")

                # ── Model breakdown ──
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("🧠 LSTM Prediction",   f"{result['lstm_prediction']:.1f}")
                bc2.metric("🌳 RF Prediction",     f"{result['rf_prediction']:.1f}")
                bc3.metric("⚡ Hybrid (Final)",    f"{result['hybrid_prediction']:.1f}")

                st.caption(f"**Confidence:** {result['confidence_note']}")
                st.caption(
                    f"Formula: `{int(HYBRID_LSTM_WEIGHT*100)}% × {result['lstm_prediction']} "
                    f"+ {int(HYBRID_RF_WEIGHT*100)}% × {result['rf_prediction']} = {result['hybrid_prediction']}`"
                )

                # ── Gauge chart ──
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["predicted_demand"],
                    title={"text": "Demand Level", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 250]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0,   50],  "color": "#d4edda"},
                            {"range": [50,  120], "color": "#fff3cd"},
                            {"range": [120, 200], "color": "#fde8d8"},
                            {"range": [200, 250], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "darkred", "width": 3},
                            "thickness": 0.8,
                            "value": result["predicted_demand"]
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=40, r=40))
                st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Methodology explanation ──
    with st.expander("🔬 How does Hybrid Prediction work?"):
        st.markdown(f"""
        **Two-Model Architecture:**

        1. **LSTM (Long Short-Term Memory)** — Captures time-series patterns:
           - Receives 24 hours of historical demand as context
           - Learns morning/evening peak cycles, weekly patterns
           - Weight in hybrid: **{int(HYBRID_LSTM_WEIGHT*100)}%**

        2. **Random Forest** — Captures external factor relationships:
           - Features: weather, holidays, events, hour, day, lag values
           - Learns non-linear relationships between conditions and demand
           - Weight in hybrid: **{int(HYBRID_RF_WEIGHT*100)}%**

        **Hybrid Formula:**
        ```
        Final = (0.60 × LSTM_output) + (0.40 × RF_output)
        ```

        The weights were validated on the test set — LSTM gets higher weight
        because temporal patterns explain more variance in bus demand.
        """)


# ═══════════════════════════════════════════════
# PAGE 5 — GHOST HOTSPOTS
# ═══════════════════════════════════════════════
def page_hotspots(df: pd.DataFrame):
    st.markdown("## 🗺️ Ghost Hotspot Detection", unsafe_allow_html=True)

    st.markdown("""
    **Ghost Hotspots** are bus stops with **consistently high passenger demand**
    but served by only **one route** (low transit coverage).
    K-Means clustering (k=5) is used to identify and visualize these underserved areas.
    """)

    cluster_info = load_cluster_data()

    # ── Folium Map ──
    st.markdown("#### 🗺️ Interactive Map")
    if HOTSPOT_MAP_PATH.exists():
        from streamlit_folium import st_folium
        import folium
        # Load the saved map HTML
        with open(HOTSPOT_MAP_PATH, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=520, scrolling=False)
        st.caption("🔴 Red markers = Ghost Hotspots | Heatmap intensity = demand level | Click markers for stop details")
    else:
        st.warning("Map not generated yet. Run `python src/clustering.py` first.")

    # ── Cluster Summary Table ──
    st.markdown("#### 📋 Cluster Summary")
    if cluster_info:
        cluster_df = pd.DataFrame([{
            "Cluster": c["cluster_id"],
            "Type":    c["cluster_type"],
            "Stops":   c["n_stops"],
            "Avg Demand/hr": f"{c['avg_demand']:.0f}",
            "Avg Routes": f"{c['avg_routes']:.1f}",
            "Ghost Hotspots": c["ghost_hotspots"],
        } for c in cluster_info])
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

        # ── Ghost Hotspot recommendations ──
        ghost_clusters = [c for c in cluster_info if c["ghost_hotspots"] > 0]
        if ghost_clusters:
            st.markdown("#### ⚠️ Ghost Hotspot Stops")
            for c in ghost_clusters:
                st.markdown(f"""
                <div class='rec-card rec-card-high'>
                    <b>Cluster {c['cluster_id']} — {c['cluster_type']}</b><br>
                    {c['recommendation']}<br>
                    <small><b>Affected stops:</b> {', '.join(c['stop_names'])}</small>
                </div>
                """, unsafe_allow_html=True)

    # ── Methodology ──

def page_recommendations(df: pd.DataFrame):
    st.markdown("## 💡 Transit Recommendations", unsafe_allow_html=True)

    st.markdown("""
    Rule-based recommendations generated from **model predictions**, **clustering results**,
    and **data patterns**. These simulate prescriptive analytics for transit planners.
    """)

    cluster_info = load_cluster_data()
    metrics      = load_model_metrics()

    # ── Priority Recommendations ──
    st.markdown("#### 🔴 High Priority")
    high_recs = []

    if not df.empty and "passenger_count" in df.columns:
        # Find peak hour + route combinations with highest demand
        if "route_name" in df.columns and "hour" in df.columns:
            peak_df = df[df["hour"].isin([7,8,9,17,18,19])]
            if len(peak_df):
                top_route = peak_df.groupby("route_name")["passenger_count"].mean().idxmax()
                top_demand = peak_df.groupby("route_name")["passenger_count"].mean().max()
                high_recs.append(
                    f"🚌 **Increase frequency on '{top_route}'** during peak hours (7-9am, 5-7pm). "
                    f"Average peak demand: {top_demand:.0f} passengers/hr — highest among all routes."
                )

    # Add ghost hotspot recommendations
    ghost_clusters = [c for c in cluster_info if c["ghost_hotspots"] > 0]
    for gc in ghost_clusters:
        high_recs.append(
            f"📍 **Add new routes near Ghost Hotspot stops**: "
            f"{', '.join(gc['stop_names'][:3])}{'...' if len(gc['stop_names'])>3 else ''}. "
            f"Avg demand {gc['avg_demand']:.0f} pass/hr but only {gc['avg_routes']:.0f} route(s) serving them."
        )

    if not high_recs:
        high_recs.append("🔮 Run clustering and model training to generate specific high-priority recommendations.")

    for rec in high_recs:
        st.error(f"{rec}")

    # ── Medium Priority ──
    st.markdown("#### 🟡 Medium Priority")
    med_recs = []

    if not df.empty and "weather_condition" in df.columns:
        rain_demand = df[df["weather_condition"].isin(["Heavy Rain","Thunderstorm"])]["passenger_count"].mean()
        sun_demand  = df[df["weather_condition"] == "Sunny"]["passenger_count"].mean()
        if rain_demand < sun_demand:
            drop_pct = (sun_demand - rain_demand) / sun_demand * 100
            med_recs.append(
                f"🌧️ **Weather-Adaptive Scheduling**: Demand drops {drop_pct:.0f}% during heavy rain. "
                "Deploy smaller vehicles on low-demand routes during rainy hours to reduce fuel costs."
            )

    if not df.empty and "is_holiday" in df.columns:
        hol_demand  = df[df["is_holiday"]==1]["passenger_count"].mean()
        norm_demand = df[df["is_holiday"]==0]["passenger_count"].mean()
        if hol_demand < norm_demand:
            drop_pct = (norm_demand - hol_demand) / norm_demand * 100
            med_recs.append(
                f"📅 **Holiday Schedule Adjustment**: Demand reduces ~{drop_pct:.0f}% on public holidays. "
                "Reduce frequency by 30% and deploy surplus buses on event-day routes."
            )

    if not df.empty and "event_weight" in df.columns:
        event_demand = df[df["event_weight"]>0]["passenger_count"].mean()
        base_demand  = df[df["event_weight"]==0]["passenger_count"].mean()
        if event_demand > base_demand:
            boost_pct = (event_demand - base_demand) / base_demand * 100
            med_recs.append(
                f"🎪 **Event-Day Surge Planning**: Local events increase demand by ~{boost_pct:.0f}%. "
                "Pre-position 2-3 extra buses on routes to event-adjacent stops."
            )

    if not med_recs:
        med_recs.append("Monitor routes with steadily increasing demand. Conduct quarterly passenger surveys.")

    for rec in med_recs:
        st.warning(f"{rec}")

    # ── Low Priority ──
    st.markdown("#### 🟢 Routine / Long-Term")
    low_recs = [
        "📊 **Weekly demand monitoring**: Set up automated alerts when route demand exceeds 90th percentile for 3+ consecutive days.",
        "🛑 **Underutilised routes**: Routes with <30 average passengers/hour during off-peak can shift to on-demand mini-buses.",
        "🔄 **Model retraining**: Retrain LSTM and Random Forest models monthly as new ridership data becomes available.",
        "📱 **Passenger info system**: Integrate predictions into a real-time passenger app to redistribute demand across routes.",
        "♻️ **Eco-routing**: Prioritise electric buses on high-frequency routes to reduce emissions in dense urban zones.",
    ]
    for rec in low_recs:
        st.info(f"{rec}")

    # ── Summary table ──
    st.markdown("#### 📊 Route-wise Action Summary")
    if not df.empty and "route_name" in df.columns:
        route_summary = df.groupby(["route_id", "route_name"]).agg(
            avg_demand = ("passenger_count", "mean"),
            max_demand = ("passenger_count", "max"),
        ).reset_index().round(1)
        route_summary["Action"] = route_summary["avg_demand"].apply(
            lambda x: "🔴 Increase Frequency" if x > 90
                      else "🟡 Monitor" if x > 50
                      else "🟢 Adequate"
        )
        route_summary = route_summary.rename(columns={
            "route_id":"Route ID","route_name":"Route Name",
            "avg_demand":"Avg Demand/hr","max_demand":"Max Demand"
        })
        st.dataframe(route_summary[["Route ID","Route Name","Avg Demand/hr","Max Demand","Action"]],
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# MAIN APP ENTRY POINT
# ═══════════════════════════════════════════════
def main():
    # Load data
    with st.spinner("Loading data..."):
        df      = load_data()
        metrics = load_model_metrics()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to correct page
    if page == "Home":
        page_home(df, metrics)
    elif page == "Data Explorer":
        page_data_explorer(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Prediction":
        page_prediction(df)
    elif page == "Hotspots":
        page_hotspots(df)
    elif page == "Recommendations":
        page_recommendations(df)


if __name__ == "__main__":
    main()
