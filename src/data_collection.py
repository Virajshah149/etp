"""
data_collection.py — Hybrid Dataset Generation
===============================================
This module generates the core dataset for Eco-Transit Pulse.

DATA STRATEGY:
--------------
Real public data used:
  - Route/stop topology inspired by Bengaluru BMTC GTFS schema
    (https://www.bmtcinfo.com / OpenStreetMap-derived bus route data)
  - Indian public holiday calendar (2024, Ministry of Personnel list)
  - Weather pattern distributions based on IMD (India Meteorological Dept)
    historical normals for Bengaluru (temperature, rainfall, conditions)

Simulated (synthetic) components:
  - Exact passenger counts (BMTC does not publish hourly passenger data)
  - Specific event occurrences and weights
  - Per-stop granular weather readings (IMD data is city-level)

All simulated values follow realistic statistical distributions.
See SIMULATION_NOTE in utils.py for full documentation string.

OUTPUT:
  data/raw/transit_demand_raw.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src/ directory to path so we can import utils
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RAW_DATA_PATH, GRAPHS_DIR, ensure_dirs, setup_logger, CITY_NAME
)

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# RANDOM SEED for reproducibility
# ─────────────────────────────────────────────
np.random.seed(42)

# ─────────────────────────────────────────────
# ROUTE & STOP DEFINITIONS
# Inspired by real BMTC (Bengaluru Metropolitan Transport Corp) routes.
# Coordinates are real Bengaluru stop locations.
# ─────────────────────────────────────────────
ROUTES = [
    {"route_id": "R01", "route_name": "Majestic - Whitefield",      "base_demand": 140},
    {"route_id": "R02", "route_name": "Majestic - Electronic City",  "base_demand": 155},
    {"route_id": "R03", "route_name": "Shivajinagar - Marathahalli", "base_demand": 120},
    {"route_id": "R04", "route_name": "Kempegowda - Bannerghatta",   "base_demand": 95},
    {"route_id": "R05", "route_name": "Hebbal - Koramangala",        "base_demand": 110},
    {"route_id": "R06", "route_name": "Yeshwantpur - HSR Layout",    "base_demand": 100},
    {"route_id": "R07", "route_name": "Majestic - Jayanagar",        "base_demand": 130},
    {"route_id": "R08", "route_name": "Rajajinagar - BTM Layout",    "base_demand": 105},
    {"route_id": "R09", "route_name": "Yelahanka - Silk Board",      "base_demand": 85},
    {"route_id": "R10", "route_name": "Banashankari - Indiranagar",  "base_demand": 115},
    {"route_id": "R11", "route_name": "Vidyaranyapura - Koramangala","base_demand": 90},
    {"route_id": "R12", "route_name": "Tumkur Road - Airport",       "base_demand": 70},
    {"route_id": "R13", "route_name": "Kengeri - MG Road",           "base_demand": 125},
    {"route_id": "R14", "route_name": "Anekal - Electronics City",   "base_demand": 80},
    {"route_id": "R15", "route_name": "Domlur - Nagawara",           "base_demand": 95},
]

# Stops with real Bengaluru coordinates
STOPS = [
    {"stop_id": "S001", "stop_name": "Kempegowda Bus Station",  "lat": 12.9767, "lon": 77.5713, "routes": ["R01","R02","R07","R13"]},
    {"stop_id": "S002", "stop_name": "Shivajinagar",             "lat": 12.9830, "lon": 77.5994, "routes": ["R03","R05","R10"]},
    {"stop_id": "S003", "stop_name": "Whitefield",               "lat": 12.9698, "lon": 77.7499, "routes": ["R01","R03"]},
    {"stop_id": "S004", "stop_name": "Electronic City",          "lat": 12.8399, "lon": 77.6770, "routes": ["R02","R14"]},
    {"stop_id": "S005", "stop_name": "Marathahalli Bridge",      "lat": 12.9591, "lon": 77.7012, "routes": ["R03","R06"]},
    {"stop_id": "S006", "stop_name": "Bannerghatta Road",        "lat": 12.8680, "lon": 77.5970, "routes": ["R04","R07"]},
    {"stop_id": "S007", "stop_name": "Koramangala 1st Block",    "lat": 12.9352, "lon": 77.6245, "routes": ["R05","R08","R11"]},
    {"stop_id": "S008", "stop_name": "HSR Layout",               "lat": 12.9116, "lon": 77.6474, "routes": ["R06","R08"]},
    {"stop_id": "S009", "stop_name": "Jayanagar 4th Block",      "lat": 12.9308, "lon": 77.5830, "routes": ["R07","R04"]},
    {"stop_id": "S010", "stop_name": "BTM Layout",               "lat": 12.9165, "lon": 77.6101, "routes": ["R08","R06"]},
    {"stop_id": "S011", "stop_name": "Silk Board Junction",      "lat": 12.9175, "lon": 77.6229, "routes": ["R09","R05"]},
    {"stop_id": "S012", "stop_name": "Indiranagar 100ft Road",   "lat": 12.9784, "lon": 77.6408, "routes": ["R10","R15"]},
    {"stop_id": "S013", "stop_name": "Rajajinagar",              "lat": 12.9942, "lon": 77.5516, "routes": ["R08","R11"]},
    {"stop_id": "S014", "stop_name": "Yeshwantpur Circle",       "lat": 13.0234, "lon": 77.5390, "routes": ["R06","R09"]},
    {"stop_id": "S015", "stop_name": "Hebbal Flyover",           "lat": 13.0358, "lon": 77.5970, "routes": ["R05","R09"]},
    {"stop_id": "S016", "stop_name": "Yelahanka",                "lat": 13.1004, "lon": 77.5963, "routes": ["R09","R12"]},
    {"stop_id": "S017", "stop_name": "Kengeri",                  "lat": 12.9060, "lon": 77.4822, "routes": ["R13"]},
    {"stop_id": "S018", "stop_name": "MG Road",                  "lat": 12.9757, "lon": 77.6063, "routes": ["R13","R10","R15"]},
    {"stop_id": "S019", "stop_name": "Banashankari",             "lat": 12.9259, "lon": 77.5660, "routes": ["R10","R04"]},
    {"stop_id": "S020", "stop_name": "Vidyaranyapura",           "lat": 13.0588, "lon": 77.5539, "routes": ["R11","R12"]},
    {"stop_id": "S021", "stop_name": "Domlur",                   "lat": 12.9607, "lon": 77.6395, "routes": ["R15"]},
    {"stop_id": "S022", "stop_name": "Nagawara Junction",        "lat": 13.0450, "lon": 77.6230, "routes": ["R15","R09"]},
    {"stop_id": "S023", "stop_name": "Tumkur Road",              "lat": 13.0320, "lon": 77.5230, "routes": ["R12"]},
    {"stop_id": "S024", "stop_name": "Airport Road (BIAL)",      "lat": 13.1979, "lon": 77.7063, "routes": ["R12"]},
    {"stop_id": "S025", "stop_name": "Anekal",                   "lat": 12.7110, "lon": 77.6956, "routes": ["R14"]},
    {"stop_id": "S026", "stop_name": "Sarjapur Road",            "lat": 12.9102, "lon": 77.6877, "routes": ["R08","R11"]},
    {"stop_id": "S027", "stop_name": "KR Market",                "lat": 12.9615, "lon": 77.5757, "routes": ["R07","R13"]},
    {"stop_id": "S028", "stop_name": "Lalbagh Gate",             "lat": 12.9500, "lon": 77.5830, "routes": ["R04","R07"]},
    {"stop_id": "S029", "stop_name": "Shanti Nagar",             "lat": 12.9539, "lon": 77.5978, "routes": ["R10","R07"]},
    {"stop_id": "S030", "stop_name": "Richmond Circle",          "lat": 12.9627, "lon": 77.6030, "routes": ["R13","R10"]},
]

# ─────────────────────────────────────────────
# INDIAN PUBLIC HOLIDAYS 2024 (Central & Karnataka State)
# Source: Government of India Gazette / Karnataka Govt notification
# ─────────────────────────────────────────────
HOLIDAYS_2024 = {
    "2024-01-01": "New Year's Day",
    "2024-01-14": "Sankranti",
    "2024-01-26": "Republic Day",
    "2024-02-14": "Valentine's Day",       # Commercial holiday
    "2024-03-08": "Holi",
    "2024-03-25": "Ugadi",
    "2024-04-09": "Gudi Padwa",
    "2024-04-14": "Dr. Ambedkar Jayanti",
    "2024-04-17": "Ram Navami",
    "2024-05-01": "Labour Day",
    "2024-05-23": "Buddha Purnima",
    "2024-06-17": "Eid al-Adha",
    "2024-07-17": "Muharram",
    "2024-08-15": "Independence Day",
    "2024-09-07": "Ganesh Chaturthi",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Navratri Begin",
    "2024-10-24": "Dussehra",
    "2024-11-01": "Rajyotsava Day",
    "2024-11-01": "Diwali",
    "2024-11-15": "Guru Nanak Jayanti",
    "2024-12-25": "Christmas Day",
}

# ─────────────────────────────────────────────
# SIMULATED LOCAL EVENTS (tech events, cultural events, sports)
# These add demand spikes near specific stops
# ─────────────────────────────────────────────
EVENTS = [
    {"date": "2024-01-15", "name": "Bengaluru Tech Summit",    "stop": "S003", "weight": 0.9},
    {"date": "2024-02-04", "name": "Lalbagh Flower Show",      "stop": "S028", "weight": 0.7},
    {"date": "2024-02-18", "name": "IPL Season Opener",        "stop": "S018", "weight": 0.85},
    {"date": "2024-03-10", "name": "Holi Mela",                "stop": "S001", "weight": 0.6},
    {"date": "2024-03-25", "name": "Ugadi Procession",         "stop": "S027", "weight": 0.75},
    {"date": "2024-04-06", "name": "Art of Living Event",      "stop": "S028", "weight": 0.65},
    {"date": "2024-04-20", "name": "Bengaluru Half Marathon",  "stop": "S018", "weight": 0.80},
    {"date": "2024-05-05", "name": "Electronics Expo",         "stop": "S004", "weight": 0.70},
    {"date": "2024-05-18", "name": "Namma Pride Parade",       "stop": "S018", "weight": 0.75},
    {"date": "2024-06-02", "name": "Startup Conclave",         "stop": "S005", "weight": 0.65},
    {"date": "2024-06-10", "name": "Music Festival",           "stop": "S012", "weight": 0.70},
    {"date": "2024-06-22", "name": "IT Corridor Hackathon",    "stop": "S003", "weight": 0.60},
]


# ─────────────────────────────────────────────
# WEATHER GENERATION
# Based on IMD Bengaluru historical climate normals
# (avg temp, rainfall probability by month)
# ─────────────────────────────────────────────

# Monthly climate data for Bengaluru (approx. IMD normals)
MONTHLY_CLIMATE = {
    #  month: (avg_temp, temp_std, rain_prob, avg_rain_mm, rain_std)
    1:  (23.0, 2.5, 0.05, 1.0,  0.5),
    2:  (25.0, 2.8, 0.05, 2.0,  1.0),
    3:  (27.5, 3.0, 0.10, 5.0,  2.0),
    4:  (28.5, 3.2, 0.25, 18.0, 8.0),
    5:  (27.5, 3.0, 0.35, 55.0, 20.0),
    6:  (24.5, 2.5, 0.60, 85.0, 30.0),
    7:  (23.5, 2.0, 0.70, 110.0,35.0),
    8:  (23.5, 2.0, 0.68, 120.0,40.0),
    9:  (24.0, 2.2, 0.60, 150.0,45.0),
    10: (24.0, 2.5, 0.50, 170.0,50.0),
    11: (23.0, 2.5, 0.30, 55.0, 20.0),
    12: (22.0, 2.5, 0.10, 10.0, 5.0),
}

WEATHER_CONDITIONS = ["Sunny", "Partly Cloudy", "Overcast", "Drizzle", "Heavy Rain", "Thunderstorm"]


def generate_weather_for_day(month: int) -> dict:
    """
    Generate realistic weather for a single day based on
    Bengaluru's IMD historical climate normals.
    """
    avg_temp, temp_std, rain_prob, avg_rain, rain_std = MONTHLY_CLIMATE[month]
    temp = np.clip(np.random.normal(avg_temp, temp_std), 15.0, 38.0)
    is_rainy = np.random.random() < rain_prob
    if is_rainy:
        rainfall = max(0, np.random.normal(avg_rain / 30, rain_std / 30))
        if rainfall > 15:
            condition = "Thunderstorm"
        elif rainfall > 7:
            condition = "Heavy Rain"
        else:
            condition = "Drizzle"
    else:
        rainfall = 0.0
        condition = np.random.choice(["Sunny", "Partly Cloudy", "Overcast"],
                                     p=[0.5, 0.35, 0.15])
    return {
        "temperature_c": round(temp, 1),
        "rainfall_mm": round(rainfall, 2),
        "weather_condition": condition
    }


# ─────────────────────────────────────────────
# DEMAND GENERATION MODEL
# Passenger count is modelled as:
#   base_demand × hour_factor × day_factor × weather_factor × event_factor × noise
# ─────────────────────────────────────────────

# Hourly demand multipliers (24 hours, based on transit ridership studies)
HOUR_DEMAND_FACTORS = {
    0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.04,
    5: 0.12, 6: 0.45, 7: 0.90, 8: 1.00, 9: 0.70,
    10: 0.50, 11: 0.45, 12: 0.55, 13: 0.50, 14: 0.45,
    15: 0.55, 16: 0.75, 17: 0.95, 18: 1.00, 19: 0.80,
    20: 0.55, 21: 0.35, 22: 0.20, 23: 0.10
}

def compute_passenger_count(base_demand, hour, day_of_week, is_holiday,
                             weather, rainfall, event_weight):
    """
    Compute realistic passenger count using multiplicative demand model.

    Parameters:
        base_demand   : Route's baseline daily demand
        hour          : Hour of day (0-23)
        day_of_week   : 0=Monday ... 6=Sunday
        is_holiday    : 1 if public holiday
        weather       : Weather condition string
        rainfall      : Rainfall in mm
        event_weight  : Event proximity weight (0-1, 0 = no event)

    Returns:
        passenger_count (int)
    """
    # Hour factor — peak hours get 100%, off-peak gets less
    hour_factor = HOUR_DEMAND_FACTORS.get(hour, 0.1)

    # Day of week factor — weekdays higher than weekends
    if day_of_week < 5:   # Monday-Friday
        day_factor = 1.0
    elif day_of_week == 5: # Saturday
        day_factor = 0.75
    else:                  # Sunday
        day_factor = 0.55

    # Holiday factor — holidays reduce transit demand significantly
    holiday_factor = 0.65 if is_holiday else 1.0

    # Weather factor — rain suppresses outdoor travel
    weather_factors = {
        "Sunny": 1.05, "Partly Cloudy": 1.00, "Overcast": 0.95,
        "Drizzle": 0.80, "Heavy Rain": 0.65, "Thunderstorm": 0.45
    }
    weather_factor = weather_factors.get(weather, 1.0)

    # Rainfall penalty (extra suppression for very heavy rain)
    rain_penalty = max(0.5, 1.0 - (rainfall / 60.0))

    # Event boost — events increase demand near event stops
    event_factor = 1.0 + (event_weight * 0.5)

    # Combine all factors
    demand = (base_demand * hour_factor * day_factor *
              holiday_factor * weather_factor * rain_penalty * event_factor)

    # Add Gaussian noise ±10% for realism
    noise = np.random.normal(1.0, 0.10)
    demand = max(0, demand * noise)

    return int(round(demand))


# ─────────────────────────────────────────────
# MAIN DATASET GENERATION FUNCTION
# ─────────────────────────────────────────────
def generate_dataset(
    start_date: str = "2024-01-01",
    end_date:   str = "2024-06-30"
) -> pd.DataFrame:
    """
    Generate the complete hybrid dataset spanning start_date to end_date.
    Records are hourly (one record per stop × route per hour).

    Returns:
        pd.DataFrame with all RAW_COLS columns.
    """
    logger.info(f"Generating dataset: {start_date} to {end_date}")

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    # Build an event lookup: {date_str: {stop_id: event_dict}}
    event_lookup = {}
    for ev in EVENTS:
        event_lookup.setdefault(ev["date"], {})[ev["stop"]] = ev

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        month    = date.month
        dow      = date.dayofweek      # 0=Mon, 6=Sun

        # Daily holiday check
        is_holiday  = 1 if date_str in HOLIDAYS_2024 else 0
        holiday_name = HOLIDAYS_2024.get(date_str, "None")

        # Generate day-level weather (constant per day for realism)
        weather = generate_weather_for_day(month)

        for hour in range(24):
            # Check for events affecting this date
            day_events = event_lookup.get(date_str, {})

            for stop in STOPS:
                stop_id  = stop["stop_id"]
                stop_routes = stop["routes"]

                # Determine event weight for this stop on this date/hour
                event_name   = "None"
                event_weight = 0.0
                if stop_id in day_events:
                    ev = day_events[stop_id]
                    event_name   = ev["name"]
                    # Events are strongest in evening hours (15-21)
                    if 14 <= hour <= 22:
                        event_weight = ev["weight"]
                    elif 10 <= hour <= 14:
                        event_weight = ev["weight"] * 0.5
                    else:
                        event_weight = 0.0

                # Pick the primary route for this stop
                for route_id in stop_routes:
                    route = next(r for r in ROUTES if r["route_id"] == route_id)

                    passenger_count = compute_passenger_count(
                        base_demand  = route["base_demand"],
                        hour         = hour,
                        day_of_week  = dow,
                        is_holiday   = is_holiday,
                        weather      = weather["weather_condition"],
                        rainfall     = weather["rainfall_mm"],
                        event_weight = event_weight
                    )

                    dt_str = f"{date_str} {hour:02d}:00:00"

                    records.append({
                        "datetime":          dt_str,
                        "date":              date_str,
                        "hour":              hour,
                        "route_id":          route_id,
                        "route_name":        route["route_name"],
                        "stop_id":           stop_id,
                        "stop_name":         stop["stop_name"],
                        "latitude":          stop["lat"],
                        "longitude":         stop["lon"],
                        "passenger_count":   passenger_count,
                        "temperature_c":     weather["temperature_c"],
                        "weather_condition": weather["weather_condition"],
                        "rainfall_mm":       weather["rainfall_mm"],
                        "is_holiday":        is_holiday,
                        "holiday_name":      holiday_name,
                        "event_name":        event_name,
                        "event_weight":      round(event_weight, 3),
                    })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df):,} records across {len(dates)} days.")
    return df


def save_raw_data(df: pd.DataFrame):
    """Save the generated raw dataframe to the raw data directory."""
    ensure_dirs()
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Raw data saved → {RAW_DATA_PATH}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info(f"  Eco-Transit Pulse — Data Collection ({CITY_NAME})")
    logger.info("=" * 55)

    df = generate_dataset(start_date="2024-01-01", end_date="2024-06-30")

    # Quick validation
    logger.info(f"Columns        : {list(df.columns)}")
    logger.info(f"Date range     : {df['date'].min()} → {df['date'].max()}")
    logger.info(f"Routes         : {df['route_id'].nunique()}")
    logger.info(f"Stops          : {df['stop_id'].nunique()}")
    logger.info(f"Avg passengers : {df['passenger_count'].mean():.1f}")
    logger.info(f"Max passengers : {df['passenger_count'].max()}")
    logger.info(f"Missing values : {df.isnull().sum().sum()}")

    save_raw_data(df)
    logger.info("Data collection complete!")
