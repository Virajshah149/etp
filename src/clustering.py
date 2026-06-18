"""
clustering.py — K-Means Ghost Hotspot Detection
================================================
This module identifies "Ghost Hotspots" — areas in the city with
consistently high passenger demand but insufficient transit coverage.

Methodology:
  1. Aggregate average passenger demand per stop (lat, lon)
  2. Apply K-Means clustering (k=5, selected via elbow method)
  3. Classify clusters: High-demand clusters with few routes = Ghost Hotspots
  4. Generate Folium interactive map with color-coded markers
  5. Produce rule-based transit recommendations per cluster

Ghost Hotspot Definition:
  A stop is classified as a Ghost Hotspot if:
    - It belongs to a high-demand cluster (top 2 clusters by avg demand)
    - It is served by only 1 route (low coverage)

OUTPUT:
  models/kmeans_model.pkl
  outputs/heatmaps/ghost_hotspots_map.html
  outputs/reports/cluster_info.json
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    FEATURED_DATA_PATH, KMEANS_MODEL_PATH,
    HOTSPOT_MAP_PATH, CLUSTER_INFO_PATH,
    GRAPHS_DIR, REPORTS_DIR,
    KMEANS_N_CLUSTERS, KMEANS_RANDOM_STATE,
    CITY_CENTER, ensure_dirs, setup_logger,
    classify_demand, save_cluster_info
)

logger = setup_logger(__name__)

# Cluster color mapping for Folium markers
CLUSTER_COLORS = {
    0: "blue",
    1: "green",
    2: "orange",
    3: "darkred",
    4: "purple",
}

# Icon for ghost hotspot markers
GHOST_HOTSPOT_ICON = "exclamation-sign"


def aggregate_stop_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the featured dataset to one record per stop.

    Computes:
      - avg_demand    : Mean passenger count across all hours
      - peak_demand   : 90th percentile demand (captures peak load)
      - route_count   : Number of unique routes serving this stop
      - latitude/longitude: Stop coordinates

    Returns:
        DataFrame with one row per stop, aggregated metrics
    """
    logger.info("Aggregating passenger demand per stop...")

    stop_agg = df.groupby(["stop_id", "stop_name", "latitude", "longitude"]).agg(
        avg_demand   = ("passenger_count", "mean"),
        peak_demand  = ("passenger_count", lambda x: x.quantile(0.90)),
        route_count  = ("route_id", "nunique"),
        total_records= ("passenger_count", "count")
    ).reset_index()

    stop_agg["avg_demand"]  = stop_agg["avg_demand"].round(2)
    stop_agg["peak_demand"] = stop_agg["peak_demand"].round(2)

    logger.info(f"Aggregated {len(stop_agg)} unique stops")
    logger.info(f"Avg demand range: {stop_agg['avg_demand'].min():.1f} – {stop_agg['avg_demand'].max():.1f}")
    return stop_agg


def run_kmeans(stop_df: pd.DataFrame, n_clusters=KMEANS_N_CLUSTERS):
    """
    Apply K-Means clustering on [latitude, longitude, avg_demand].

    Features are standardised before clustering so that geographic
    coordinates and demand values are on the same scale.

    Elbow method was run offline; k=5 gave optimal inertia drop.

    Returns:
        stop_df with 'cluster' column added, fitted KMeans model, scaler
    """
    logger.info(f"Running K-Means with k={n_clusters}...")

    # Features: lat, lon, demand (standardised)
    cluster_features = stop_df[["latitude", "longitude", "avg_demand"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,                # Multiple initialisations for stability
        max_iter=500,
        random_state=KMEANS_RANDOM_STATE
    )
    stop_df["cluster"] = kmeans.fit_predict(features_scaled)

    logger.info(f"Cluster distribution:\n{stop_df['cluster'].value_counts().sort_index()}")
    return stop_df, kmeans, scaler


def identify_ghost_hotspots(stop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Ghost Hotspot stops.

    A stop is a Ghost Hotspot if:
      1. avg_demand >= 75th percentile of all stops (high demand)
      2. route_count == 1 (only one route serves it — low coverage)

    These are candidate areas for transit expansion.
    """
    demand_threshold = stop_df["avg_demand"].quantile(0.75)
    coverage_threshold = 1  # Only 1 route = underserved

    stop_df["is_ghost_hotspot"] = (
        (stop_df["avg_demand"] >= demand_threshold) &
        (stop_df["route_count"] <= coverage_threshold)
    ).astype(int)

    n_hotspots = stop_df["is_ghost_hotspot"].sum()
    logger.info(f"Demand threshold (75th pct): {demand_threshold:.1f}")
    logger.info(f"Ghost Hotspots identified  : {n_hotspots}")
    return stop_df


def generate_cluster_info(stop_df: pd.DataFrame) -> list:
    """
    Summarise each cluster for display in the dashboard.

    Returns:
        List of dicts with cluster-level statistics and recommendations.
    """
    cluster_summaries = []

    for cluster_id in sorted(stop_df["cluster"].unique()):
        cluster_stops = stop_df[stop_df["cluster"] == cluster_id]
        ghost_count   = cluster_stops["is_ghost_hotspot"].sum()
        avg_demand    = cluster_stops["avg_demand"].mean()
        avg_routes    = cluster_stops["route_count"].mean()
        n_stops       = len(cluster_stops)

        # Determine cluster category based on demand level
        demand_label  = classify_demand(avg_demand)

        # Generate recommendation based on cluster profile
        if ghost_count > 0 and avg_demand >= 60:
            recommendation = (
                f"⚠️ Ghost Hotspot Cluster: {ghost_count} underserved stop(s) with "
                f"high demand (~{avg_demand:.0f} passengers/hour). "
                "Recommend adding new routes or increasing frequency."
            )
            cluster_type = "Ghost Hotspot"
        elif avg_demand >= 80:
            recommendation = (
                f"🔴 High Demand Zone: {n_stops} stop(s) with avg {avg_demand:.0f} "
                "passengers/hour. Recommend increasing bus frequency during peak hours."
            )
            cluster_type = "High Demand"
        elif avg_demand >= 40:
            recommendation = (
                f"🟡 Moderate Demand Zone: Monitor {n_stops} stop(s). "
                "Consider express services during peak hours."
            )
            cluster_type = "Moderate Demand"
        else:
            recommendation = (
                f"🟢 Low Demand Zone: {n_stops} stop(s) with avg {avg_demand:.0f} "
                "passengers/hour. Current coverage appears adequate."
            )
            cluster_type = "Low Demand"

        cluster_summaries.append({
            "cluster_id":       int(cluster_id),
            "cluster_type":     cluster_type,
            "n_stops":          int(n_stops),
            "avg_demand":       round(float(avg_demand), 2),
            "avg_routes":       round(float(avg_routes), 2),
            "ghost_hotspots":   int(ghost_count),
            "demand_label":     demand_label,
            "recommendation":   recommendation,
            "stop_names":       cluster_stops["stop_name"].tolist()
        })

    logger.info(f"Generated info for {len(cluster_summaries)} clusters")
    return cluster_summaries


def build_folium_map(stop_df: pd.DataFrame, df_full: pd.DataFrame) -> folium.Map:
    """
    Build an interactive Folium map showing:
      - Heatmap layer of passenger demand
      - Cluster-colored stop markers
      - Ghost Hotspot markers with warning icons
      - Info popups for each stop

    Args:
        stop_df  : Aggregated stop data with cluster labels
        df_full  : Full featured dataset (for heatmap)

    Returns:
        folium.Map object
    """
    logger.info("Building interactive Folium map...")

    # ── Base map centred on Bengaluru ──
    fmap = folium.Map(
        location=CITY_CENTER,
        zoom_start=12,
        tiles="CartoDB positron"
    )

    # ── Heatmap layer ──
    heat_data = [
        [row["latitude"], row["longitude"], row["avg_demand"]]
        for _, row in stop_df.iterrows()
    ]
    HeatMap(
        heat_data,
        radius=25, blur=18, max_zoom=14,
        gradient={"0.2": "blue", "0.5": "yellow", "0.8": "red"}
    ).add_to(fmap)

    # ── Cluster stop markers ──
    for _, stop in stop_df.iterrows():
        cluster_id = int(stop["cluster"])
        is_ghost   = bool(stop["is_ghost_hotspot"])

        # Color and icon
        color = "red" if is_ghost else CLUSTER_COLORS.get(cluster_id, "gray")
        icon  = GHOST_HOTSPOT_ICON if is_ghost else "bus"

        # Popup HTML
        popup_html = f"""
        <div style='font-family:sans-serif; font-size:13px; min-width:200px'>
            <b style='color:#333'>{stop['stop_name']}</b><br>
            <hr style='margin:4px 0'>
            <b>Stop ID:</b> {stop['stop_id']}<br>
            <b>Avg Demand:</b> {stop['avg_demand']:.0f} passengers/hr<br>
            <b>Peak Demand:</b> {stop['peak_demand']:.0f} passengers/hr<br>
            <b>Routes Served:</b> {int(stop['route_count'])}<br>
            <b>Cluster:</b> {cluster_id}<br>
            {'<br><b style="color:red">⚠️ GHOST HOTSPOT</b>' if is_ghost else ''}
        </div>
        """

        folium.Marker(
            location=[stop["latitude"], stop["longitude"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{stop['stop_name']} | {stop['avg_demand']:.0f} pass/hr",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")
        ).add_to(fmap)

    # ── Legend ──
    legend_html = """
    <div style='position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 18px; border-radius:8px;
                border:1px solid #ccc; font-family:sans-serif; font-size:13px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.2)'>
        <b>🗺️ Eco-Transit Pulse — Map Legend</b><br><br>
        <span style='color:red'>●</span> Ghost Hotspot (High demand, low coverage)<br>
        <span style='color:blue'>●</span> Cluster 0<br>
        <span style='color:green'>●</span> Cluster 1<br>
        <span style='color:orange'>●</span> Cluster 2<br>
        <span style='color:darkred'>●</span> Cluster 3<br>
        <span style='color:purple'>●</span> Cluster 4<br>
        <br>Heatmap: 🔵 Low → 🟡 Mid → 🔴 High demand
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


def plot_elbow_curve(stop_df: pd.DataFrame):
    """
    Generate and save the elbow curve used to choose k=5 for K-Means.
    Demonstrates the methodology for model selection.
    """
    features_scaled = StandardScaler().fit_transform(
        stop_df[["latitude", "longitude", "avg_demand"]].values
    )

    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=KMEANS_RANDOM_STATE)
        km.fit(features_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.axvline(x=KMEANS_N_CLUSTERS, color="#FF5722", linestyle="--",
               linewidth=1.5, label=f"Chosen k={KMEANS_N_CLUSTERS}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (Within-cluster Sum of Squares)", fontsize=12)
    ax.set_title("K-Means Elbow Curve — Ghost Hotspot Detection", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path = GRAPHS_DIR / "kmeans_elbow_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Elbow curve saved → {save_path}")


def run_clustering(featured_path=FEATURED_DATA_PATH) -> pd.DataFrame:
    """
    Execute the complete clustering pipeline.

    Returns:
        stop_df: Aggregated stop-level DataFrame with cluster labels
    """
    logger.info("Starting K-Means Ghost Hotspot detection...")

    df = pd.read_csv(featured_path, parse_dates=["datetime"])

    stop_df = aggregate_stop_demand(df)
    stop_df, kmeans, scaler = run_kmeans(stop_df)
    stop_df = identify_ghost_hotspots(stop_df)

    # Save KMeans model
    ensure_dirs()
    joblib.dump({"model": kmeans, "scaler": scaler}, KMEANS_MODEL_PATH)
    logger.info(f"K-Means model saved → {KMEANS_MODEL_PATH}")

    # Generate and save cluster info
    cluster_info = generate_cluster_info(stop_df)
    save_cluster_info(cluster_info)
    logger.info(f"Cluster info saved → {CLUSTER_INFO_PATH}")

    # Build and save Folium map
    fmap = build_folium_map(stop_df, df)
    fmap.save(str(HOTSPOT_MAP_PATH))
    logger.info(f"Folium map saved → {HOTSPOT_MAP_PATH}")

    # Save elbow curve
    plot_elbow_curve(stop_df)

    return stop_df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Eco-Transit Pulse — K-Means Clustering")
    logger.info("=" * 55)

    stop_df = run_clustering()

    print("\n--- Cluster Summary ---")
    summary = stop_df.groupby("cluster").agg(
        n_stops       = ("stop_id", "count"),
        avg_demand    = ("avg_demand", "mean"),
        ghost_hotspots= ("is_ghost_hotspot", "sum")
    ).round(2)
    print(summary)

    n_hotspots = stop_df["is_ghost_hotspot"].sum()
    print(f"\nTotal Ghost Hotspots identified: {n_hotspots}")
    print(f"Map saved to: {HOTSPOT_MAP_PATH}")
