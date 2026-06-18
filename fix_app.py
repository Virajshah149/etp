import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove CUSTOM_CSS completely
content = re.sub(r'CUSTOM_CSS\s*=\s*\"\"\"[\s\S]*?\"\"\"', 'CUSTOM_CSS = ""', content)
content = content.replace('st.markdown(CUSTOM_CSS, unsafe_allow_html=True)', '')

# 2. Simplify render_sidebar
old_sidebar = """def render_sidebar():
    with st.sidebar:
        st.markdown(\"\"\"
        <div style='text-align:center; padding: 16px 0 8px 0;'>
            <div style='font-size:2.5rem'>🚌</div>
            <div style='font-size:1.1rem; font-weight:700; color:#e8f4fd; letter-spacing:0.03em;'>
                Eco-Transit Pulse
            </div>
            <div style='font-size:0.75rem; color:#a0b4c8; margin-top:4px;'>
                Urban Mobility Optimizer
            </div>
        </div>
        <hr style='border-color:#2d4a6b; margin: 8px 0 16px 0;'>
        \"\"\", unsafe_allow_html=True)

        pages = {
            "🏠  Home":               "Home",
            "📂  Data Explorer":       "Data Explorer",
            "📊  Exploratory Analysis":"EDA",
            "🔮  Demand Prediction":   "Prediction",
            "🗺️   Ghost Hotspots":     "Hotspots",
            "💡  Recommendations":     "Recommendations",
        }

        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(\"\"\"
        <div style='font-size:0.73rem; color:#7a96b0; padding: 8px 0;'>
            <b>Technology Stack</b><br>
            Python • Streamlit • TensorFlow<br>
            Scikit-learn • Plotly • Folium
        </div>
        <div style='font-size:0.73rem; color:#7a96b0; padding: 8px 0; margin-top: 10px;'>
            <b>Models</b><br>
            LSTM (60%) + Random Forest (40%)
        </div>
        \"\"\", unsafe_allow_html=True)
        return pages[selected]"""

new_sidebar = """def render_sidebar():
    with st.sidebar:
        st.title("🚌 Eco-Transit Pulse")
        st.write("Urban Mobility Optimizer")
        
        pages = {
            "🏠 Home": "Home",
            "📂 Data Explorer": "Data Explorer",
            "📊 Exploratory Analysis": "EDA",
            "🔮 Demand Prediction": "Prediction",
            "🗺️ Ghost Hotspots": "Hotspots",
            "💡 Recommendations": "Recommendations",
        }
        
        selected = st.radio("Navigate", list(pages.keys()))
        
        st.write("---")
        st.write("**Tech Stack:** Python, Streamlit, TensorFlow, Scikit-learn")
        st.write("**Models:** LSTM + Random Forest")
        return pages[selected]"""

content = content.replace(old_sidebar, new_sidebar)

# 3. Replace kpi_card definition and usage
# We will just change kpi_card usages to st.metric and delete the function.
# Or just change the kpi_card function to not be used.
content = re.sub(
    r'c1\.markdown\(kpi_card\(".*?", str\(n_routes\), "Total Routes"\), unsafe_allow_html=True\)',
    'c1.metric("Total Routes", n_routes)',
    content
)
content = re.sub(
    r'c2\.markdown\(kpi_card\(".*?", str\(n_stops\), "Bus Stops"\), unsafe_allow_html=True\)',
    'c2.metric("Bus Stops", n_stops)',
    content
)
content = re.sub(
    r'c3\.markdown\(kpi_card\(".*?", avg_demand, "Avg Demand / hr"\), unsafe_allow_html=True\)',
    'c3.metric("Avg Demand / hr", avg_demand)',
    content
)
content = re.sub(
    r'c4\.markdown\(kpi_card\(".*?", total_records, "Total Records"\), unsafe_allow_html=True\)',
    'c4.metric("Total Records", total_records)',
    content
)
content = re.sub(
    r'c5\.markdown\(kpi_card\(".*?", date_range, "Date Range"\), unsafe_allow_html=True\)',
    'c5.metric("Date Range", date_range)',
    content
)

# 4. Remove section header styles
content = re.sub(r'<h2 class=\'section-header\'>(.*?)</h2>', r'## \1', content)

# 5. Fix trendline="ols" issue in page_eda
content = content.replace('trendline="ols",', '')

# 6. Replace recommendation HTML with st.error / st.warning / st.info
content = re.sub(
    r'st\.markdown\(f\"<div class=\'rec-card rec-card-high\'>(.*?)</div>\", unsafe_allow_html=True\)',
    r'st.error(f"\1")',
    content
)
content = re.sub(
    r'st\.markdown\(f\"<div class=\'rec-card rec-card-medium\'>(.*?)</div>\", unsafe_allow_html=True\)',
    r'st.warning(f"\1")',
    content
)
content = re.sub(
    r'st\.markdown\(f\"<div class=\'rec-card rec-card-low\'>(.*?)</div>\", unsafe_allow_html=True\)',
    r'st.info(f"\1")',
    content
)

# 7. Replace ghost hotspot text wrapper
content = re.sub(
    r'st\.markdown\(f\"\"\"\s*<div style=\'.*?\'>\s*<b style=\'.*?\'>Cluster {c} — {c_type}</b><br>\s*<div style=\'.*?\'>\s*🚨 {c_type} Zone: Monitor {n_stops} stop\(s\)\. Consider express services during peak hours\.\s*</div>\s*<div style=\'.*?\'>\s*Affected stops: {stop_names}\s*</div>\s*</div>\s*\"\"\", unsafe_allow_html=True\)',
    r'st.error(f"**Cluster {c} - {c_type}**\\n\\n🚨 Monitor {n_stops} stop(s). Consider express services.\\n\\nAffected stops: {stop_names}")',
    content
)
content = re.sub(
    r'st\.markdown\(f\"\"\"\s*<div style=\'.*?\'>\s*<b style=\'.*?\'>Cluster {c} — {c_type}</b><br>\s*<div style=\'.*?\'>\s*🟡 {c_type} Zone: Monitor {n_stops} stop\(s\)\. Consider express services during peak hours\.\s*</div>\s*<div style=\'.*?\'>\s*Affected stops: {stop_names}\s*</div>\s*</div>\s*\"\"\", unsafe_allow_html=True\)',
    r'st.warning(f"**Cluster {c} - {c_type}**\\n\\n🟡 Monitor {n_stops} stop(s). Consider express services.\\n\\nAffected stops: {stop_names}")',
    content
)

# 8. Replace prediction output box
old_pred_box = """st.markdown(f\"\"\"
                <div class='pred-result-box'>
                    <div style='font-size:1.1rem; opacity:0.9'>Predicted Hourly Demand</div>
                    <div class='pred-number'>{res['predicted_demand']} <span style='font-size:1.5rem'>passengers</span></div>
                    <div class='pred-label'>Demand Level: <b>{res['demand_category']}</b></div>
                </div>
                \"\"\", unsafe_allow_html=True)"""
new_pred_box = """st.success(f"**Predicted Hourly Demand:** {res['predicted_demand']} passengers \\n\\n**Demand Level:** {res['demand_category']}")"""
content = content.replace(old_pred_box, new_pred_box)

# 9. Remove Data Note wrapper
content = re.sub(
    r'st\.markdown\(f\"\"\"\s*<div class=\'data-note\'>\s*📌 \*\*Dataset Note:\*\* .*?\s*</div>\s*\"\"\", unsafe_allow_html=True\)',
    r'st.info("📌 **Dataset Note:** Dataset is a hybrid of real GTFS structural schema and realistically simulated demand, weather, and event data. Route topology is inspired by Bengaluru\'s BMTC bus network. Passenger counts, weather, and events are synthetically generated using statistically realistic distributions and validated patterns.")',
    content
)

# 10. Also replace template colours in plots for standard look
content = content.replace('color_discrete_sequence=["#3f5efb"]', 'color_discrete_sequence=px.colors.qualitative.Plotly')
content = content.replace('color_discrete_sequence=["#2ea44f"]', 'color_discrete_sequence=px.colors.qualitative.Plotly')

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated dashboard/app.py successfully!")
