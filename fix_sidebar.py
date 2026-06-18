import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_sidebar = """def render_sidebar():
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

new_sidebar = """from streamlit_option_menu import option_menu

def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🚌 Eco-Transit Pulse</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray; margin-top: 0;'>Urban Mobility Optimizer</p>", unsafe_allow_html=True)
        st.write("---")
        
        selected_page = option_menu(
            menu_title=None,
            options=["Home", "Data Explorer", "Exploratory Analysis", "Demand Prediction", "Ghost Hotspots", "Recommendations"],
            icons=["house", "folder", "bar-chart", "graph-up", "map", "lightbulb"],
            default_index=0,
        )
        
        st.write("---")
        st.write("**Tech Stack:** Python, Streamlit, TensorFlow, Scikit-learn")
        st.write("**Models:** LSTM + Random Forest")
        
        page_mapping = {
            "Home": "Home",
            "Data Explorer": "Data Explorer",
            "Exploratory Analysis": "EDA",
            "Demand Prediction": "Prediction",
            "Ghost Hotspots": "Hotspots",
            "Recommendations": "Recommendations"
        }
        return page_mapping[selected_page]"""

content = content.replace(old_sidebar, new_sidebar)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Sidebar updated successfully!")
