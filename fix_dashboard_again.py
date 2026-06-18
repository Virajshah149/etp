import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "def render_sidebar():" in line:
        new_lines.append("from streamlit_option_menu import option_menu\n")
        new_lines.append("def render_sidebar():\n")
        new_lines.append("    with st.sidebar:\n")
        new_lines.append("        st.markdown(\"<h2 style='text-align: center; margin-bottom: 0;'>🚌 Eco-Transit Pulse</h2>\", unsafe_allow_html=True)\n")
        new_lines.append("        st.markdown(\"<p style='text-align: center; color: gray; margin-top: 0;'>Urban Mobility Optimizer</p>\", unsafe_allow_html=True)\n")
        new_lines.append("        st.write(\"---\")\n")
        new_lines.append("        \n")
        new_lines.append("        selected_page = option_menu(\n")
        new_lines.append("            menu_title=None,\n")
        new_lines.append("            options=[\"Home\", \"Data Explorer\", \"Exploratory Analysis\", \"Demand Prediction\", \"Ghost Hotspots\", \"Recommendations\"],\n")
        new_lines.append("            icons=[\"house\", \"folder\", \"bar-chart\", \"graph-up\", \"map\", \"lightbulb\"],\n")
        new_lines.append("            default_index=0,\n")
        new_lines.append("        )\n")
        new_lines.append("        \n")
        new_lines.append("        st.write(\"---\")\n")
        new_lines.append("        st.write(\"**Tech Stack:** Python, Streamlit, TensorFlow, Scikit-learn\")\n")
        new_lines.append("        st.write(\"**Models:** LSTM + Random Forest\")\n")
        new_lines.append("        \n")
        new_lines.append("        page_mapping = {\n")
        new_lines.append("            \"Home\": \"Home\",\n")
        new_lines.append("            \"Data Explorer\": \"Data Explorer\",\n")
        new_lines.append("            \"Exploratory Analysis\": \"EDA\",\n")
        new_lines.append("            \"Demand Prediction\": \"Prediction\",\n")
        new_lines.append("            \"Ghost Hotspots\": \"Hotspots\",\n")
        new_lines.append("            \"Recommendations\": \"Recommendations\"\n")
        new_lines.append("        }\n")
        new_lines.append("        return page_mapping[selected_page]\n")
        skip = True
        continue
    
    if skip and "def kpi_card" in line:
        skip = False
        new_lines.append("\n") # Add it back
        new_lines.append(line)
        continue
    
    if "Project Workflow" in line and "with st.expander" in line:
        skip = True
        continue
    
    if skip and "def page_data_explorer" in line:
        skip = False
        new_lines.append("\n")
        new_lines.append(line)
        continue
        
    if "K-Means Methodology" in line and "with st.expander" in line:
        skip = True
        continue
    
    if skip and "def page_recommendations" in line:
        skip = False
        new_lines.append("\n")
        new_lines.append(line)
        continue

    if not skip:
        if "trendline=\"lowess\"" in line:
            new_lines.append(line.replace("trendline=\"lowess\",", ""))
        else:
            new_lines.append(line)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Changes applied successfully!")
