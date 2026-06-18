import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove plot_bgcolor and paper_bgcolor from update_layout
content = re.sub(r'plot_bgcolor="white",\s*paper_bgcolor="white",?', '', content)

# 2. Fix the remaining custom HTML boxes that we missed
# data-note
old_data_note = """st.markdown(f\"\"\"
    <div class='data-note'>
        📌 <b>Dataset Note:</b> {SIMULATION_NOTE}
    </div>
    \"\"\", unsafe_allow_html=True)"""
content = content.replace(old_data_note, 'st.info(f"📌 **Dataset Note:** {SIMULATION_NOTE}")')

# pred-result-box
old_pred_box = """st.markdown(f\"\"\"
                <div class='pred-result-box' style='background:linear-gradient(135deg, {color}cc 0%, {color} 100%)'>
                    <div class='pred-label'>Predicted Passenger Demand</div>
                    <div class='pred-number'>{result['predicted_demand']}</div>
                    <div class='pred-label'>passengers / hour &nbsp;|&nbsp; Category: <b>{cat}</b></div>
                </div>
                \"\"\", unsafe_allow_html=True)"""
new_pred_box = """st.success(f"**Predicted Passenger Demand:** {result['predicted_demand']} passengers / hour | Category: {cat}")"""
content = content.replace(old_pred_box, new_pred_box)

# rec-card-high
old_rec_high = """st.markdown(f\"\"\"
                <div class='rec-card rec-card-high'>
                    <b>Cluster {c['cluster_id']} - {c['cluster_type']}</b><br>
                    {c['recommendation']}<br>
                    <small><b>Affected stops:</b> {', '.join(c['stop_names'])}</small>
                </div>
                \"\"\", unsafe_allow_html=True)"""
new_rec_high = """st.error(f"**Cluster {c['cluster_id']} - {c['cluster_type']}**\\n\\n{c['recommendation']}\\n\\nAffected stops: {', '.join(c['stop_names'])}")"""
content = content.replace(old_rec_high, new_rec_high)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Final UI fixes applied successfully!")
