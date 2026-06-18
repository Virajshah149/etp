import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the Hybrid Prediction Strategy from the Model Performance Summary
# We'll change st.columns(3) to st.columns(2) and remove the mc3 block
content = content.replace("mc1, mc2, mc3 = st.columns(3)", "mc1, mc2 = st.columns(2)")

hybrid_block = """        with mc3:
            st.markdown("**Hybrid Prediction**")
            st.markdown(f\"\"\"
            <div style='background:#eef1ff;border-radius:10px;padding:14px 16px;margin-top:4px;'>
                <div style='font-size:0.85rem;color:#5a6474;'>Strategy</div>
                <div style='font-weight:600;margin:4px 0;'>Weighted Average</div>
                <div style='font-size:0.85rem;color:#5a6474;'>LSTM Weight</div>
                <div style='font-weight:600;margin:4px 0;'>{int(HYBRID_LSTM_WEIGHT*100)}%</div>
                <div style='font-size:0.85rem;color:#5a6474;'>RF Weight</div>
                <div style='font-weight:600;margin:4px 0;'>{int(HYBRID_RF_WEIGHT*100)}%</div>
            </div>
            \"\"\", unsafe_allow_html=True)"""

content = content.replace(hybrid_block, "")

# Remove the Dataset Note
dataset_note_block = """    # ── Data Note ──
    st.info(f"📌 **Dataset Note:** {SIMULATION_NOTE}")"""

content = content.replace(dataset_note_block, "")

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Home page updated successfully!")
