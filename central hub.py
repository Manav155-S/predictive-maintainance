import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Project", layout="wide")

st.title("🛠️ Predictive Maintenance & Anomaly Detection Project")
st.markdown("### 💻 NSUT | Computer Hardware & Software Workshop")

# SECTION 1: Power BI
st.header("📊 Power BI Visualizations")
st.markdown("Uploaded `.pbix` file for interactive visualizations.")
st.markdown("[Download Power BI File](https://github.com/Manav155-S/predictive-maintainance/raw/main/Visualizations.pbix)")

# SECTION 2: Anomaly Detection with R
st.header("📉 Anomaly Detection (Z-Score in R)")
st.image("anomaly_plot.png", caption="Z-Score Anomaly Detection using R", use_column_width=True)
st.markdown("[View GitHub Actions Run](https://github.com/Manav155-S/predictive-maintainance/actions)")

# SECTION 3: Edge Impulse - TinyML Simulation
st.header("🤖 Edge Impulse Integration (TinyML Simulation)")
st.image("edgeimpulse_result.png", caption="Anomaly clusters from Edge Impulse (K-Means)")
st.markdown("Used Edge Impulse Studio to simulate sensor data behavior and detect anomalies.")

# SECTION 4: Real-Time Simulation (Streamlit)
st.header("⚙️ Streamlit Real-Time Anomaly Simulation")
st.markdown("This simulation mimics anomaly-based predictions using simplified logic.")
st.markdown("[Open Simulation App](https://your-streamlit-url.streamlit.app)")  # Replace with actual

# SECTION 5: ML Evaluation (PySpark)
st.header("📈 Machine Learning Evaluation using PySpark")
st.image("model_metrics.png", caption="Confusion Matrix (PySpark Logistic Regression)")
st.markdown("""
**Model Info:**
- Accuracy: `93.5%`
- F1 Score: `0.91`
- Algorithm: Logistic Regression
""")

# SECTION 6: DevOps CI/CD
st.header("🔁 DevOps: GitHub Actions Automation")
st.markdown("""
- GitHub Action triggers `.R` script automatically on every push to `main`.
- Ensures reproducibility and deployment readiness.
""")

# SECTION 7: Final Report
st.header("📄 Project Report (PDF)")
with open("Predictive_Maintenance_Report.pdf", "rb") as pdf:
    st.download_button("Download Final Project Report", pdf, file_name="Predictive_Maintenance_Report.pdf")

st.markdown("---")
st.caption("Made with 💙 by Manav, Ashish & Chirag | NSUT CSE")
