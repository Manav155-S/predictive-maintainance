import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Project", layout="centered")

st.title("🛠️ Predictive Maintenance & Anomaly Detection")
st.markdown("This unified dashboard presents all modules of the project in one place.")

# Section 1 - Power BI
st.header("📊 Power BI Dashboard")
st.markdown("**Visual Insights of Machine Data**")
st.image("power_bi.png", caption="Power BI Dashboard Screenshot")
st.markdown("[Download Full Power BI PDF](https://drive.google.com/your_link)")

# Section 2 - Anomaly Detection with R
st.header("📉 Anomaly Detection (R Z-Score Method)")
st.image("zscore_plot.png", caption="Z-Score based anomaly detection (R)")
st.markdown("[GitHub Actions - Auto Run](https://github.com/your_user/your_repo/actions)")

# Section 3 - TinyML via Edge Impulse
st.header("🤖 TinyML Integration (Edge Impulse)")
st.image("edgeimpulse_result.png", caption="Edge Impulse K-means Clustering Output")
st.markdown("[Open Edge Impulse Project](https://studio.edgeimpulse.com/studio/your_project_id)")

# Section 4 - PySpark Evaluation
st.header("⚙️ PySpark ML Evaluation")
st.image("confusion_matrix.png", caption="Confusion Matrix from PySpark Model")
st.markdown("""
- **Accuracy**: 92.5%  
- **F1-Score**: 0.89  
- **Algorithm Used**: Logistic Regression
""")

# Section 5 - Streamlit Real-Time Anomaly Detection App
st.header("📍 Live Anomaly Detection App")
st.markdown("[Click to Open Anomaly Detector](https://your-streamlit-app-url.streamlit.app)")

# Section 6 - DevOps Automation with GitHub Actions
st.header("🔁 CI/CD Automation (DevOps)")
st.markdown("Every R script runs automatically on every push using GitHub Actions. It installs packages, runs detection, and generates logs.")

# Section 7 - Download Final Report
st.header("📄 Project Report")
with open("Predictive_Maintenance_Report.pdf", "rb") as pdf_file:
    st.download_button("Download Final Report (PDF)", pdf_file, file_name="Predictive_Maintenance_Report.pdf")

# Footer
st.markdown("---")
st.caption("Developed by: Manav, Ashish & Chirag | NSUT CSE | 2025")
