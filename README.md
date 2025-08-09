# Predictive Maintenance Dashboard using Machine Learning

![Dashboard Screenshot](https://i.imgur.com/8aZ3X1h.png)

This project is an end-to-end web application that demonstrates how machine learning can be used to predict industrial equipment failures before they happen. The system analyzes real-time sensor data, predicts the probability of failure, and provides an interactive dashboard for forensic analysis of historical machine data.

The core goal is to shift from expensive, **reactive maintenance** (fixing things after they break) to cost-effective, **proactive maintenance** (fixing things before they break).

---

## 🚀 Key Features

* **Live Predictive Model:** Utilizes a trained `RandomForestClassifier` to provide real-time failure probability scores based on sensor inputs.
* **Forensic Replay Mode:** The standout feature of this application. Select any machine by its unique ID (`UDI`) from the historical dataset and replay its entire operational history. Watch as the sensor data evolves and see the model's prediction change, providing a powerful validation of the system's effectiveness on real-world data.
* **Interactive "What-If" Analysis:** Use manual sliders to input specific sensor values and see the model's prediction instantly, allowing engineers to test hypothetical scenarios.
* **Model Explainability:** Includes a "Feature Importance" chart that shows which sensor parameters (e.g., Torque, Temperature) are most influential in the model's predictions, aiding in root cause analysis.
* **Dynamic & Responsive UI:** A clean and intuitive user interface built with Streamlit that visualizes complex data through interactive Plotly charts.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn (`RandomForestClassifier`)
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas
* **Data Visualization:** Plotly Express
* **Model Persistence:** Joblib

---

## ⚙️ How It Works

The project follows a standard two-phase machine learning workflow:

1.  **Offline Training:** The system first checks if a trained model file (`rf_model.joblib`) exists. If not, it uses the historical `predictive_maintenance.csv` dataset to train a Random Forest model. The trained model and the dataset are then saved to disk using `joblib`. This ensures the computationally expensive training process is only done once.

2.  **Online Deployment & Inference:** The Streamlit application loads the pre-trained model from the saved file. It then uses this lightweight, loaded model to perform fast, real-time predictions on new data, whether from the manual sliders or the forensic replay simulation.

---
