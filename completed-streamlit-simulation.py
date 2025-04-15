import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import random

st.set_page_config(layout="wide")

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>Industrial Hardware Simulation</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Real-Time Monitoring Dashboard</h4>", unsafe_allow_html=True)

# Sidebar with controls
st.sidebar.header("Control Panel")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False

# Start/Stop Simulation Button
status_placeholder = st.empty()
if st.sidebar.button("Start Simulation" if not st.session_state.running else "Stop Simulation"):
    st.session_state.running = not st.session_state.running
    if st.session_state.running:
        status_placeholder.markdown(f"<h3>✅ Simulation Running</h3>", unsafe_allow_html=True)
    else:
        status_placeholder.markdown(f"<h3>⏹️ Simulation Stopped</h3>", unsafe_allow_html=True)

# Placeholders for charts and alerts
placeholder = st.empty()
alerts_placeholder = st.empty()

# Initialize data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Time", "Temperature", "Pressure", "Vibration"])

# Main simulation loop
while st.session_state.running:
    # Simulate sensor readings
    temperature = random.uniform(60, 100)
    pressure = random.uniform(20, 80)
    vibration = random.uniform(0.1, 5.0)

    current_time = pd.Timestamp.now().strftime("%H:%M:%S")

    new_data = pd.DataFrame([[current_time, temperature, pressure, vibration]], columns=["Time", "Temperature", "Pressure", "Vibration"])
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

    # Display charts
    with placeholder.container():
        st.subheader("Real-Time Sensor Readings")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig_temp = px.line(st.session_state.data, x="Time", y="Temperature", title="Temperature (°C)", markers=True)
            st.plotly_chart(fig_temp, use_container_width=True)

        with col2:
            fig_pres = px.line(st.session_state.data, x="Time", y="Pressure", title="Pressure (Bar)", markers=True)
            st.plotly_chart(fig_pres, use_container_width=True)

        with col3:
            fig_vib = px.line(st.session_state.data, x="Time", y="Vibration", title="Vibration (mm/s)", markers=True)
            st.plotly_chart(fig_vib, use_container_width=True)

    # Check for alerts
    alerts = []
    if temperature > 90:
        alerts.append(f"🔥 High Temperature Alert: {temperature:.2f}°C")
    if pressure > 70:
        alerts.append(f"⚠️ High Pressure Alert: {pressure:.2f} Bar")
    if vibration > 4.0:
        alerts.append(f"🚨 High Vibration Alert: {vibration:.2f} mm/s")

    with alerts_placeholder.container():
        if alerts:
            st.error(" | ".join(alerts))
        else:
            st.success("All systems normal.")

    time.sleep(1)
