import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta

# Configure page settings
st.set_page_config(layout="wide", page_title="Predictive Maintenance and Anomaly Detection")

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>Predictive Maintenance and Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Real-Time Industrial Monitoring Dashboard</h4>", unsafe_allow_html=True)

# Sidebar with controls
st.sidebar.header("Control Panel")

# Initialize session state for simulation status and data
if 'running' not in st.session_state:
    st.session_state.running = False
    
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Time", "Temperature", "Pressure", "Vibration", "RPM", "Oil_Level"])

if 'maintenance_history' not in st.session_state:
    st.session_state.maintenance_history = []
    
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []
    
if 'machine_health' not in st.session_state:
    st.session_state.machine_health = 100  # Start with perfect health
    
if 'next_maintenance' not in st.session_state:
    # Random future date for next scheduled maintenance
    st.session_state.next_maintenance = datetime.now() + timedelta(days=random.randint(10, 30))

# Start/Stop Simulation Button
status_placeholder = st.empty()

if st.sidebar.button("Start Monitoring" if not st.session_state.running else "Stop Monitoring"):
    st.session_state.running = not st.session_state.running
    
    if st.session_state.running:
        status_placeholder.markdown(f"<h3>✅ Real-Time Monitoring Active</h3>", unsafe_allow_html=True)
    else:
        status_placeholder.markdown(f"<h3>⏹️ Monitoring Paused</h3>", unsafe_allow_html=True)

# Show current status on initial load
if st.session_state.running:
    status_placeholder.markdown(f"<h3>✅ Real-Time Monitoring Active</h3>", unsafe_allow_html=True)
else:
    status_placeholder.markdown(f"<h3>⏹️ Monitoring Paused</h3>", unsafe_allow_html=True)

# Add machine selection in sidebar
st.sidebar.subheader("Machine Selection")
machine_type = st.sidebar.selectbox(
    "Machine Type", 
    ["Industrial Pump", "CNC Machine", "HVAC System", "Turbine Generator"]
)

# Add simulation parameters in sidebar
st.sidebar.subheader("Monitoring Parameters")
update_interval = st.sidebar.slider("Update Interval (seconds)", 0.5, 5.0, 1.0, 0.1)
max_data_points = st.sidebar.slider("Max Data Points to Display", 10, 100, 30)

# Anomaly injection controls
st.sidebar.subheader("Simulation Controls")
anomaly_frequency = st.sidebar.slider("Anomaly Frequency", 0.0, 1.0, 0.1, 0.05,
                                   help="Higher values increase the chance of anomalies")

if st.sidebar.button("Force Anomaly"):
    st.session_state.force_anomaly = True
else:
    if 'force_anomaly' not in st.session_state:
        st.session_state.force_anomaly = False

# Sensor threshold controls
st.sidebar.subheader("Alert Thresholds")
temp_threshold = st.sidebar.slider("Temperature Alert (°C)", 70, 110, 90)
pressure_threshold = st.sidebar.slider("Pressure Alert (Bar)", 50, 100, 70)
vibration_threshold = st.sidebar.slider("Vibration Alert (mm/s)", 2.0, 8.0, 4.0)

# Placeholders for charts and alerts
chart_placeholder = st.empty()
health_placeholder = st.empty()
alerts_placeholder = st.empty()
maintenance_placeholder = st.empty()
anomaly_placeholder = st.empty()

# Helper functions for anomaly detection and prediction
def detect_anomalies(data, current_values, thresholds):
    """Detect anomalies in sensor data"""
    anomalies = []
    
    # Simple threshold-based anomalies
    if current_values["Temperature"] > thresholds["temp"]:
        anomalies.append({
            "type": "High Temperature",
            "value": current_values["Temperature"],
            "threshold": thresholds["temp"],
            "severity": "High" if current_values["Temperature"] > thresholds["temp"] + 5 else "Medium",
            "time": current_values["Time"],
            "impact": "Potential overheating damage, reduced efficiency"
        })
    
    if current_values["Pressure"] > thresholds["pressure"]:
        anomalies.append({
            "type": "High Pressure",
            "value": current_values["Pressure"],
            "threshold": thresholds["pressure"],
            "severity": "High" if current_values["Pressure"] > thresholds["pressure"] + 10 else "Medium",
            "time": current_values["Time"],
            "impact": "Risk of seal failure, pipe rupture"
        })
    
    if current_values["Vibration"] > thresholds["vibration"]:
        anomalies.append({
            "type": "Excessive Vibration",
            "value": current_values["Vibration"],
            "threshold": thresholds["vibration"],
            "severity": "High" if current_values["Vibration"] > thresholds["vibration"] + 1 else "Medium",
            "time": current_values["Time"],
            "impact": "Bearing wear, component misalignment"
        })
    
    # Pattern-based anomalies (if we have enough data points)
    if len(data) > 5:
        # Detect rapid vibration increase (potential bearing failure)
        if len(data) >= 3:
            vib_values = data["Vibration"].tail(3).values
            if vib_values[2] > vib_values[0] * 1.5 and vib_values[2] > vib_values[1] * 1.25:
                anomalies.append({
                    "type": "Rapid Vibration Increase",
                    "value": vib_values[2],
                    "threshold": "Trend",
                    "severity": "High",
                    "time": current_values["Time"],
                    "impact": "Potential bearing failure imminent"
                })
        
        # Detect temperature and pressure correlation anomaly
        if len(data) >= 5:
            temp_trend = data["Temperature"].tail(5).values
            pres_trend = data["Pressure"].tail(5).values
            if np.std(temp_trend) > 5 and np.std(pres_trend) < 1:
                anomalies.append({
                    "type": "Temperature-Pressure Decorrelation",
                    "value": f"Temp Var: {np.std(temp_trend):.2f}, Press Var: {np.std(pres_trend):.2f}",
                    "threshold": "Pattern",
                    "severity": "Medium",
                    "time": current_values["Time"],
                    "impact": "Possible sensor failure or fluid leakage"
                })
    
    # Forced anomaly for demonstration
    if st.session_state.force_anomaly:
        anomaly_type = random.choice(["Bearing Failure", "Seal Leakage", "Motor Overload", "Coolant Contamination"])
        anomalies.append({
            "type": anomaly_type,
            "value": "Simulated",
            "threshold": "Forced",
            "severity": "Critical",
            "time": current_values["Time"],
            "impact": "Immediate maintenance required to prevent failure"
        })
        st.session_state.force_anomaly = False
    
    return anomalies

def predict_maintenance(data, machine_health, anomalies):
    """Predict maintenance needs based on data and detected anomalies"""
    maintenance_prediction = {
        "needed": False,
        "urgency": "Low",
        "reason": "",
        "estimated_time": None,
        "recommended_actions": []
    }
    
    # Determine if maintenance is needed based on anomalies
    if anomalies:
        max_severity = max([a["severity"] for a in anomalies], key=lambda x: {"Medium": 1, "High": 2, "Critical": 3}.get(x, 0))
        if max_severity == "Critical":
            maintenance_prediction["needed"] = True
            maintenance_prediction["urgency"] = "Immediate"
            maintenance_prediction["reason"] = f"Critical {anomalies[0]['type']} detected"
            maintenance_prediction["estimated_time"] = datetime.now() + timedelta(hours=random.randint(1, 4))
            maintenance_prediction["recommended_actions"] = [
                f"Inspect {anomalies[0]['type'].lower()} components",
                "Prepare replacement parts",
                "Schedule emergency maintenance"
            ]
        elif max_severity == "High":
            maintenance_prediction["needed"] = True
            maintenance_prediction["urgency"] = "High"
            maintenance_prediction["reason"] = f"High severity {anomalies[0]['type']} detected"
            maintenance_prediction["estimated_time"] = datetime.now() + timedelta(days=random.randint(1, 3))
            maintenance_prediction["recommended_actions"] = [
                "Monitor closely",
                f"Inspect {anomalies[0]['type'].lower()} components",
                "Schedule maintenance within 48 hours"
            ]
    
    # Consider machine health for maintenance prediction
    if machine_health < 40:
        maintenance_prediction["needed"] = True
        maintenance_prediction["urgency"] = "High"
        maintenance_prediction["reason"] = f"Poor machine health ({machine_health}%)"
        maintenance_prediction["estimated_time"] = datetime.now() + timedelta(days=random.randint(1, 2))
        maintenance_prediction["recommended_actions"] = [
            "Complete system inspection",
            "Preventive component replacement",
            "Lubrication and calibration"
        ]
    elif machine_health < 60:
        maintenance_prediction["needed"] = True
        maintenance_prediction["urgency"] = "Medium"
        maintenance_prediction["reason"] = f"Degrading machine health ({machine_health}%)"
        maintenance_prediction["estimated_time"] = datetime.now() + timedelta(days=random.randint(3, 7))
        maintenance_prediction["recommended_actions"] = [
            "Inspect wear components",
            "Check calibration",
            "Schedule preventive maintenance"
        ]
    
    return maintenance_prediction

def update_machine_health(current_health, anomalies):
    """Update machine health based on anomalies and random degradation"""
    # Natural degradation
    health_change = -random.uniform(0.1, 0.5)
    
    # Impact from anomalies
    if anomalies:
        for anomaly in anomalies:
            if anomaly["severity"] == "Critical":
                health_change -= random.uniform(5.0, 10.0)
            elif anomaly["severity"] == "High":
                health_change -= random.uniform(2.0, 5.0)
            elif anomaly["severity"] == "Medium":
                health_change -= random.uniform(1.0, 2.0)
    
    # Calculate new health value
    new_health = max(0, min(100, current_health + health_change))
    return new_health

# Main simulation loop
def run_simulation():
    # Keep data within size limit
    if len(st.session_state.data) > max_data_points:
        st.session_state.data = st.session_state.data.iloc[-max_data_points:]
    
    # Base sensor patterns with randomness
    if len(st.session_state.data) > 0:
        last_temp = st.session_state.data["Temperature"].iloc[-1]
        last_pressure = st.session_state.data["Pressure"].iloc[-1]
        last_vibration = st.session_state.data["Vibration"].iloc[-1]
        last_rpm = st.session_state.data["RPM"].iloc[-1] if "RPM" in st.session_state.data else 1750
        last_oil = st.session_state.data["Oil_Level"].iloc[-1] if "Oil_Level" in st.session_state.data else 95
    else:
        last_temp = 70
        last_pressure = 50
        last_vibration = 2.0
        last_rpm = 1750
        last_oil = 95
    
    # Simulate sensor readings with realistic patterns
    temperature = max(50, min(120, last_temp + random.uniform(-2, 2)))
    pressure = max(20, min(90, last_pressure + random.uniform(-3, 3)))
    vibration = max(0.1, min(8.0, last_vibration + random.uniform(-0.3, 0.3)))
    rpm = max(1000, min(2000, last_rpm + random.uniform(-50, 50)))
    oil_level = max(70, min(100, last_oil - random.uniform(0.0, 0.2)))  # Oil level slowly decreases
    
    # Introduce anomalies randomly
    if random.random() < anomaly_frequency:
        # Choose a random anomaly type
        anomaly_type = random.choice(["temp_spike", "pressure_surge", "vibration_spike", "rpm_drop"])
        
        if anomaly_type == "temp_spike":
            temperature += random.uniform(10, 25)
        elif anomaly_type == "pressure_surge":
            pressure += random.uniform(15, 30)
        elif anomaly_type == "vibration_spike":
            vibration += random.uniform(2, 4)
        elif anomaly_type == "rpm_drop":
            rpm -= random.uniform(200, 500)
    
    current_time = pd.Timestamp.now().strftime("%H:%M:%S")
    
    # Add new data point
    new_data = pd.DataFrame([[current_time, temperature, pressure, vibration, rpm, oil_level]], 
                             columns=["Time", "Temperature", "Pressure", "Vibration", "RPM", "Oil_Level"])
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
    
    # Current sensor values for anomaly detection
    current_values = {
        "Time": current_time,
        "Temperature": temperature,
        "Pressure": pressure,
        "Vibration": vibration,
        "RPM": rpm,
        "Oil_Level": oil_level
    }
    
    # Detect anomalies
    thresholds = {
        "temp": temp_threshold, 
        "pressure": pressure_threshold, 
        "vibration": vibration_threshold
    }
    
    anomalies = detect_anomalies(st.session_state.data, current_values, thresholds)
    
    # Update machine health
    st.session_state.machine_health = update_machine_health(st.session_state.machine_health, anomalies)
    
    # Make maintenance predictions
    maintenance_prediction = predict_maintenance(st.session_state.data, st.session_state.machine_health, anomalies)
    
    # Store anomalies in history
    for anomaly in anomalies:
        if len(st.session_state.anomaly_history) >= 10:  # Keep only 10 most recent anomalies
            st.session_state.anomaly_history.pop(0)
        st.session_state.anomaly_history.append(anomaly)
    
    # Store maintenance predictions in history if needed
    if maintenance_prediction["needed"]:
        if len(st.session_state.maintenance_history) >= 5:  # Keep only 5 most recent predictions
            st.session_state.maintenance_history.pop(0)
        st.session_state.maintenance_history.append(maintenance_prediction)
    
    # Display charts
    with chart_placeholder.container():
        st.subheader(f"Real-Time Sensor Readings - {machine_type}")
        
        # Create 2 rows of charts
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        # Top row: Temperature and Vibration (main indicators)
        with row1_col1:
            fig_temp = px.line(st.session_state.data, x="Time", y="Temperature", 
                               title="Temperature (°C)", markers=True)
            fig_temp.add_hline(y=temp_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with row1_col2:
            fig_vib = px.line(st.session_state.data, x="Time", y="Vibration", 
                              title="Vibration (mm/s)", markers=True)
            fig_vib.add_hline(y=vibration_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig_vib, use_container_width=True)
        
        # Bottom row: Pressure, RPM, Oil Level
        with row2_col1:
            fig_pres = px.line(st.session_state.data, x="Time", y="Pressure", 
                               title="Pressure (Bar)", markers=True)
            fig_pres.add_hline(y=pressure_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pres, use_container_width=True)
            
        with row2_col2:
            fig_rpm = px.line(st.session_state.data, x="Time", y="RPM", 
                              title="RPM", markers=True)
            st.plotly_chart(fig_rpm, use_container_width=True)
            
        with row2_col3:
            fig_oil = px.line(st.session_state.data, x="Time", y="Oil_Level", 
                              title="Oil Level (%)", markers=True)
            fig_oil.add_hline(y=80, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_oil, use_container_width=True)
    
    # Show machine health gauge
    with health_placeholder.container():
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Create a gauge chart for machine health
            health_color = "green" if st.session_state.machine_health > 70 else "orange" if st.session_state.machine_health > 40 else "red"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.machine_health,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Machine Health Index", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': health_color},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 40
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Show next scheduled maintenance
            next_maint_date = st.session_state.next_maintenance.strftime("%b %d, %Y")
            
            st.markdown("### Machine Status")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"**Machine Type:** {machine_type}")
                st.markdown(f"**Current Status:** {'Online' if st.session_state.running else 'Paused'}")
                
                # Determine operational status
                if st.session_state.machine_health > 70:
                    status = "✅ Normal Operation"
                elif st.session_state.machine_health > 40:
                    status = "⚠️ Caution - Degraded Performance"
                else:
                    status = "🚨 Critical - Immediate Attention Required"
                st.markdown(f"**Operational Status:** {status}")
                
            with col_b:
                st.markdown(f"**Next Scheduled Maintenance:** {next_maint_date}")
                
                # Remaining useful life based on health
                remaining_days = max(0, int((st.session_state.machine_health / 100) * 30))
                st.markdown(f"**Estimated Remaining Useful Life:** {remaining_days} days")
                
                if maintenance_prediction["needed"]:
                    st.markdown(f"**Maintenance Urgency:** 🚨 **{maintenance_prediction['urgency']}**")
                else:
                    st.markdown("**Maintenance Urgency:** ✅ No urgent maintenance needed")
    
    # Check for alerts
    alerts = []
    for anomaly in anomalies:
        alert_emoji = "🔥" if "Temperature" in anomaly["type"] else "⚠️" if "Pressure" in anomaly["type"] else "🚨"
        alerts.append(f"{alert_emoji} {anomaly['severity']} {anomaly['type']}: {anomaly['value']}")
    
    with alerts_placeholder.container():
        if alerts:
            st.error(" | ".join(alerts))
        else:
            st.success("All systems operating within normal parameters")
    
    # Display maintenance predictions and recommendations
    with maintenance_placeholder.container():
        st.subheader("Predictive Maintenance Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Maintenance Recommendations")
            
            if maintenance_prediction["needed"]:
                st.markdown(f"**Reason:** {maintenance_prediction['reason']}")
                st.markdown(f"**Urgency:** {maintenance_prediction['urgency']}")
                
                if maintenance_prediction["estimated_time"]:
                    est_time = maintenance_prediction["estimated_time"].strftime("%b %d, %Y")
                    st.markdown(f"**Recommended Before:** {est_time}")
                
                st.markdown("**Recommended Actions:**")
                for action in maintenance_prediction["recommended_actions"]:
                    st.markdown(f"- {action}")
            else:
                st.markdown("✅ No maintenance actions required at this time")
                st.markdown("Continue regular monitoring and scheduled maintenance")
        
        with col2:
            st.markdown("#### Recent Anomaly History")
            if st.session_state.anomaly_history:
                # Take only the 5 most recent anomalies
                recent_anomalies = st.session_state.anomaly_history[-5:]
                for idx, anomaly in enumerate(reversed(recent_anomalies)):
                    severity_color = "red" if anomaly["severity"] == "Critical" else "orange" if anomaly["severity"] == "High" else "blue"
                    st.markdown(f"**{anomaly['time']} - {anomaly['type']}**")
                    st.markdown(f"<span style='color:{severity_color}'>Severity: {anomaly['severity']}</span>", unsafe_allow_html=True)
                    st.markdown(f"Impact: {anomaly['impact']}")
                    if idx < len(recent_anomalies) - 1:
                        st.markdown("---")
            else:
                st.markdown("No anomalies detected in recent history")
    
    # Display detailed anomaly analysis
    with anomaly_placeholder.container():
        if anomalies:
            st.subheader("Real-Time Anomaly Analysis")
            
            for anomaly in anomalies:
                severity_color = "red" if anomaly["severity"] == "Critical" else "orange" if anomaly["severity"] == "High" else "blue"
                
                st.markdown(f"### {anomaly['type']} Detected")
                st.markdown(f"<span style='color:{severity_color}; font-size:18px'>Severity: {anomaly['severity']}</span>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Detection Time:** {anomaly['time']}")
                    st.markdown(f"**Current Value:** {anomaly['value']}")
                    st.markdown(f"**Threshold:** {anomaly['threshold']}")
                    
                with col2:
                    st.markdown("**Potential Impact:**")
                    st.markdown(anomaly['impact'])
                    
                    # Add component-specific analysis
                    if "Temperature" in anomaly["type"]:
                        st.markdown("**Affected Components:** Bearings, Motor, Cooling System")
                    elif "Pressure" in anomaly["type"]:
                        st.markdown("**Affected Components:** Seals, Pipes, Valves")
                    elif "Vibration" in anomaly["type"]:
                        st.markdown("**Affected Components:** Shaft, Bearings, Mounting")
                    else:
                        st.markdown("**Affected Components:** Multiple Systems")

# Use the correct rerun function for recent Streamlit versions
if st.session_state.running:
    run_simulation()
    time.sleep(update_interval)
    st.rerun()  # For Streamlit 1.10.0 and newer
else:
    # Display static charts if not running
    if not st.session_state.data.empty:
        run_simulation()  # Just update the display without data generation
