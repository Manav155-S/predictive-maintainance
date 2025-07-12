import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta

# Configure page settings
st.set_page_config(layout="wide", page_title="Maintenance Dashboard", page_icon="🔧")

# Custom CSS for improved visuals
st.markdown("""
<style>
    .failure-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .failure-card.error {
        background-color: rgba(255, 99, 71, 0.2);
        border-left: 5px solid #ff6347;
    }
    .failure-card.warning {
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 5px solid orange;
    }
    .failure-card.success {
        background-color: rgba(0, 128, 0, 0.1);
        border-left: 5px solid green;
    }
    .status-title {
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    .status-detail {
        font-size: 0.95em;
    }
    /* Make tabs larger and more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Predictive Maintenance Dashboard</h1>", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.running = False
    st.session_state.data = pd.DataFrame(columns=[
        "Time", "Air_Temperature", "Process_Temperature", "Rotational_Speed", 
        "Torque", "Tool_Wear", "Machine_Failure", "TWF", "HDF", "PWF", "OSF", "RNF"
    ])
    st.session_state.next_maintenance = datetime.now() + timedelta(days=random.randint(10, 30))
    st.session_state.anomaly_history = []
    st.session_state.force_anomaly = False
    st.session_state.iteration = 0
    st.session_state.initialized = True
    st.session_state.failure_details = None
    st.session_state.failure_count = 0

# Sidebar controls
with st.sidebar:
    st.header("Control Panel")
    
    if st.button("Start/Stop Monitoring"):
        st.session_state.running = not st.session_state.running
    
    st.write("Status: " + ("✅ Active" if st.session_state.running else "⏹️ Paused"))
    
    st.subheader("Settings")
    machine_type = st.selectbox("Machine Type", ["Production Line", "CNC Machine", "Industrial Robot"], index=0)
    update_interval = st.slider("Update Speed", 1, 10, 3)
    max_points = st.slider("History Length", 20, 100, 50)
    
    st.subheader("Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        air_temp_threshold = st.number_input("Air Temp (K)", 300, 350, 330, step=5)
        process_temp_threshold = st.number_input("Process Temp (K)", 310, 360, 340, step=5)
    with col2:
        torque_threshold = st.number_input("Torque (Nm)", 35, 80, 60, step=5)
        tool_wear_threshold = st.number_input("Tool Wear (min)", 150, 250, 200, step=10)
    
    st.subheader("Test Anomalies")
    anomaly_options = ["Random Failure", "Tool Wear Failure", "Heat Failure", "Power Failure", "Overstrain Failure"]
    selected_anomaly = st.selectbox("Anomaly Type", anomaly_options)
    
    if st.button("Force Selected Failure"):
        st.session_state.force_anomaly = True
        st.session_state.forced_anomaly_type = selected_anomaly
    
    st.subheader("Maintenance")
    if st.button("Perform Maintenance"):
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.next_maintenance = datetime.now() + timedelta(days=random.randint(10, 30))
        # Reset tool wear
        if not st.session_state.data.empty:
            last_row = st.session_state.data.iloc[-1].copy()
            last_row["Tool_Wear"] = 0
            st.session_state.data.iloc[-1] = last_row

# Create main layout
col1, col2 = st.columns([3, 2])

# The main simulation function
def generate_data():
    # Keep data within size limit
    if len(st.session_state.data) > max_points:
        st.session_state.data = st.session_state.data.iloc[-max_points:]
    
    # Get last values or initialize defaults
    if len(st.session_state.data) > 0:
        last_air_temp = st.session_state.data["Air_Temperature"].iloc[-1]
        last_process_temp = st.session_state.data["Process_Temperature"].iloc[-1]
        last_rpm = st.session_state.data["Rotational_Speed"].iloc[-1]
        last_torque = st.session_state.data["Torque"].iloc[-1]
        last_tool_wear = st.session_state.data["Tool_Wear"].iloc[-1]
    else:
        last_air_temp, last_process_temp = 298.0, 308.0
        last_rpm, last_torque, last_tool_wear = 1500, 40.0, 0
    
    # Simulate sensor readings with realistic patterns
    tool_wear = last_tool_wear + random.uniform(0.8, 2.2)
    air_temp = max(285, min(345, last_air_temp + random.uniform(-1.2, 1.8)))
    process_temp = max(295, min(355, last_process_temp + random.uniform(-0.8, 1.2)))
    rpm_variance = 50
    rpm = max(1000, min(2500, last_rpm + random.uniform(-rpm_variance, rpm_variance)))
    torque = max(20, min(80, last_torque + random.uniform(-3, 3)))
    
    # Initialize failure indicators
    machine_failure = twf = hdf = pwf = osf = rnf = 0
    
    # Store failure reasons and insights
    failure_reasons = []
    prediction_insights = []
    
    # Check for failures based on thresholds
    if air_temp > air_temp_threshold:
        hdf = machine_failure = 1
        failure_reasons.append(f"Air temperature spike: {air_temp:.1f}K > {air_temp_threshold}K")
        prediction_insights.append("Check cooling system - potential fan failure")
    
    if process_temp > process_temp_threshold:
        hdf = machine_failure = 1
        failure_reasons.append(f"Process temperature critical: {process_temp:.1f}K > {process_temp_threshold}K")
        prediction_insights.append("Investigate heat dissipation and lubrication")
    
    if torque > torque_threshold:
        osf = machine_failure = 1
        failure_reasons.append(f"Excessive torque: {torque:.1f}Nm > {torque_threshold}Nm")
        prediction_insights.append("Check for mechanical obstructions")
    
    if tool_wear > tool_wear_threshold:
        twf = machine_failure = 1
        failure_reasons.append(f"Tool critically worn: {tool_wear:.1f}min > {tool_wear_threshold}min")
        prediction_insights.append("Replace cutting tool immediately")
        
    if rpm < 800:
        pwf = machine_failure = 1
        failure_reasons.append(f"RPM drop: {rpm:.0f}RPM < 800RPM")
        prediction_insights.append("Investigate power supply or motor issues")
    
    # Random failure probability
    if random.random() < 0.01:
        rnf = machine_failure = 1
        failure_reasons.append("Unexpected system anomaly detected")
        prediction_insights.append("Run diagnostic tests on control systems")
    
    # Force failure if button was pressed
    if st.session_state.force_anomaly:
        if hasattr(st.session_state, 'forced_anomaly_type'):
            failure_type = st.session_state.forced_anomaly_type
            
            if failure_type == "Tool Wear Failure":
                twf = 1
                failure_reasons.append(f"Manual simulation: Tool wear failure (wear: {tool_wear:.1f}min)")
                prediction_insights.append("Replace tool and check calibration")
            elif failure_type == "Heat Failure":
                hdf = 1
                failure_reasons.append(f"Manual simulation: Heat failure (air: {air_temp:.1f}K)")
                prediction_insights.append("Check coolant levels and cooling system")
            elif failure_type == "Power Failure":
                pwf = 1
                failure_reasons.append(f"Manual simulation: Power failure (RPM: {rpm:.0f})")
                prediction_insights.append("Inspect power connections and controller")
            elif failure_type == "Overstrain Failure":
                osf = 1
                failure_reasons.append(f"Manual simulation: Overstrain (torque: {torque:.1f}Nm)")
                prediction_insights.append("Check mechanical alignment")
            else:
                rnf = 1
                failure_reasons.append("Manual simulation: Random system failure")
                prediction_insights.append("Run full system diagnostics")
                
        machine_failure = 1
        st.session_state.force_anomaly = False
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Add new data point
    new_data = pd.DataFrame([[
        current_time, air_temp, process_temp, rpm, torque, tool_wear,
        machine_failure, twf, hdf, pwf, osf, rnf
    ]], columns=st.session_state.data.columns)
    
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
    
    # Update failure count and store details
    if machine_failure:
        st.session_state.failure_count += 1
        
        failure_types = []
        if twf: failure_types.append("Tool Wear")
        if hdf: failure_types.append("Heat Dissipation")
        if pwf: failure_types.append("Power")
        if osf: failure_types.append("Overstrain")
        if rnf: failure_types.append("Random")
        
        st.session_state.failure_details = {
            "time": current_time,
            "types": failure_types,
            "reasons": failure_reasons,
            "insights": prediction_insights
        }
        
        # Record anomalies for history
        st.session_state.anomaly_history.append({
            "time": current_time,
            "types": failure_types,
            "reasons": failure_reasons,
            "insights": prediction_insights
        })
        
        # Keep only recent anomalies
        if len(st.session_state.anomaly_history) > 10:
            st.session_state.anomaly_history.pop(0)
    else:
        st.session_state.failure_details = None

# Main dashboard area
with col1:
    if not st.session_state.data.empty:
        # Create the sensor charts in tabs with larger size for better visibility
        tab1, tab2 = st.tabs(["Temperature Parameters", "Mechanical Parameters"])
        
        with tab1:
            # Temperature charts with increased height
            st.markdown("### Temperature Monitoring")
            
            # Air Temperature chart
            fig_air = px.line(st.session_state.data, x="Time", y="Air_Temperature", 
                            title="Air Temperature (K)")
            fig_air.add_hline(y=air_temp_threshold, line_dash="dash", line_color="red", 
                            annotation_text="Threshold", annotation_position="bottom right")
            fig_air.update_traces(line_color='#1f77b4', line_width=4)
            fig_air.update_layout(
                height=400,  # Increased height
                plot_bgcolor='rgba(240, 240, 240, 0.5)', 
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_air, use_container_width=True)
            
            # Process Temperature chart
            fig_process = px.line(st.session_state.data, x="Time", y="Process_Temperature", 
                                title="Process Temperature (K)")
            fig_process.add_hline(y=process_temp_threshold, line_dash="dash", line_color="red",
                                annotation_text="Threshold", annotation_position="bottom right")
            fig_process.update_traces(line_color='#ff7f0e', line_width=4)
            fig_process.update_layout(
                height=400,
                plot_bgcolor='rgba(240, 240, 240, 0.5)', 
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_process, use_container_width=True)
        
        with tab2:
            # Mechanical parameters
            st.markdown("### Mechanical Parameter Monitoring")
            
            # Use columns for better organization
            col1a, col1b = st.columns(2)
            
            with col1a:
                # RPM chart
                fig_rpm = px.line(st.session_state.data, x="Time", y="Rotational_Speed", 
                                title="Rotational Speed (RPM)")
                fig_rpm.update_traces(line_color='#2ca02c', line_width=4)
                fig_rpm.add_hline(y=800, line_dash="dash", line_color="red", 
                                annotation_text="Min RPM", annotation_position="bottom right")
                fig_rpm.update_layout(
                    height=350,
                    plot_bgcolor='rgba(240, 240, 240, 0.5)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                st.plotly_chart(fig_rpm, use_container_width=True)
            
            with col1b:
                # Torque chart
                fig_torque = px.line(st.session_state.data, x="Time", y="Torque", 
                                    title="Torque (Nm)")
                fig_torque.add_hline(y=torque_threshold, line_dash="dash", line_color="red",
                                    annotation_text="Threshold", annotation_position="bottom right")
                fig_torque.update_traces(line_color='#d62728', line_width=4)
                fig_torque.update_layout(
                    height=350,
                    plot_bgcolor='rgba(240, 240, 240, 0.5)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                st.plotly_chart(fig_torque, use_container_width=True)
            
            # Tool wear chart - full width
            fig_wear = px.line(st.session_state.data, x="Time", y="Tool_Wear", 
                            title="Tool Wear (min)")
            fig_wear.add_hline(y=tool_wear_threshold, line_dash="dash", line_color="red",
                            annotation_text="Replacement Threshold", annotation_position="bottom right")
            fig_wear.add_hline(y=tool_wear_threshold*0.7, line_dash="dot", line_color="orange",
                            annotation_text="Warning Level", annotation_position="bottom right")
            fig_wear.update_traces(line_color='#9467bd', line_width=4)
            fig_wear.update_layout(
                height=350,
                plot_bgcolor='rgba(240, 240, 240, 0.5)', 
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_wear, use_container_width=True)

# Right column for status and failure information
with col2:
    # Enhanced failure information
    st.subheader("System Status")
    
    if not st.session_state.data.empty:
        latest_data = st.session_state.data.iloc[-1]
        
        # Current status indicator
        if latest_data["Machine_Failure"]:
            failure_types = []
            if latest_data["TWF"]: failure_types.append("Tool Wear")
            if latest_data["HDF"]: failure_types.append("Heat Dissipation")
            if latest_data["PWF"]: failure_types.append("Power")
            if latest_data["OSF"]: failure_types.append("Overstrain")
            if latest_data["RNF"]: failure_types.append("Random")
            
            # Display current failure with reasons
            st.markdown(f"""
            <div class="failure-card error">
                <div class="status-title">🚨 MACHINE FAILURE DETECTED</div>
                <div class="status-detail">
                    <b>Failure type(s):</b> {", ".join(failure_types)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show failure reasons and actionable insights
            if st.session_state.failure_details:
                with st.expander("Failure Details", expanded=True):
                    for reason in st.session_state.failure_details["reasons"]:
                        st.warning(reason)
                
                with st.expander("Recommended Actions", expanded=True):
                    for insight in st.session_state.failure_details["insights"]:
                        st.info(insight)
        else:
            st.markdown("""
            <div class="failure-card success">
                <div class="status-title">✅ SYSTEM OPERATING NORMALLY</div>
                <div class="status-detail">
                    All parameters within acceptable ranges
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Failure history with improved visuals
    st.subheader("Failure History")
    
    if st.session_state.anomaly_history:
        # Summary stats
        st.markdown(f"**Total failures detected:** {st.session_state.failure_count}")
        
        # Show pie chart of failure types if there are failures
        if st.session_state.failure_count > 0:
            failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
            failure_counts = [st.session_state.data[ft].sum() for ft in failure_types]
            
            if sum(failure_counts) > 0:
                fig_failure_dist = go.Figure(data=[go.Pie(
                    labels=["Tool Wear", "Heat Dissipation", "Power", "Overstrain", "Random"],
                    values=failure_counts,
                    hole=.4,
                    marker_colors=['#ff9ff3','#feca57','#1dd1a1','#54a0ff','#5f27cd']
                )])
                fig_failure_dist.update_layout(
                    title="Failure Type Distribution",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_failure_dist, use_container_width=True)
        
        # Recent failure log with detailed information
        st.markdown("#### Recent Failure Log")
        for anomaly in reversed(st.session_state.anomaly_history):
            with st.expander(f"🕒 {anomaly['time']} - {', '.join(anomaly['types'])} Failure"):
                st.markdown("**Detected issues:**")
                for reason in anomaly["reasons"]:
                    st.markdown(f"- {reason}")
                
                st.markdown("**Recommended actions:**")
                for insight in anomaly["insights"]:
                    st.markdown(f"- {insight}")
    else:
        st.success("No failures detected in recent history")
    
    # Maintenance recommendations section
    st.subheader("Maintenance Information")
    
    # Display current tool wear status and prediction
    if not st.session_state.data.empty:
        current_wear = st.session_state.data["Tool_Wear"].iloc[-1]
        wear_percentage = (current_wear / tool_wear_threshold) * 100
        
        # Create progress bar for tool wear
        st.markdown("#### Tool Wear Status")
        wear_color = "green" if wear_percentage < 70 else "orange" if wear_percentage < 90 else "red"
        st.markdown(f"""
        <div style="margin-bottom: 10px;">Current: {current_wear:.1f}min / {tool_wear_threshold}min ({wear_percentage:.1f}%)</div>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px;">
            <div style="background-color: {wear_color}; width: {min(100, wear_percentage)}%; height: 100%; border-radius: 10px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Maintenance recommendation based on wear
        if wear_percentage > 90:
            st.error("🚨 **CRITICAL:** Schedule immediate tool replacement")
        elif wear_percentage > 70:
            st.warning("⚠️ **WARNING:** Plan for tool replacement soon")
        else:
            next_maint = st.session_state.next_maintenance.strftime("%b %d, %Y")
            st.info(f"✓ Tool in good condition. Next scheduled maintenance: {next_maint}")
        
        # Predictive information
        if wear_percentage > 50:
            # Simple linear prediction
            remaining_cycles = int((tool_wear_threshold - current_wear) / (st.session_state.data["Tool_Wear"].diff().mean() or 1))
            st.markdown(f"**Estimated remaining operational time:** ~{remaining_cycles} cycles")
            
            # Failure probability based on current conditions
            if wear_percentage > 80:
                st.markdown("**Failure probability:** High ⚠️")
            elif wear_percentage > 60:
                st.markdown("**Failure probability:** Medium 🔍")
            else:
                st.markdown("**Failure probability:** Low ✓")

# Auto-update mechanism
if st.session_state.running:
    # Update data
    generate_data()
    
    # Increment iteration counter
    st.session_state.iteration += 1
    
    # Auto-refresh using streamlit's rerun
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()

    current_time = time.time()
    elapsed = current_time - st.session_state.last_update
    
    if elapsed >= (1.0 / update_interval * 2):
        st.session_state.last_update = current_time
        time.sleep(0.1)  # Prevent CPU overuse
        st.rerun()
else:
    # Generate initial data point if needed
    if st.session_state.data.empty:
        generate_data()
