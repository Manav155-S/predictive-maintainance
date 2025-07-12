import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI-Powered Predictive Maintenance", 
    page_icon="🔧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .stAlert {
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize ML models and scalers
@st.cache_resource
def initialize_models():
    """Initialize and return ML models"""
    anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    return anomaly_detector, failure_predictor, scaler

# Generate historical data for model training
@st.cache_data
def generate_historical_data(n_samples=10000):
    """Generate synthetic historical data for model training"""
    np.random.seed(42)
    
    # Generate normal operating conditions
    normal_temp = np.random.normal(75, 8, int(n_samples * 0.8))
    normal_pressure = np.random.normal(45, 10, int(n_samples * 0.8))
    normal_vibration = np.random.normal(2.0, 0.8, int(n_samples * 0.8))
    normal_humidity = np.random.normal(60, 15, int(n_samples * 0.8))
    normal_rpm = np.random.normal(1500, 200, int(n_samples * 0.8))
    
    # Generate anomalous conditions
    anomaly_temp = np.random.normal(95, 5, int(n_samples * 0.2))
    anomaly_pressure = np.random.normal(75, 8, int(n_samples * 0.2))
    anomaly_vibration = np.random.normal(4.5, 1.0, int(n_samples * 0.2))
    anomaly_humidity = np.random.normal(85, 10, int(n_samples * 0.2))
    anomaly_rpm = np.random.normal(1200, 300, int(n_samples * 0.2))
    
    # Combine data
    temperature = np.concatenate([normal_temp, anomaly_temp])
    pressure = np.concatenate([normal_pressure, anomaly_pressure])
    vibration = np.concatenate([normal_vibration, anomaly_vibration])
    humidity = np.concatenate([normal_humidity, anomaly_humidity])
    rpm = np.concatenate([normal_rpm, anomaly_rpm])
    
    # Create timestamps
    base_time = datetime.now() - timedelta(days=365)
    timestamps = [base_time + timedelta(minutes=i*10) for i in range(n_samples)]
    
    # Calculate remaining useful life (RUL) - simplified model
    rul = np.maximum(0, 1000 - 0.1 * temperature - 0.05 * pressure - 100 * vibration + np.random.normal(0, 50, n_samples))
    
    # Create machine IDs
    machine_ids = np.random.choice(['M001', 'M002', 'M003', 'M004', 'M005'], n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'machine_id': machine_ids,
        'temperature': temperature,
        'pressure': pressure,
        'vibration': vibration,
        'humidity': humidity,
        'rpm': rpm,
        'remaining_useful_life': rul
    })
    
    return df

# Train ML models
@st.cache_resource
def train_models():
    """Train ML models on historical data"""
    # Generate and prepare data
    df = generate_historical_data()
    
    # Prepare features for anomaly detection
    anomaly_features = ['temperature', 'pressure', 'vibration', 'humidity', 'rpm']
    X_anomaly = df[anomaly_features].values
    
    # Prepare features for RUL prediction
    rul_features = ['temperature', 'pressure', 'vibration', 'humidity', 'rpm']
    X_rul = df[rul_features].values
    y_rul = df['remaining_useful_life'].values
    
    # Initialize models
    anomaly_detector, failure_predictor, scaler = initialize_models()
    
    # Train anomaly detector
    X_anomaly_scaled = scaler.fit_transform(X_anomaly)
    anomaly_detector.fit(X_anomaly_scaled)
    
    # Train RUL predictor
    X_train, X_test, y_train, y_test = train_test_split(X_rul, y_rul, test_size=0.2, random_state=42)
    failure_predictor.fit(X_train, y_train)
    
    # Calculate model performance
    y_pred = failure_predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return anomaly_detector, failure_predictor, scaler, mse, r2, df

# Header
st.markdown("""
<div class="main-header">
    <h1>🔧 AI-Powered Predictive Maintenance System</h1>
    <p>Real-time Machine Health Monitoring with Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "timestamp", "machine_id", "temperature", "pressure", 
        "vibration", "humidity", "rpm", "anomaly_score", "predicted_rul"
    ])

# Machine selection
selected_machine = st.sidebar.selectbox(
    "Select Machine ID:",
    ["M001", "M002", "M003", "M004", "M005"],
    index=0
)

# Simulation controls
st.sidebar.markdown("### Simulation Controls")
simulation_speed = st.sidebar.slider("Simulation Speed (seconds)", 0.5, 5.0, 1.0, 0.5)
add_noise = st.sidebar.checkbox("Add Sensor Noise", value=True)

# Model information
st.sidebar.markdown("### 🤖 Model Information")
with st.sidebar.expander("Model Performance"):
    # Train models and get performance metrics
    anomaly_detector, failure_predictor, scaler, mse, r2, historical_df = train_models()
    
    st.metric("RUL Model R² Score", f"{r2:.3f}")
    st.metric("RUL Model MSE", f"{mse:.2f}")
    st.info("Models trained on 10,000 historical data points")

# Start/Stop button
if st.sidebar.button("🚀 Start Simulation" if not st.session_state.running else "⏹️ Stop Simulation"):
    st.session_state.running = not st.session_state.running

# Display current status
if st.session_state.running:
    st.sidebar.success("✅ Simulation Running")
else:
    st.sidebar.info("⏸️ Simulation Stopped")

# Main dashboard
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# Placeholders for real-time updates
metrics_placeholder = st.empty()
charts_placeholder = st.empty()
alerts_placeholder = st.empty()
analysis_placeholder = st.empty()

# Main simulation loop
while st.session_state.running:
    # Generate realistic sensor readings
    base_temp = 75 + np.sin(time.time() / 100) * 10
    base_pressure = 45 + np.sin(time.time() / 80) * 5
    base_vibration = 2.0 + np.sin(time.time() / 60) * 0.5
    base_humidity = 60 + np.sin(time.time() / 120) * 10
    base_rpm = 1500 + np.sin(time.time() / 90) * 100
    
    # Add noise if enabled
    if add_noise:
        noise_factor = 0.1
        temperature = base_temp + np.random.normal(0, noise_factor * base_temp)
        pressure = base_pressure + np.random.normal(0, noise_factor * base_pressure)
        vibration = base_vibration + np.random.normal(0, noise_factor * base_vibration)
        humidity = base_humidity + np.random.normal(0, noise_factor * base_humidity)
        rpm = base_rpm + np.random.normal(0, noise_factor * base_rpm)
    else:
        temperature, pressure, vibration, humidity, rpm = base_temp, base_pressure, base_vibration, base_humidity, base_rpm
    
    # Occasionally simulate equipment degradation
    if random.random() < 0.1:  # 10% chance of anomaly
        temperature += random.uniform(15, 25)
        pressure += random.uniform(20, 30)
        vibration += random.uniform(2, 3)
    
    # Prepare data for ML predictions
    current_features = np.array([[temperature, pressure, vibration, humidity, rpm]])
    current_features_scaled = scaler.transform(current_features)
    
    # Predict anomaly score
    anomaly_score = anomaly_detector.decision_function(current_features_scaled)[0]
    is_anomaly = anomaly_detector.predict(current_features_scaled)[0] == -1
    
    # Predict remaining useful life
    predicted_rul = failure_predictor.predict(current_features)[0]
    
    # Add to session data
    current_time = datetime.now()
    new_data = pd.DataFrame([[
        current_time, selected_machine, temperature, pressure, vibration, 
        humidity, rpm, anomaly_score, predicted_rul
    ]], columns=st.session_state.data.columns)
    
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
    
    # Keep only last 50 records for display
    if len(st.session_state.data) > 50:
        st.session_state.data = st.session_state.data.tail(50).reset_index(drop=True)
    
    # Display real-time metrics
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🌡️ Temperature", 
                f"{temperature:.1f}°C",
                delta=f"{temperature - 75:.1f}°C" if len(st.session_state.data) > 1 else None
            )
        
        with col2:
            st.metric(
                "🔧 Pressure", 
                f"{pressure:.1f} Bar",
                delta=f"{pressure - 45:.1f} Bar" if len(st.session_state.data) > 1 else None
            )
        
        with col3:
            st.metric(
                "📳 Vibration", 
                f"{vibration:.2f} mm/s",
                delta=f"{vibration - 2.0:.2f} mm/s" if len(st.session_state.data) > 1 else None
            )
        
        with col4:
            rul_color = "normal" if predicted_rul > 200 else "inverse"
            st.metric(
                "⏰ Predicted RUL", 
                f"{predicted_rul:.0f} hours",
                delta=f"{'⚠️' if predicted_rul < 100 else '✅'} {'Critical' if predicted_rul < 100 else 'Good'}"
            )
    
    # Display charts
    with charts_placeholder.container():
        if len(st.session_state.data) > 1:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature Trend', 'Pressure & Vibration', 'Anomaly Score', 'RUL Prediction'),
                specs=[[{"secondary_y": False}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Temperature plot
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data['timestamp'],
                    y=st.session_state.data['temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Pressure and Vibration plot
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data['timestamp'],
                    y=st.session_state.data['pressure'],
                    mode='lines+markers',
                    name='Pressure',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data['timestamp'],
                    y=st.session_state.data['vibration'],
                    mode='lines+markers',
                    name='Vibration',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                ),
                row=1, col=2, secondary_y=True
            )
            
            # Anomaly score plot
            colors = ['red' if score < -0.1 else 'green' for score in st.session_state.data['anomaly_score']]
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data['timestamp'],
                    y=st.session_state.data['anomaly_score'],
                    mode='lines+markers',
                    name='Anomaly Score',
                    line=dict(color='orange', width=2),
                    marker=dict(color=colors)
                ),
                row=2, col=1
            )
            
            # RUL prediction plot
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data['timestamp'],
                    y=st.session_state.data['predicted_rul'],
                    mode='lines+markers',
                    name='Predicted RUL',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="Real-time Machine Health Analytics")
            st.plotly_chart(fig, use_container_width=True)
    
    # Display alerts and analysis
    with alerts_placeholder.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚨 System Alerts")
            alerts = []
            
            if temperature > 90:
                alerts.append(f"🔥 **High Temperature**: {temperature:.1f}°C (Threshold: 90°C)")
            if pressure > 70:
                alerts.append(f"⚠️ **High Pressure**: {pressure:.1f} Bar (Threshold: 70 Bar)")
            if vibration > 4.0:
                alerts.append(f"📳 **High Vibration**: {vibration:.2f} mm/s (Threshold: 4.0 mm/s)")
            if is_anomaly:
                alerts.append(f"🤖 **ML Anomaly Detected**: Score {anomaly_score:.3f}")
            if predicted_rul < 100:
                alerts.append(f"⏰ **Maintenance Required**: RUL {predicted_rul:.0f} hours")
            
            if alerts:
                for alert in alerts:
                    st.error(alert)
            else:
                st.success("✅ All systems operating normally")
        
        with col2:
            st.markdown("### 📊 AI Analysis")
            
            # Health score calculation
            health_score = 100 - (
                max(0, (temperature - 75) / 25 * 30) +
                max(0, (pressure - 45) / 35 * 25) +
                max(0, (vibration - 2.0) / 3.0 * 25) +
                max(0, (-anomaly_score + 0.1) / 0.5 * 20)
            )
            health_score = max(0, min(100, health_score))
            
            # Display health score with color coding
            if health_score > 80:
                st.success(f"🟢 **Machine Health Score**: {health_score:.1f}/100 (Excellent)")
            elif health_score > 60:
                st.warning(f"🟡 **Machine Health Score**: {health_score:.1f}/100 (Good)")
            else:
                st.error(f"🔴 **Machine Health Score**: {health_score:.1f}/100 (Poor)")
            
            # Maintenance recommendations
            if predicted_rul < 100:
                st.info("📋 **Recommendation**: Schedule immediate maintenance")
            elif predicted_rul < 200:
                st.info("📋 **Recommendation**: Plan maintenance within 1 week")
            else:
                st.info("📋 **Recommendation**: Continue normal operation")
    
    # Wait before next iteration
    time.sleep(simulation_speed)

# Footer when simulation is stopped
if not st.session_state.running:
    st.markdown("---")
    st.markdown("""
    ### 📈 Project Features:
    - **Real-time Monitoring**: Live sensor data visualization
    - **Anomaly Detection**: ML-based anomaly detection using Isolation Forest
    - **Predictive Maintenance**: Remaining Useful Life (RUL) prediction using Random Forest
    - **Interactive Dashboard**: Professional UI with real-time updates
    - **Alert System**: Intelligent alerting based on thresholds and ML predictions
    - **Health Scoring**: Comprehensive machine health assessment
    
    ### 🛠️ Technologies Used:
    - **Frontend**: Streamlit, Plotly
    - **Machine Learning**: Scikit-learn (Isolation Forest, Random Forest)
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Custom CSS
    """)
    
    # Display sample of historical data
    if st.checkbox("Show Historical Training Data Sample"):
        st.dataframe(historical_df.head(100))
