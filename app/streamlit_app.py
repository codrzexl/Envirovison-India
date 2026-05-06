import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import sys
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import pandas as pd
import pydeck as pdk

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_service import EnvironmentalAPI
from utils.ai_service import GeminiService
from utils.config_loader import config
from utils.data_utils import DataPreprocessor
from utils.grad_cam import GradCAM
from utils.simple_grad_cam import SimpleGradCAM

# Page configuration
st.set_page_config(
    page_title="EnviroVision India",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.grad_cams = {}
        self.preprocessor = DataPreprocessor(config)
        self.load_models()
        self.load_label_encoders()
    
    def load_models(self):
        """Load trained models"""
        models_dir = Path(config.get('paths.models_dir'))
        categories = ['rainfall', 'heatwave', 'air_quality']
        
        for category in categories:
            model_path = models_dir / f"{category}_best_model.h5"
            if model_path.exists():
                try:
                    self.models[category] = tf.keras.models.load_model(str(model_path))
                    
                    # Try to use regular GradCAM, fallback to SimpleGradCAM for dummy models
                    try:
                        self.grad_cams[category] = GradCAM(self.models[category])
                    except Exception:
                        self.grad_cams[category] = SimpleGradCAM(self.models[category])
                        
                except Exception as e:
                    st.error(f"❌ Failed to load {category} model: {e}")
    
    def load_label_encoders(self):
        """Load label encoders"""
        encoder_path = Path(config.get('paths.models_dir')) / 'label_encoders.json'
        if encoder_path.exists():
            with open(encoder_path, 'r') as f:
                encoders_data = json.load(f)
            
            for category, encoder_data in encoders_data.items():
                self.label_encoders[category] = encoder_data['classes']
        else:
            for category in ['rainfall', 'heatwave', 'air_quality']:
                self.label_encoders[category] = config.get(f'classes.{category}')
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = cv2.resize(image, tuple(config.get('data.image_size')))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return image
    
    def predict_category(self, image, category):
        """Make prediction for a specific category"""
        if category not in self.models:
            return None, None, None
        processed_image = self.preprocess_image(image)
        image_batch = np.expand_dims(processed_image, axis=0)
        predictions = self.models[category].predict(image_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        class_names = self.label_encoders[category]
        predicted_class = class_names[predicted_class_idx]
        return predicted_class, confidence, predictions
    
    def generate_grad_cam(self, image, category, class_idx=None):
        """Generate Grad-CAM visualization"""
        try:
            if category not in self.grad_cams:
                return None, None
            processed_image = self.preprocess_image(image)
            heatmap = self.grad_cams[category].generate_heatmap(processed_image, class_index=class_idx)
            overlay, heatmap_colored = self.grad_cams[category].overlay_heatmap(processed_image, heatmap)
            return overlay, heatmap_colored
        except Exception:
            return None, None

# Initialize model manager
@st.cache_resource
def load_model_manager():
    return ModelManager()

model_manager = load_model_manager()

# --- Advanced UI Components ---
def display_risk_banner(api_results):
    """Global Risk Summary Banner"""
    risk_score = api_results.get('overall_risk', 0)
    status = api_results.get('weather_status', 'Unknown')
    emoji = api_results.get('weather_emoji', '❓')
    
    if risk_score < 3:
        color = "#28a745"
        msg = "Safe Environment"
        sub = "All indicators are within healthy limits."
    elif risk_score < 6:
        color = "#ffc107"
        msg = "Moderate Risk"
        sub = "Use caution. Some environmental factors are elevated."
    else:
        color = "#dc3545"
        msg = "High Danger Alert"
        sub = "Critical environmental conditions detected. Follow health advisories!"

    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px;">
        <h1 style="margin: 0; font-size: 2.5rem;">{emoji} {status} | Risk Score: {risk_score}/10</h1>
        <h2 style="margin: 5px 0 0 0; opacity: 0.9;">{msg}</h2>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem;">{sub}</p>
    </div>
    """, unsafe_allow_html=True)

def display_api_data(city_data, api_results):
    """Display real-time environmental data from API"""
    st.markdown(f"### 📡 Live Monitoring: {city_data['name']}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        val = api_results['rainfall']['value']
        pop = api_results['rainfall'].get('pop', 0)
        label = api_results['rainfall']['label']
        st.metric("Current Rainfall", f"{val} mm", delta=f"Chance: {pop}%", delta_color="normal")
        
    with col2:
        val = api_results['heatwave']['value']
        feels_val = api_results['heatwave']['feels_like']
        label = api_results['heatwave']['label']
        st.metric("Temperature", f"{val} °C", delta=f"Feels: {feels_val} °C", delta_color="inverse")
        
    with col3:
        aqi_val = api_results['air_quality']['value']
        aqi_label = api_results['air_quality']['label']
        st.metric("Air Quality (AQI)", f"{aqi_val}", delta=aqi_label, delta_color="inverse")
    
    st.markdown("#### 🔘 Real-time AQI Meter")
    aqi_color = api_results['air_quality'].get('color', 'blue')
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = aqi_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Current AQI: {aqi_label}"},
        gauge = {
            'axis': {'range': [None, 500]},
            'bar': {'color': aqi_color},
            'steps' : [
                {'range': [0, 50], 'color': "#00E400"},
                {'range': [51, 100], 'color': "#FFFF00"},
                {'range': [101, 150], 'color': "#FF7E00"},
                {'range': [151, 200], 'color': "#FF0000"},
                {'range': [201, 300], 'color': "#8F3F97"},
                {'range': [301, 500], 'color': "#7E0023"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': aqi_val}
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

def display_heatwave_deep_dive(heat_data):
    """Visual thermometer and heat safety advisory"""
    st.markdown("---")
    st.markdown("### 🌡️ Heatwave Analysis & Safety Advisory")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        temp = heat_data['value']
        color = heat_data['color']
        fig_therm = go.Figure(go.Indicator(
            mode = "gauge+number", value = temp, domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Thermometer (°C)"},
            gauge = {
                'shape': "bullet", 'axis': {'range': [20, 60]}, 'bar': {'color': color},
                'steps': [
                    {'range': [20, 35], 'color': "lightblue"},
                    {'range': [35, 40], 'color': "yellow"},
                    {'range': [40, 45], 'color': "orange"},
                    {'range': [45, 60], 'color': "red"}
                ],
            }
        ))
        fig_therm.update_layout(height=150)
        st.plotly_chart(fig_therm, use_container_width=True)
        
    with col2:
        st.markdown(f"#### **Category:** {heat_data['label']}")
        st.warning(f"🌡️ **Heat Advisory:** {heat_data['recommendation']}")
        c1, c2 = st.columns(2)
        c1.metric("Relative Humidity", f"{heat_data['humidity']}%")
        c2.metric("Heat Index (Feels Like)", f"{heat_data['feels_like']} °C")

def display_rainfall_deep_dive(rain_data):
    """Rainfall intensity gauge and flood safety advisory"""
    st.markdown("---")
    st.markdown("### 🌧️ Rainfall Analysis & Flood Safety")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        val = rain_data['value']
        color = rain_data['color']
        fig_rain = go.Figure(go.Indicator(
            mode = "gauge+number", value = val, domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Rain Intensity (mm)"},
            gauge = {
                'axis': {'range': [0, 50]}, 'bar': {'color': color},
                'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2.5], 'color': "#e0f7fa"},
                    {'range': [2.5, 7.5], 'color': "#81d4fa"},
                    {'range': [7.5, 35.5], 'color': "#0288d1"},
                    {'range': [35.5, 50], 'color': "#01579b"}
                ],
            }
        ))
        fig_rain.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_rain, use_container_width=True)
        
    with col2:
        st.markdown(f"#### **Intensity:** {rain_data['label']}")
        st.info(f"🌊 **Flood Advisory:** {rain_data['recommendation']}")
        c1, c2 = st.columns(2)
        c1.metric("Precipitation Probability", f"{rain_data['pop']}%")
        c2.metric("Accumulated (Current)", f"{rain_data['value']} mm")

def display_comparison(ai_preds, api_data):
    """Compare AI predictions with Ground Truth API data"""
    st.markdown("### ⚖️ AI vs. Ground Truth Comparison")
    comparison_data = []
    for cat in ['rainfall', 'heatwave', 'air_quality']:
        if cat in ai_preds and cat in api_data:
            ai_label = ai_preds[cat]['predicted_class']
            api_label = api_data[cat]['label']
            match = "✅ Match" if ai_label.lower() in api_label.lower() or api_label.lower() in ai_label.lower() else "⚠️ Variance"
            comparison_data.append({"Category": cat.title().replace("_", " "), "AI Prediction": ai_label, "API Ground Truth": api_label, "Status": match})
    if comparison_data: st.table(pd.DataFrame(comparison_data))

def display_location_map(lat, lon):
    """Show the selected location on a map"""
    layer = pdk.Layer('ScatterplotLayer', data=pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius=1000)
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=11, pitch=50)
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer]))

def display_trend_charts(trends):
    """Display 24-hour trend charts for Temp and Rain"""
    if not trends['times']: return
    df = pd.DataFrame({'Time': pd.to_datetime(trends['times']), 'Temperature (°C)': trends['temp'], 'Precipitation (mm)': trends['precip']})
    st.markdown("### 📈 24-Hour Trends")
    tab1, tab2 = st.tabs(["Temperature Trend", "Rainfall Trend"])
    with tab1: st.plotly_chart(px.line(df, x='Time', y='Temperature (°C)', title='Temperature Over Time'), use_container_width=True)
    with tab2: st.plotly_chart(px.bar(df, x='Time', y='Precipitation (mm)', title='Expected Rainfall'), use_container_width=True)

def display_aqi_deep_dive(aqi_data, trends):
    """Deep dive into Air Quality Pollutants and Health Advice"""
    st.markdown("---")
    st.markdown("### 🌫️ AQI Deep Dive & Health Advisory")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"#### **Status:** {aqi_data['label']}")
        st.info(f"💡 **Health Advice:** {aqi_data['recommendation']}")
        st.dataframe(pd.DataFrame(aqi_data['pollutants'].items(), columns=['Pollutant', 'Value (µg/m³)']), hide_index=True, use_container_width=True)
    with col2:
        if 'aqi_times' in trends:
            aqi_df = pd.DataFrame({'Time': pd.to_datetime(trends['aqi_times']), 'AQI': trends['aqi_values']})
            st.plotly_chart(px.area(aqi_df, x='Time', y='AQI', title='AQI Forecast', color_discrete_sequence=['#ff4b4b']).update_layout(height=250), use_container_width=True)
        categories = list(aqi_data['pollutants'].keys())
        values = list(aqi_data['pollutants'].values())
        fig_radar = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Pollutants', line_color='#1f77b4'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])), showlegend=False, height=300)
        st.plotly_chart(fig_radar, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🌍 EnviroVision India</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Environmental Scene Classification")
    
    st.sidebar.title("🎛️ Controls")
    input_method = st.sidebar.selectbox("Choose Input Method", ["Upload Image", "Camera Capture", "Webcam Stream"])
    selected_categories = st.sidebar.multiselect("Select Categories to Analyze", ["rainfall", "heatwave", "air_quality"], default=["rainfall", "heatwave", "air_quality"])
    show_grad_cam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
    show_confidence_chart = st.sidebar.checkbox("Show Confidence Chart", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Real-time Monitoring")
    indian_cities = ["New Delhi", "Mumbai", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Surat", "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Coimbatore", "Agra", "Madurai", "Nashik"]
    city_name = st.sidebar.selectbox("Select City (India)", sorted(indian_cities))
    
    city_data = EnvironmentalAPI.get_location_coordinates(city_name)
    api_results = None
    if city_data:
        weather_data = EnvironmentalAPI.get_weather_data(city_data['lat'], city_data['lon'])
        aq_data = EnvironmentalAPI.get_air_quality_data(city_data['lat'], city_data['lon'])
        api_results = EnvironmentalAPI.classify_from_api(weather_data, aq_data)
        st.sidebar.success(f"📍 Station: {city_data.get('admin1', 'N/A')}")
        st.sidebar.info(f"🏙️ Area: {city_data.get('name', 'N/A')}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Intelligence Layer")
    gemini_key = st.sidebar.text_input("Gemini API Key", value="AIzaSyBtt79Hl4RUNHvcMykCfBOnRWr-6bPy96g", type="password")
    ai_service = GeminiService(gemini_key) if gemini_key else None
    
    if api_results: display_risk_banner(api_results)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📤 Input")
        image = None
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file: image = Image.open(uploaded_file); st.image(image, caption="Uploaded Image", use_container_width=True)
        elif input_method == "Camera Capture":
            camera_image = st.camera_input("Take a picture")
            if camera_image: image = Image.open(camera_image); st.image(image, caption="Captured Image", use_container_width=True)
        elif input_method == "Webcam Stream":
            st.info("Webcam streaming feature - Click 'START' to begin")
            webrtc_ctx = webrtc_streamer(key="environmental-classification", rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
            if webrtc_ctx.video_transformer and st.button("Capture Frame"):
                if webrtc_ctx.video_transformer.latest_frame is not None:
                    image = Image.fromarray(cv2.cvtColor(webrtc_ctx.video_transformer.latest_frame, cv2.COLOR_BGR2RGB))
                    st.image(image, caption="Captured Frame", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Predictions")
        if image:
            predictions_data = {}
            for category in selected_categories:
                pred_class, confidence, all_preds = model_manager.predict_category(image, category)
                if pred_class: predictions_data[category] = {'predicted_class': pred_class, 'confidence': confidence, 'all_predictions': all_preds}
            
            for category, pred_data in predictions_data.items():
                st.markdown(f'<div class="category-header">🌡️ {category.title()} Classification</div>', unsafe_allow_html=True)
                conf = pred_data['confidence']
                conf_class = "confidence-high" if conf > 0.8 else "confidence-medium" if conf > 0.6 else "confidence-low"
                st.markdown(f'<div class="prediction-box"><strong>Prediction:</strong> {pred_data["predicted_class"]}<br><strong>Confidence:</strong> <span class="{conf_class}">{conf:.2%}</span></div>', unsafe_allow_html=True)
                if show_confidence_chart: st.plotly_chart(px.bar(x=model_manager.label_encoders[category], y=pred_data['all_predictions'], title=f"{category.title()} Probabilities").update_layout(height=300), use_container_width=True)
            if api_results: display_comparison(predictions_data, api_results)
    
    if api_results:
        st.markdown("---")
        c_m1, c_m2 = st.columns([2, 1])
        with c_m1: display_api_data(city_data, api_results); display_trend_charts(api_results['trends'])
        with c_m2: st.markdown("#### 🗺️ Station Location"); display_location_map(city_data['lat'], city_data['lon'])
        display_aqi_deep_dive(api_results['air_quality'], api_results['trends'])
        display_heatwave_deep_dive(api_results['heatwave'])
        display_rainfall_deep_dive(api_results['rainfall'])
    
    if image:
        st.markdown("---"); st.markdown("### 🧠 AI Environmental Intelligence Report")
        if ai_service and ai_service.is_configured():
            if st.button("🚀 Run Deep Satellite/AI Analysis"):
                with st.spinner("Analyzing..."): st.markdown(ai_service.analyze_environmental_image(image))
        else: st.warning("⚠️ Enter Gemini API Key in sidebar.")

    if image and show_grad_cam and 'predictions_data' in locals():
        st.markdown("### 🔍 Grad-CAM Visualizations")
        gc_cols = st.columns(len(selected_categories))
        for idx, cat in enumerate(selected_categories):
            if cat in predictions_data:
                with gc_cols[idx]:
                    st.markdown(f"**{cat.title()}**")
                    overlay, _ = model_manager.generate_grad_cam(image, cat)
                    if overlay is not None: st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

if __name__ == "__main__":
    main()
