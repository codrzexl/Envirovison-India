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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                        st.success(f"✅ Loaded {category} model with GradCAM")
                    except Exception as gradcam_error:
                        # Use SimpleGradCAM for dummy models
                        self.grad_cams[category] = SimpleGradCAM(self.models[category])
                        st.success(f"✅ Loaded {category} model with SimpleGradCAM")
                        
                except Exception as e:
                    st.error(f"❌ Failed to load {category} model: {e}")
            else:
                st.warning(f"⚠️ Model not found: {model_path}")
    
    def load_label_encoders(self):
        """Load label encoders"""
        encoder_path = Path(config.get('paths.models_dir')) / 'label_encoders.json'
        if encoder_path.exists():
            with open(encoder_path, 'r') as f:
                encoders_data = json.load(f)
            
            for category, encoder_data in encoders_data.items():
                self.label_encoders[category] = encoder_data['classes']
        else:
            # Use default classes from config
            for category in ['rainfall', 'heatwave', 'air_quality']:
                self.label_encoders[category] = config.get(f'classes.{category}')
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize and normalize
        image = cv2.resize(image, tuple(config.get('data.image_size')))
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def predict_category(self, image, category):
        """Make prediction for a specific category"""
        if category not in self.models:
            return None, None
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        predictions = self.models[category].predict(image_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # Get class name
        class_names = self.label_encoders[category]
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions
    
    def generate_grad_cam(self, image, category, class_idx=None):
        """Generate Grad-CAM visualization"""
        try:
            if category not in self.grad_cams:
                return None, None
            
            processed_image = self.preprocess_image(image)
            
            # Generate heatmap
            heatmap = self.grad_cams[category].generate_heatmap(
                processed_image, class_index=class_idx
            )
            
            # Create overlay
            overlay, heatmap_colored = self.grad_cams[category].overlay_heatmap(
                processed_image, heatmap
            )
            
            return overlay, heatmap_colored
        except Exception as e:
            print(f"Error generating Grad-CAM for {category}: {e}")
            return None, None

# Initialize model manager
@st.cache_resource
def load_model_manager():
    return ModelManager()

model_manager = load_model_manager()

# Main app
def main():
    st.markdown('<h1 class="main-header">🌍 EnviroVision India</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Environmental Scene Classification")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method",
        ["Upload Image", "Camera Capture", "Webcam Stream"]
    )
    
    # Category selection
    selected_categories = st.sidebar.multiselect(
        "Select Categories to Analyze",
        ["rainfall", "heatwave", "air_quality"],
        default=["rainfall", "heatwave", "air_quality"]
    )
    
    # Visualization options
    show_grad_cam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
    show_confidence_chart = st.sidebar.checkbox("Show Confidence Chart", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Input")
        
        image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an environmental scene image"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        elif input_method == "Camera Capture":
            camera_image = st.camera_input("Take a picture")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                st.image(image, caption="Captured Image", use_container_width=True)
        
        elif input_method == "Webcam Stream":
            st.info("Webcam streaming feature - Click 'START' to begin")
            
            class VideoTransformer(VideoTransformerBase):
                def __init__(self):
                    self.latest_frame = None
                
                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    self.latest_frame = img
                    return img
            
            webrtc_ctx = webrtc_streamer(
                key="environmental-classification",
                video_processor_factory=VideoTransformer,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
            )
            
            if webrtc_ctx.video_transformer:
                if st.button("Capture Frame"):
                    if webrtc_ctx.video_transformer.latest_frame is not None:
                        frame = webrtc_ctx.video_transformer.latest_frame
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        st.image(image, caption="Captured Frame", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Predictions")
        
        if image is not None:
            # Make predictions for selected categories
            predictions_data = {}
            
            for category in selected_categories:
                if category in model_manager.models:
                    pred_class, confidence, all_preds = model_manager.predict_category(image, category)
                    
                    if pred_class is not None:
                        predictions_data[category] = {
                            'predicted_class': pred_class,
                            'confidence': confidence,
                            'all_predictions': all_preds
                        }
            
            # Display predictions
            for category, pred_data in predictions_data.items():
                st.markdown(f'<div class="category-header">🌡️ {category.title()} Classification</div>', 
                           unsafe_allow_html=True)
                
                confidence = pred_data['confidence']
                pred_class = pred_data['predicted_class']
                
                # Color code confidence
                if confidence > 0.8:
                    conf_class = "confidence-high"
                elif confidence > 0.6:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"
                
                st.markdown(f'''
                <div class="prediction-box">
                    <strong>Prediction:</strong> {pred_class}<br>
                    <strong>Confidence:</strong> <span class="{conf_class}">{confidence:.2%}</span>
                </div>
                ''', unsafe_allow_html=True)
                
                # Show confidence chart
                if show_confidence_chart:
                    class_names = model_manager.label_encoders[category]
                    all_preds = pred_data['all_predictions']
                    
                    fig = px.bar(
                        x=class_names,
                        y=all_preds,
                        title=f"{category.title()} - All Class Probabilities",
                        labels={'x': 'Classes', 'y': 'Probability'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Grad-CAM Visualization
    if image is not None and show_grad_cam and predictions_data:
        st.markdown("### 🔍 Grad-CAM Visualizations")
        
        grad_cam_cols = st.columns(len(selected_categories))
        
        for idx, category in enumerate(selected_categories):
            if category in predictions_data:
                with grad_cam_cols[idx]:
                    st.markdown(f"**{category.title()}**")
                    
                    # Generate Grad-CAM
                    overlay, heatmap = model_manager.generate_grad_cam(image, category)
                    
                    if overlay is not None:
                        try:
                            # Convert BGR to RGB for display
                            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                            st.image(overlay_rgb, caption=f"Grad-CAM - {category}", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display Grad-CAM for {category}: {e}")
                    else:
                        st.warning(f"Grad-CAM visualization not available for {category}")
    
    # Information section
    with st.expander("ℹ️ About EnviroVision India"):
        st.markdown("""
        **EnviroVision India** is an AI-powered system for environmental scene classification that helps analyze:
        
        - 🌧️ **Rainfall Intensity**: Light, Moderate, Heavy
        - 🔥 **Heatwave Severity**: Normal, Mild, Extreme  
        - 🌫️ **Air Quality Levels**: Good, Moderate, Unhealthy
        
        **Features:**
        - Deep learning models using transfer learning (EfficientNet, ResNet)
        - Grad-CAM visualizations for model interpretability
        - Real-time classification through webcam
        - Multi-category analysis
        
        **Data Sources:**
        - Indian Meteorological Department (IMD)
        - NASA Earth Observatory
        - Central Pollution Control Board (CPCB)
        """)
    
    # Model performance section
    with st.expander("📈 Model Performance"):
        st.markdown("### Model Accuracy Metrics")
        
        # Load and display model performance if available
        results_dir = Path(config.get('paths.results_dir'))
        
        for category in selected_categories:
            report_path = results_dir / f'{category}_classification_report.json'
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                st.markdown(f"**{category.title()} Model:**")
                
                # Create metrics columns
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    st.metric("Accuracy", f"{report['accuracy']:.3f}")
                
                with metric_cols[1]:
                    st.metric("Macro F1-Score", f"{report['macro avg']['f1-score']:.3f}")
                
                with metric_cols[2]:
                    st.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.3f}")

if __name__ == "__main__":
    main()
