import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import config

# Page configuration
st.set_page_config(
    page_title="EnviroVision India - Demo",
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

def preprocess_image(image):
    """Preprocess image for demo"""
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize
    image = image.astype(np.float32) / 255.0
    return image

def generate_demo_predictions(image):
    """Generate demo predictions for the image"""
    # Simulate predictions for demo purposes
    np.random.seed(hash(str(image.shape)) % 2**32)
    
    categories = {
        'rainfall': ['Light', 'Moderate', 'Heavy'],
        'heatwave': ['Normal', 'Mild', 'Extreme'],
        'air_quality': ['Good', 'Moderate', 'Unhealthy']
    }
    
    predictions = {}
    for category, classes in categories.items():
        # Generate random probabilities that sum to 1
        probs = np.random.dirichlet(np.ones(len(classes)))
        predictions[category] = {
            'classes': classes,
            'probabilities': probs.tolist(),
            'predicted_class': classes[np.argmax(probs)],
            'confidence': float(np.max(probs))
        }
    
    return predictions

def create_prediction_chart(predictions, category):
    """Create a bar chart for predictions"""
    data = predictions[category]
    
    fig = go.Figure(data=[
        go.Bar(
            x=data['classes'],
            y=data['probabilities'],
            marker_color=['#28a745' if i == np.argmax(data['probabilities']) else '#6c757d' 
                        for i in range(len(data['classes']))]
        )
    ])
    
    fig.update_layout(
        title=f"{category.replace('_', ' ').title()} Predictions",
        xaxis_title="Classes",
        yaxis_title="Probability",
        height=400
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🌍 EnviroVision India - Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3>📋 About This Demo</h3>
        <p>This is a demonstration of the EnviroVision India environmental scene classification system. 
        The application can classify environmental conditions in three categories:</p>
        <ul>
            <li><strong>Rainfall:</strong> Light, Moderate, Heavy</li>
            <li><strong>Heatwave:</strong> Normal, Mild, Extreme</li>
            <li><strong>Air Quality:</strong> Good, Moderate, Unhealthy</li>
        </ul>
        <p><em>Note: This demo uses simulated predictions. For real predictions, trained models are required.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Demo Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg']
    )
    
    # Demo mode toggle
    demo_mode = st.sidebar.checkbox("Enable Demo Mode", value=True)
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert to numpy array for processing
        image_array = np.array(image)
        
        if demo_mode:
            # Generate demo predictions
            predictions = generate_demo_predictions(image_array)
            
            # Display predictions
            st.markdown('<h2 class="category-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create tabs for each category
            tab1, tab2, tab3 = st.tabs(["🌧️ Rainfall", "🔥 Heatwave", "🌬️ Air Quality"])
            
            with tab1:
                st.markdown('<h3 class="category-header">Rainfall Analysis</h3>', unsafe_allow_html=True)
                pred = predictions['rainfall']
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>Predicted: <span class="confidence-high">{pred['predicted_class']}</span></h4>
                    <p>Confidence: <span class="confidence-high">{pred['confidence']:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create chart
                fig = create_prediction_chart(predictions, 'rainfall')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown('<h3 class="category-header">Heatwave Analysis</h3>', unsafe_allow_html=True)
                pred = predictions['heatwave']
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>Predicted: <span class="confidence-high">{pred['predicted_class']}</span></h4>
                    <p>Confidence: <span class="confidence-high">{pred['confidence']:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create chart
                fig = create_prediction_chart(predictions, 'heatwave')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown('<h3 class="category-header">Air Quality Analysis</h3>', unsafe_allow_html=True)
                pred = predictions['air_quality']
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>Predicted: <span class="confidence-high">{pred['predicted_class']}</span></h4>
                    <p>Confidence: <span class="confidence-high">{pred['confidence']:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create chart
                fig = create_prediction_chart(predictions, 'air_quality')
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.markdown('<h3 class="category-header">📊 Summary</h3>', unsafe_allow_html=True)
            
            summary_data = []
            for category, pred in predictions.items():
                summary_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Prediction': pred['predicted_class'],
                    'Confidence': f"{pred['confidence']:.2%}"
                })
            
            import pandas as pd
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
        else:
            st.warning("⚠️ Demo mode is disabled. Please enable demo mode to see predictions.")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>📸 Upload an Image</h2>
            <p>Please upload an image using the sidebar to start the analysis.</p>
            <p><em>Supported formats: PNG, JPG, JPEG</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌍 EnviroVision India - Environmental Scene Classification Demo</p>
        <p>Built with Streamlit, TensorFlow, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 