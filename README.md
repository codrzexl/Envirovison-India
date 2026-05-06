# 🌍 EnviroVision India

An environmental scene classification system using deep learning to analyze rainfall, heatwave, and air quality conditions from images.

## 📋 Project Overview

EnviroVision India is a machine learning application that can classify environmental conditions in three main categories:

- **🌧️ Rainfall**: Light, Moderate, Heavy
- **🔥 Heatwave**: Normal, Mild, Extreme  
- **🌬️ Air Quality**: Good, Moderate, Unhealthy

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd envirovision-india
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv env
   .\env\Scripts\Activate.ps1
   
   # Linux/Mac
   python -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies** (if needed)
   ```bash
   pip install pyyaml scikit-learn matplotlib pandas
   ```

## 🎯 Running the Application

### Option 1: Demo Application (Recommended for testing)

The demo application works without trained models and provides simulated predictions:

```bash
streamlit run app/demo_app.py --server.port 8502
```

**Access the demo at:** http://localhost:8502

### Option 2: Full Application

The full application requires trained models:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

**Access the full app at:** http://localhost:8501

## 📁 Project Structure

```
envirovision-india/
├── app/
│   ├── streamlit_app.py      # Main application
│   └── demo_app.py           # Demo application
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── rainfall/             # Rainfall dataset
│   ├── heatwave/             # Heatwave dataset
│   └── air_quality/          # Air quality dataset
├── models/
│   └── transfer_learning.py  # Model architecture
├── scripts/
│   └── train_models.py       # Training script
├── utils/
│   ├── config_loader.py      # Configuration loader
│   ├── data_utils.py         # Data preprocessing
│   └── grad_cam.py          # Grad-CAM implementation
├── training/
│   └── trainer.py            # Training utilities
├── logs/                     # Training logs
├── results/                  # Training results
└── requirements.txt          # Dependencies
```

## 🔧 Training Models

To train the models with your own data:

1. **Prepare your dataset** in the following structure:
   ```
   data/
   ├── rainfall/
   │   ├── Light/
   │   ├── Moderate/
   │   └── Heavy/
   ├── heatwave/
   │   ├── Normal/
   │   ├── Mild/
   │   └── Extreme/
   └── air_quality/
       ├── Good/
       ├── Moderate/
       └── Unhealthy/
   ```

2. **Train all models**:
   ```bash
   python scripts/train_models.py --category all
   ```

3. **Train specific category**:
   ```bash
   python scripts/train_models.py --category rainfall
   python scripts/train_models.py --category heatwave
   python scripts/train_models.py --category air_quality
   ```

## 🎨 Features

### Demo Application
- ✅ **Image Upload**: Upload images in PNG, JPG, JPEG formats
- ✅ **Simulated Predictions**: See how the system would classify images
- ✅ **Interactive Charts**: Visualize prediction probabilities
- ✅ **Multi-category Analysis**: Rainfall, Heatwave, and Air Quality
- ✅ **Responsive Design**: Works on desktop and mobile

### Full Application
- ✅ **Real Model Predictions**: Uses trained deep learning models
- ✅ **Grad-CAM Visualization**: See what the model focuses on
- ✅ **WebRTC Camera**: Real-time camera analysis
- ✅ **Batch Processing**: Process multiple images
- ✅ **Export Results**: Save analysis results

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error: No module named 'albumentations'**
   - **Solution**: The code has been modified to use TensorFlow's built-in augmentation instead
   - **Alternative**: Install albumentations manually: `pip install albumentations`

2. **Model Loading Error**
   - **Cause**: Trained models not found
   - **Solution**: Use the demo application or train models first

3. **Port Already in Use**
   - **Solution**: Use different port: `streamlit run app/demo_app.py --server.port 8503`

4. **Virtual Environment Issues**
   - **Solution**: Recreate virtual environment:
     ```bash
     deactivate
     rmdir /s env
     python -m venv env
     .\env\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```

### Dependencies Issues

If you encounter dependency conflicts:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📊 Model Architecture

The project uses transfer learning with the following architectures:

- **EfficientNet-B0**: Primary architecture for all categories
- **ResNet-50**: Alternative architecture option
- **Vision Transformer**: Experimental architecture

## 🔍 Usage Examples

### Using the Demo Application

1. **Start the demo**:
   ```bash
   streamlit run app/demo_app.py
   ```

2. **Upload an image** using the sidebar

3. **View results** in the interactive tabs:
   - Rainfall analysis
   - Heatwave analysis  
   - Air quality analysis

4. **Check the summary** table for all predictions

### Using the Full Application

1. **Train models first** (if not already trained):
   ```bash
   python scripts/train_models.py --category all
   ```

2. **Start the application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Use all features**:
   - Upload images
   - Use webcam
   - View Grad-CAM visualizations
   - Export results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- TensorFlow/Keras for deep learning
- Plotly for interactive visualizations
- OpenCV for image processing

---

**🌍 EnviroVision India - Making Environmental Analysis Accessible** 