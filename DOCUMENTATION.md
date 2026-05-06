# 🌍 EnviroVision India - Full Documentation

EnviroVision India is a professional-grade environmental intelligence dashboard. It combines Deep Learning (TensorFlow), Generative AI (Gemini 1.5 Flash), and real-time satellite-derived data to provide a 360° view of environmental conditions across India.

---

## 🚀 1. Getting Started

### Prerequisites
- Python 3.9 or higher
- Windows/Linux/Mac
- A free API Key from [Google AI Studio](https://aistudio.google.com/)

### Installation
1. **Clone/Open the project directory.**
2. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install advanced visualization & AI packages:**
   ```bash
   pip install streamlit-webrtc av google-generativeai plotly pydeck requests pandas
   ```

### Initial Model Setup
Since the project uses deep learning, you need trained models in the `models/` folder. For testing purposes, you can generate dummy models:
```bash
python create_dummy_models.py
```

---

## 🛠️ 2. Core Modules & Architecture

### `app/streamlit_app.py` (The Command Center)
The main entry point. It handles:
- **UI/UX**: Multi-column layouts, sidebars, and tabbed interfaces.
- **State Management**: Handling image uploads and real-time city selection.
- **Visualization Wiring**: Connecting API data to Plotly gauges and Pydeck maps.

### `utils/api_service.py` (The Data Engine)
Handles all communication with the Open-Meteo API.
- **Geocoding**: Converts city names to Lat/Lon coordinates.
- **Meteorological Logic**: Implements scientific formulas like the **Heat Index (Apparent Temperature)**.
- **Standardization**:
    - **AQI**: US EPA Standard (0-500 scale).
    - **Heatwaves/Rainfall**: India Meteorological Department (IMD) criteria.

### `utils/ai_service.py` (The Intelligence Layer)
Connects to **Google Gemini 1.5 Flash**.
- **Multimodal Analysis**: Processes images alongside a specialized "Environmental Expert" prompt.
- **Satellite Extraction**: Identifies specific remote-sensing features like vegetation health and urban heat islands.

---

## 🌫️ 3. Air Quality Intelligence (AQI)

The system provides a deep-dive into air health:
- **6 Pollutant Monitoring**: PM2.5, PM10, NO2, SO2, CO, and O3.
- **AQI Meter**: A real-time gauge color-coded from Green (Good) to Maroon (Hazardous).
- **Radar Charts**: A "chemical fingerprint" visualization for identifying pollution sources.
- **Health Advisory**: Professional medical advice generated dynamically based on the current AQI.

---

## 🌡️ 4. Heatwave & Thermal Monitoring

Designed for the Indian sub-continent's climate:
- **Feels-Like Temperature**: Combines humidity and temp for accurate heat stress monitoring.
- **Digital Thermometer**: Visual bullet-gauge highlighting danger zones.
- **Heatstroke Warnings**: Automated alerts based on IMD severe weather thresholds.

---

## 🌧️ 5. Rainfall & Flood Risk

Proactive monitoring for heavy precipitation:
- **Liquid Intensity Gauge**: Visualizes rainfall in mm.
- **Chance of Rain (PoP)**: Integrated 24-hour precipitation probability.
- **Flood Advisory**: Real-time alerts for urban flooding and water-logging risks.

---

## 🧠 6. Advanced AI Satellite Analysis

Users can get a deep-dive intelligence report for any image:
1. **Paste your Gemini Key** in the sidebar.
2. **Upload an image** (Ground or Satellite).
3. **Click "Run Deep Satellite Analysis"**.
4. The AI will generate a markdown report with:
   - **Environmental Risk Score (1-10)**
   - **Scene Context Identification**
   - **Satellite-specific ground truth estimates.**

---

## 🗺️ 7. Geospatial Visualization

The app includes a **3D Interactive Map** (powered by Pydeck).
- Marks the exact location of the meteorological station being used.
- Provides a spatial context for the environmental reports.

---

## ❓ 8. Troubleshooting

**Error: ModuleNotFoundError: No module named 'streamlit_webrtc'**
- Run: `pip install streamlit-webrtc`

**Error: UnicodeEncodeError on Windows**
- This usually happens due to emojis in the terminal. The updated `create_dummy_models.py` has emojis removed to fix this.

**Error: Model not found**
- Ensure you have run `python create_dummy_models.py` or placed your trained `.h5` files in the `models/` directory.

---

## 📝 9. License & Acknowledgements
- **Data Source**: Open-Meteo (open-meteo.com)
- **AI Core**: Google Generative AI (Gemini)
- **Design**: Built with Streamlit, Plotly, and Pydeck.

**🌍 EnviroVision India - Making Environmental Analysis Accessible**
