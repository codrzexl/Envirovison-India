import google.generativeai as genai
from PIL import Image
import os

class GeminiService:
    """
    Service to interact with Google Gemini AI for image analysis.
    Provides detailed environmental and satellite image intelligence.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def analyze_environmental_image(self, image):
        """
        Analyze an environmental or satellite image and return 
        a structured intelligence report.
        """
        if not self.model:
            return "⚠️ Gemini API Key not configured. Please add your key in the sidebar."
        
        prompt = """
        You are an Environmental Intelligence Expert. Analyze this image (which could be a ground-level scene or a satellite view) and provide a detailed report on:
        1. **General Scene Description**: What is visible?
        2. **Rainfall Indicators**: Are there clouds, wet surfaces, or active rain visible? Estimate intensity.
        3. **Heat & Temperature Indicators**: Signs of dryness, sun intensity, or heat haze?
        4. **Air Quality Indicators**: Is there visible smog, dust, or clear visibility?
        5. **Satellite Specifics (if applicable)**: If this is a satellite image, identify vegetation health, water bodies, or urban density.
        6. **Environmental Risk Score**: On a scale of 1-10, how 'at risk' does this environment look (10 being extreme weather/pollution)?
        
        Format the output in professional Markdown.
        """
        
        try:
            # Ensure image is in RGB format for Gemini
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"❌ Error during AI analysis: {str(e)}"

    def is_configured(self):
        return self.api_key is not None and self.model is not None
