import requests
import pandas as pd
from datetime import datetime

class EnvironmentalAPI:
    """
    Service to fetch real-time environmental data using Open-Meteo API.
    Provides data for Temperature (Heatwave), Precipitation (Rainfall), and Air Quality.
    """
    
    BASE_URL_WEATHER = "https://api.open-meteo.com/v1/forecast"
    BASE_URL_AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"

    @staticmethod
    def get_weather_data(lat, lon):
        """Fetch current weather, rainfall, and temperature"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m", 
                "precipitation", "weather_code", "apparent_temperature"
            ],
            "hourly": ["temperature_2m", "precipitation", "precipitation_probability"],
            "timezone": "auto",
            "forecast_days": 1
        }
        try:
            response = requests.get(EnvironmentalAPI.BASE_URL_WEATHER, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

    @staticmethod
    def get_air_quality_data(lat, lon):
        """Fetch real-time Air Quality Index (AQI) and pollutants"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "european_aqi", "us_aqi", "pm2_5", "pm10", 
                "nitrogen_dioxide", "sulphur_dioxide", 
                "carbon_monoxide", "ozone", "dust"
            ],
            "hourly": ["pm2_5", "pm10", "us_aqi"],
            "timezone": "auto",
            "forecast_days": 1
        }
        try:
            response = requests.get(EnvironmentalAPI.BASE_URL_AIR_QUALITY, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching air quality data: {e}")
            return None

    @staticmethod
    def get_location_coordinates(city_name):
        """Get coordinates for a city using Open-Meteo Geocoding API"""
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "lat": result["latitude"],
                    "lon": result["longitude"],
                    "name": result["name"],
                    "country": result.get("country", ""),
                    "admin1": result.get("admin1", "")
                }
            return None
        except Exception as e:
            print(f"Error geocoding city: {e}")
            return None

    @staticmethod
    def classify_from_api(weather_data, aq_data):
        """
        Convert raw API values into classification labels 
        comparable to the AI model's output.
        """
        results = {
            "rainfall": {"label": "Unknown", "value": 0, "unit": "mm"},
            "heatwave": {"label": "Unknown", "value": 0, "unit": "°C"},
            "air_quality": {"label": "Unknown", "value": 0, "unit": "AQI"},
            "trends": {"times": [], "temp": [], "precip": []}
        }

        if weather_data and "current" in weather_data:
            # Rainfall Classification & Flood Risk
            precip = weather_data["current"]["precipitation"]
            pop = 0
            if "hourly" in weather_data and "precipitation_probability" in weather_data["hourly"]:
                pop = weather_data["hourly"]["precipitation_probability"][0]
                
            results["rainfall"]["value"] = precip
            results["rainfall"]["pop"] = pop
            
            # IMD Intensity Criteria
            if precip == 0:
                results["rainfall"]["label"] = "No Rain"
                results["rainfall"]["recommendation"] = "No precipitation recorded. Perfect for outdoor activities."
                results["rainfall"]["color"] = "gray"
            elif precip < 2.5:
                results["rainfall"]["label"] = "Light Rain"
                results["rainfall"]["recommendation"] = "Slight drizzle. Carry an umbrella for convenience."
                results["rainfall"]["color"] = "lightblue"
            elif precip < 7.5:
                results["rainfall"]["label"] = "Moderate Rain"
                results["rainfall"]["recommendation"] = "Steady rain. Avoid water-logged areas. Possible traffic delays."
                results["rainfall"]["color"] = "blue"
            elif precip < 35.5:
                results["rainfall"]["label"] = "Heavy Rain"
                results["rainfall"]["recommendation"] = "Intense downpour. High risk of urban flooding. Stay indoors if possible."
                results["rainfall"]["color"] = "darkblue"
            else:
                results["rainfall"]["label"] = "Extreme Torrential Rain"
                results["rainfall"]["recommendation"] = "DANGER: Extreme rain. Severe flood risk. Evacuate low-lying areas."
                results["rainfall"]["color"] = "navy"

            # Heatwave Classification & Heat Index
            temp = weather_data["current"]["temperature_2m"]
            humidity = weather_data["current"]["relative_humidity_2m"]
            
            # Use apparent temperature (Feels Like)
            # Open-Meteo provides this, but we can also estimate it if needed
            feels_like = temp + (0.5 * (humidity - 50) / 10) if temp > 25 else temp
            
            results["heatwave"]["value"] = temp
            results["heatwave"]["feels_like"] = round(feels_like, 1)
            results["heatwave"]["humidity"] = humidity
            
            # IMD Criteria (Simplified)
            if temp < 38:
                results["heatwave"]["label"] = "Normal"
                results["heatwave"]["recommendation"] = "Temperature is within normal range for human comfort."
                results["heatwave"]["color"] = "blue"
            elif temp < 42:
                results["heatwave"]["label"] = "Mild Heatwave"
                results["heatwave"]["recommendation"] = "Stay hydrated and avoid prolonged sun exposure."
                results["heatwave"]["color"] = "orange"
            elif temp < 45:
                results["heatwave"]["label"] = "Severe Heatwave"
                results["heatwave"]["recommendation"] = "Extreme caution required. Stay indoors in a cool environment."
                results["heatwave"]["color"] = "red"
            else:
                results["heatwave"]["label"] = "Extreme Heatwave / Heatstroke Risk"
                results["heatwave"]["recommendation"] = "High risk of heatstroke. Do not go outside. Seek cooling immediately."
                results["heatwave"]["color"] = "darkred"
            
            # Trend Data
            if "hourly" in weather_data:
                results["trends"]["times"] = weather_data["hourly"]["time"]
                results["trends"]["temp"] = weather_data["hourly"]["temperature_2m"]
                results["trends"]["precip"] = weather_data["hourly"]["precipitation"]
            
            # Weather Code Mapping (WMO)
            wmo_codes = {
                0: ("Clear Sky", "☀️"), 1: ("Partly Cloudy", "🌤️"), 2: ("Partly Cloudy", "⛅"), 3: ("Overcast", "☁️"),
                45: ("Foggy", "🌫️"), 48: ("Foggy", "🌫️"), 51: ("Light Drizzle", "🌦️"), 53: ("Drizzle", "🌧️"),
                55: ("Heavy Drizzle", "🌧️"), 61: ("Light Rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy Rain", "⛈️"),
                80: ("Rain Showers", "🌦️"), 81: ("Rain Showers", "🌦️"), 82: ("Violent Rain Showers", "⛈️"),
                95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm", "⛈️"), 99: ("Heavy Thunderstorm", "⛈️")
            }
            code = weather_data["current"].get("weather_code", 0)
            status, emoji = wmo_codes.get(code, ("Unknown", "❓"))
            results["weather_status"] = status
            results["weather_emoji"] = emoji

        if aq_data and "current" in aq_data:
            # Air Quality Classification (US AQI Scale)
            aqi = aq_data["current"]["us_aqi"]
            results["air_quality"]["value"] = aqi
            
            # Detailed Pollutants
            results["air_quality"]["pollutants"] = {
                "PM2.5": aq_data["current"]["pm2_5"],
                "PM10": aq_data["current"]["pm10"],
                "NO2": aq_data["current"]["nitrogen_dioxide"],
                "SO2": aq_data["current"]["sulphur_dioxide"],
                "CO": aq_data["current"]["carbon_monoxide"],
                "O3": aq_data["current"]["ozone"]
            }

            # Health Recommendations
            if aqi <= 50:
                results["air_quality"]["label"] = "Good"
                results["air_quality"]["recommendation"] = "Air quality is satisfactory. Enjoy outdoor activities."
                results["air_quality"]["color"] = "green"
                results["air_quality"]["risk_val"] = 0
            elif aqi <= 100:
                results["air_quality"]["label"] = "Moderate"
                results["air_quality"]["recommendation"] = "Sensitive people should limit prolonged outdoor exertion."
                results["air_quality"]["color"] = "yellow"
                results["air_quality"]["risk_val"] = 2
            elif aqi <= 150:
                results["air_quality"]["label"] = "Unhealthy for Sensitive Groups"
                results["air_quality"]["recommendation"] = "Members of sensitive groups should limit outdoor exertion."
                results["air_quality"]["color"] = "orange"
                results["air_quality"]["risk_val"] = 5
            elif aqi <= 200:
                results["air_quality"]["label"] = "Unhealthy"
                results["air_quality"]["recommendation"] = "Everyone should limit prolonged outdoor exertion."
                results["air_quality"]["color"] = "red"
                results["air_quality"]["risk_val"] = 8
            else:
                results["air_quality"]["label"] = "Very Unhealthy / Hazardous"
                results["air_quality"]["recommendation"] = "Health alert: everyone may experience more serious health effects."
                results["air_quality"]["color"] = "purple"
                results["air_quality"]["risk_val"] = 10
                
            # AQI Trends
            if "hourly" in aq_data:
                results["trends"]["aqi_times"] = aq_data["hourly"]["time"]
                results["trends"]["aqi_values"] = aq_data["hourly"]["us_aqi"]

        # Calculate Overall Risk Score (0-10)
        risk_components = []
        if "rainfall" in results:
            rain_val = results["rainfall"]["value"]
            risk_components.append(min(10, rain_val / 5)) # 50mm rain = 10 risk
        if "heatwave" in results:
            temp = results["heatwave"]["value"]
            risk_components.append(max(0, min(10, (temp - 30) / 1.5))) # 45C = 10 risk
        if "air_quality" in results:
            risk_components.append(results["air_quality"].get("risk_val", 0))
            
        results["overall_risk"] = round(sum(risk_components) / len(risk_components), 1) if risk_components else 0

        return results
