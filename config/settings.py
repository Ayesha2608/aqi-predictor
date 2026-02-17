"""
Configuration for AQI Predictor pipeline.
Set env vars or create .env (see .env.example).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Ensure dirs exist
for d in (DATA_DIR, RAW_DIR, MODELS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# City: Karachi by default (next 3 days AQI prediction)
DEFAULT_CITY = os.getenv("AQI_CITY", "Karachi")
DEFAULT_LAT = float(os.getenv("AQI_LAT", "24.8607"))
DEFAULT_LON = float(os.getenv("AQI_LON", "67.0011"))
# Timezone for all timestamps (convert API times to this)
TIMEZONE = os.getenv("AQI_TIMEZONE", "Asia/Karachi")

# MongoDB (Feature Store + Model Registry)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "aqi_predictor")
FEATURES_COLLECTION = "aqi_features"
MODELS_COLLECTION = "models"
ALERTS_COLLECTION = "alerts"

# APIs: Use Open-Meteo for AQI (real US AQI 0-500 for Karachi). OpenWeather optional for weather only.
USE_OPENMETEO_AQI = os.getenv("USE_OPENMETEO_AQI", "true").lower() in ("true", "1", "yes")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")

# OpenWeather (AQI 1-5 scale; no normalization needed if using only OpenWeather)
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_WEATHER = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
OPENWEATHER_AIR_POLLUTION = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_AIR_FORECAST = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

# Open-Meteo (fallback when no OpenWeather key; no key required)
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
OPEN_METEO_AIR_QUALITY = "https://air-quality.api.open-meteo.com/v1/air-quality"

# Forecast horizons (days)
FORECAST_DAYS = 3

# Alert thresholds: when using Open-Meteo AQI we use US scale (0-500)
AQI_SCALE_1_5 = os.getenv("AQI_SCALE_1_5", "false" if USE_OPENMETEO_AQI else "true").lower() in ("true", "1", "yes")
AQI_HAZARDOUS_THRESHOLD = 5 if AQI_SCALE_1_5 else 200   # 5 (1-5 scale) or 200 (US)
AQI_UNHEALTHY_THRESHOLD = 4 if AQI_SCALE_1_5 else 150   # 4 (1-5 scale) or 150 (US)

# Calibration: reference AQI for Karachi (Moderate ~96) to correct for API discrepancies
# If stored AQI is way off from this, apply scaling to bring predictions into realistic range
KARACHI_REFERENCE_AQI = float(os.getenv("KARACHI_REFERENCE_AQI", "96.0"))  # Typical Moderate AQI for Karachi
AQI_CALIBRATION_ENABLED = os.getenv("AQI_CALIBRATION_ENABLED", "true").lower() in ("true", "1", "yes")
