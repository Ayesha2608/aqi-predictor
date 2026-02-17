"""
Fetch raw weather and air quality from OpenWeather API (AQI 1-5 scale).
Used when OPENWEATHER_API_KEY is set. No normalization needed for single-API use.
"""
import sys
from pathlib import Path
from datetime import datetime
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    OPENWEATHER_WEATHER,
    OPENWEATHER_FORECAST,
    OPENWEATHER_AIR_POLLUTION,
    OPENWEATHER_AIR_FORECAST,
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_CITY,
    TIMEZONE,
)


def _dt_from_unix(ts: int, tz_name: str = TIMEZONE) -> datetime:
    """Convert Unix timestamp to datetime in given timezone (e.g. Asia/Karachi)."""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(ts, tz=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        from datetime import timezone
        return datetime.fromtimestamp(ts, tz=timezone.utc)


def fetch_openweather_weather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, api_key: str = None) -> dict:
    """Fetch current weather. Requires api_key."""
    if not api_key:
        return {}
    url = f"{OPENWEATHER_WEATHER}?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_openweather_forecast(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, api_key: str = None) -> dict:
    """Fetch 5-day weather forecast (3h steps)."""
    if not api_key:
        return {}
    url = f"{OPENWEATHER_FORECAST}?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_openweather_air_pollution(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, api_key: str = None) -> dict:
    """Fetch current air pollution (AQI 1-5 + components)."""
    if not api_key:
        return {}
    url = f"{OPENWEATHER_AIR_POLLUTION}?lat={lat}&lon={lon}&appid={api_key}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_openweather_air_forecast(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, api_key: str = None) -> dict:
    """Fetch air pollution forecast (hourly, ~4 days)."""
    if not api_key:
        return {}
    url = f"{OPENWEATHER_AIR_FORECAST}?lat={lat}&lon={lon}&appid={api_key}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_openweather_raw(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    city: str = DEFAULT_CITY,
    api_key: str = None,
) -> dict:
    """
    Fetch weather + air (current + forecast) from OpenWeather.
    Returns unified raw payload for feature pipeline (no CSV; in-memory only).
    AQI is 1-5 scale; no normalization when using only OpenWeather.
    """
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY required for OpenWeather fetch")
    weather_now = fetch_openweather_weather(lat=lat, lon=lon, api_key=api_key)
    forecast = fetch_openweather_forecast(lat=lat, lon=lon, api_key=api_key)
    air_now = fetch_openweather_air_pollution(lat=lat, lon=lon, api_key=api_key)
    air_forecast = fetch_openweather_air_forecast(lat=lat, lon=lon, api_key=api_key)

    # Build hourly air list (current + forecast) in Asia/Karachi
    hourly_air = []
    if air_now.get("list"):
        for item in air_now["list"]:
            dt = _dt_from_unix(item["dt"])
            main = item.get("main", {})
            comp = item.get("components", {})
            hourly_air.append({
                "datetime_iso": dt.isoformat(),
                "dt": item["dt"],
                "aqi": main.get("aqi"),  # 1-5
                "pm2_5": comp.get("pm2_5"),
                "pm10": comp.get("pm10"),
                "o3": comp.get("o3"),
                "no2": comp.get("no2"),
                "so2": comp.get("so2"),
                "co": comp.get("co"),
            })
    if air_forecast.get("list"):
        for item in air_forecast["list"]:
            dt = _dt_from_unix(item["dt"])
            main = item.get("main", {})
            comp = item.get("components", {})
            hourly_air.append({
                "datetime_iso": dt.isoformat(),
                "dt": item["dt"],
                "aqi": main.get("aqi"),
                "pm2_5": comp.get("pm2_5"),
                "pm10": comp.get("pm10"),
                "o3": comp.get("o3"),
                "no2": comp.get("no2"),
                "so2": comp.get("so2"),
                "co": comp.get("co"),
            })

    # Daily weather from current (and repeat from forecast for alignment)
    daily_weather = {}
    if weather_now:
        daily_weather["temp_max"] = weather_now.get("main", {}).get("temp_max") or weather_now.get("main", {}).get("temp")
        daily_weather["temp_min"] = weather_now.get("main", {}).get("temp_min") or weather_now.get("main", {}).get("temp")
        daily_weather["humidity"] = weather_now.get("main", {}).get("humidity")
        daily_weather["pressure"] = weather_now.get("main", {}).get("pressure")
        daily_weather["wind_speed"] = weather_now.get("wind", {}).get("speed")
    if forecast.get("list"):
        temps = [x.get("main", {}).get("temp") for x in forecast["list"] if x.get("main")]
        if temps:
            daily_weather["temp_max"] = daily_weather.get("temp_max") or max(temps)
            daily_weather["temp_min"] = daily_weather.get("temp_min") or min(temps)
        hum = [x.get("main", {}).get("humidity") for x in forecast["list"] if x.get("main", {}).get("humidity") is not None]
        if hum:
            daily_weather["humidity"] = daily_weather.get("humidity") or (sum(hum) / len(hum))
        ws = [x.get("wind", {}).get("speed") for x in forecast["list"] if x.get("wind", {}).get("speed") is not None]
        if ws:
            daily_weather["wind_speed"] = daily_weather.get("wind_speed") or (sum(ws) / len(ws))

    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "date": today,
        "city": city,
        "lat": lat,
        "lon": lon,
        "source": "openweather",
        "aqi_scale": "1-5",
        "weather": daily_weather,
        "forecast_list": forecast.get("list", []),
        "air_quality": {"hourly": hourly_air},
        "fetched_at": datetime.utcnow().isoformat(),
    }
