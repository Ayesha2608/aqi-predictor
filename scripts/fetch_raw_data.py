"""
Fetch raw weather and air quality data.
- If OPENWEATHER_API_KEY is set: use OpenWeather (AQI 1-5; no normalization needed).
- Else: use Open-Meteo (no API key). If using multiple APIs later, normalize in feature pipeline.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import requests

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    OPEN_METEO_BASE,
    OPEN_METEO_AIR_QUALITY,
    OPENWEATHER_API_KEY,
    USE_OPENMETEO_AQI,
    DEFAULT_LAT,
    DEFAULT_LON,
    RAW_DIR,
    DEFAULT_CITY,
)


def fetch_weather(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Fetch weather for a given range (or today)."""
    if start_date is None:
        start_date = datetime.utcnow().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = start_date
    url = (
        f"{OPEN_METEO_BASE}/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max&"
        f"start_date={start_date}&end_date={end_date}&timezone=auto"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_air_quality(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Fetch air quality (AQI and pollutants) from Open-Meteo Air Quality API."""
    if start_date is None:
        start_date = datetime.utcnow().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = start_date
    url = (
        f"{OPEN_METEO_AIR_QUALITY}?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        "hourly=us_aqi,pm10,pm2_5,ozone,nitrogen_dioxide"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _build_openmeteo_style_weather_from_openweather(ow_weather: dict, ow_forecast: dict, date: str) -> dict:
    """Build weather dict in Open-Meteo daily format from OpenWeather current + forecast."""
    daily = {"time": [date], "temperature_2m_max": [None], "temperature_2m_min": [None],
             "precipitation_sum": [0.0], "relative_humidity_2m_mean": [None], "wind_speed_10m_max": [None]}
    if ow_weather:
        m = ow_weather.get("main", {})
        daily["temperature_2m_max"][0] = m.get("temp_max") or m.get("temp")
        daily["temperature_2m_min"][0] = m.get("temp_min") or m.get("temp")
        daily["relative_humidity_2m_mean"][0] = m.get("humidity")
        w = ow_weather.get("wind", {})
        daily["wind_speed_10m_max"][0] = w.get("speed")
    if ow_forecast and ow_forecast.get("list"):
        temps = [x.get("main", {}).get("temp") for x in ow_forecast["list"] if x.get("main")]
        if temps:
            daily["temperature_2m_max"][0] = daily["temperature_2m_max"][0] or max(temps)
            daily["temperature_2m_min"][0] = daily["temperature_2m_min"][0] or min(temps)
    return {"daily": daily}


def fetch_raw_range(
    start_date: str,
    end_date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """Fetch weather + air quality for a range of dates as a single payload."""
    aq = fetch_air_quality(lat=lat, lon=lon, start_date=start_date, end_date=end_date)
    weather = fetch_weather(lat=lat, lon=lon, start_date=start_date, end_date=end_date)
    return {
        "start_date": start_date,
        "end_date": end_date,
        "city": DEFAULT_CITY,
        "air_quality": aq,
        "weather": weather,
    }


def fetch_raw_for_date(
    date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    save_dir: Path = None,
    save_to_disk: bool = True,
) -> dict:
    """
    Fetch weather + air quality for one date.
    When USE_OPENMETEO_AQI is true (default): AQI always from Open-Meteo (real US AQI 0-500 for Karachi).
    Weather from OpenWeather if key set, else Open-Meteo. This gives correct, comparable predictions.
    When USE_OPENMETEO_AQI is false: OpenWeather for both when key set (AQI 1-5).
    """
    save_dir = save_dir or RAW_DIR
    out = None

    # Prefer real US AQI from Open-Meteo (no key; CAMS model covers Karachi)
    if USE_OPENMETEO_AQI:
        aq = None
        try:
            aq = fetch_air_quality(lat=lat, lon=lon, start_date=date, end_date=date)
        except (requests.RequestException, OSError) as e:
            # Open-Meteo unreachable (DNS/network/firewall) -> fall back to OpenWeather if key set
            if OPENWEATHER_API_KEY:
                try:
                    from scripts.fetch_openweather import fetch_openweather_raw
                    out = fetch_openweather_raw(lat=lat, lon=lon, city=DEFAULT_CITY, api_key=OPENWEATHER_API_KEY)
                    out["date"] = date
                    if save_to_disk:
                        save_path = save_dir / f"raw_{date}.json"
                        with open(save_path, "w") as f:
                            json.dump(out, f, indent=2, default=str)
                    return out
                except Exception:
                    pass
            raise RuntimeError(
                f"Air quality fetch failed for {date}: {e}. "
                "Open-Meteo (air-quality.api.open-meteo.com) is unreachable (DNS/network). "
                "Set OPENWEATHER_API_KEY in .env to use OpenWeather as fallback, or set USE_OPENMETEO_AQI=false."
            ) from e
        if OPENWEATHER_API_KEY:
            try:
                from scripts.fetch_openweather import (
                    fetch_openweather_weather,
                    fetch_openweather_forecast,
                )
                ow_weather = fetch_openweather_weather(lat=lat, lon=lon, api_key=OPENWEATHER_API_KEY)
                ow_forecast = fetch_openweather_forecast(lat=lat, lon=lon, api_key=OPENWEATHER_API_KEY)
                weather = _build_openmeteo_style_weather_from_openweather(ow_weather, ow_forecast, date)
            except Exception as e:
                resp = getattr(e, "response", None)
                if resp is not None and getattr(resp, "status_code", None) == 401:
                    weather = fetch_weather(lat=lat, lon=lon, date=date)
                else:
                    raise RuntimeError(f"Weather fetch failed for {date}: {e}") from e
        else:
            weather = fetch_weather(lat=lat, lon=lon, date=date)
        out = {
            "date": date,
            "city": DEFAULT_CITY,
            "lat": lat,
            "lon": lon,
            "weather": weather,
            "air_quality": aq,
            "fetched_at": datetime.utcnow().isoformat(),
        }
    else:
        # Legacy: OpenWeather for everything when key set
        if OPENWEATHER_API_KEY:
            try:
                from scripts.fetch_openweather import fetch_openweather_raw
                out = fetch_openweather_raw(lat=lat, lon=lon, city=DEFAULT_CITY, api_key=OPENWEATHER_API_KEY)
                out["date"] = date
            except Exception as e:
                resp = getattr(e, "response", None)
                if resp is not None and getattr(resp, "status_code", None) == 401:
                    out = None
                else:
                    raise RuntimeError(f"OpenWeather fetch failed for {date}: {e}") from e
        if out is None:
            try:
                weather = fetch_weather(lat=lat, lon=lon, date=date)
            except requests.RequestException as e:
                raise RuntimeError(f"Open-Meteo weather fetch failed for {date}: {e}") from e
            try:
                aq = fetch_air_quality(lat=lat, lon=lon, start_date=date, end_date=date)
            except (requests.RequestException, OSError) as e:
                raise RuntimeError(f"Air quality fetch failed for {date}: {e}") from e
            out = {
                "date": date,
                "city": DEFAULT_CITY,
                "lat": lat,
                "lon": lon,
                "weather": weather,
                "air_quality": aq,
                "fetched_at": datetime.utcnow().isoformat(),
            }

    if save_to_disk:
        save_path = save_dir / f"raw_{date}.json"
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
    return out


def main():
    """Fetch today's data by default; optional date from argv."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    if len(sys.argv) > 1:
        date = sys.argv[1]
    data = fetch_raw_for_date(date)
    print(f"Fetched and saved raw data for {date} ({DEFAULT_CITY})")
    return data


if __name__ == "__main__":
    main()
