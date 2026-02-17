"""
Feature pipeline: load raw data, compute time-based and derived features, build targets.
Stores results in Feature Store (MongoDB).
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RAW_DIR,
    DEFAULT_CITY,
    DEFAULT_LAT,
    DEFAULT_LON,
    FORECAST_DAYS,
    TIMEZONE,
)
from scripts.fetch_raw_data import fetch_raw_for_date
from scripts.db import save_features
from scripts.data_cleaning import clean_features_df


def pm25_to_us_aqi(pm25_ugm3: float) -> float:
    """
    Convert PM2.5 (µg/m³) to US EPA AQI (0-500). Uses standard breakpoints.
    Returns float in 0-500; None if input invalid.
    """
    if pm25_ugm3 is None or (isinstance(pm25_ugm3, float) and (np.isnan(pm25_ugm3) or pm25_ugm3 < 0)):
        return None
    cp = float(pm25_ugm3)
    # EPA breakpoints (PM2.5 low, PM2.5 high, AQI low, AQI high)
    breakpoints = [
        (0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 425.4, 401, 500),
    ]
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= cp <= bp_hi:
            return round((i_hi - i_lo) / (bp_hi - bp_lo) * (cp - bp_lo) + i_lo, 1)
    return 500.0 if cp > 425.4 else 0.0


def load_raw_file(date: str, raw_dir: Path = None) -> dict:
    """Load raw JSON for a date. Returns None if file missing."""
    raw_dir = raw_dir or RAW_DIR
    path = raw_dir / f"raw_{date}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_features_from_raw_openweather(raw: dict) -> pd.DataFrame:
    """
    Build feature rows from OpenWeather raw (AQI 1-5; no normalization).
    - Time converted to TIMEZONE (Asia/Karachi).
    - Time-based: hour, day_of_week, month, is_weekend
    - Weather from raw.weather; pollutants from air_quality.hourly
    - Derived: aqi_change_rate
    """
    hourly = raw.get("air_quality", {}).get("hourly", [])
    if not hourly:
        return pd.DataFrame()
    weather = raw.get("weather", {})
    date_str = raw.get("date", "")
    rows = []
    prev_aqi = None
    for i, item in enumerate(hourly):
        dt_str = item.get("datetime_iso") or item.get("timestamp")
        if not dt_str:
            continue
        try:
            dt = pd.to_datetime(dt_str)
            if getattr(dt, "tz", None) and str(TIMEZONE):
                try:
                    from zoneinfo import ZoneInfo
                    dt = dt.tz_convert(ZoneInfo(TIMEZONE))
                except Exception:
                    pass
        except Exception:
            continue
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        is_weekend = 1 if day_of_week >= 5 else 0
        aqi_val = item.get("aqi")  # 1-5 raw from OpenWeather
        if aqi_val is not None:
            try:
                aqi_val = float(aqi_val)
            except (TypeError, ValueError):
                aqi_val = None
        pm25 = item.get("pm2_5")
        if pm25 is not None:
            try:
                pm25 = float(pm25)
            except (TypeError, ValueError):
                pm25 = None
        # Use US AQI from PM2.5 when available so values match weather apps (~50-100); else fall back to 1-5
        us_aqi_val = pm25_to_us_aqi(pm25) if pm25 is not None else (aqi_val * 50.0 if aqi_val is not None else None)  # rough 1-5 -> 50-250 if no PM2.5
        if us_aqi_val is None and aqi_val is not None:
            us_aqi_val = aqi_val * 50.0  # 1->50, 2->100, 3->150, 4->200, 5->250
        aqi_change_rate = None
        if prev_aqi is not None and us_aqi_val is not None and prev_aqi != 0:
            aqi_change_rate = (us_aqi_val - prev_aqi) / max(prev_aqi, 1e-6)
        if us_aqi_val is not None:
            prev_aqi = us_aqi_val
        rows.append({
            "timestamp": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
            "datetime_iso": dt_str,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "temperature_max": weather.get("temp_max"),
            "temperature_min": weather.get("temp_min"),
            "precipitation": None,
            "humidity": weather.get("humidity"),
            "wind_speed": weather.get("wind_speed"),
            "pm2_5": item.get("pm2_5"),
            "pm10": item.get("pm10"),
            "ozone": item.get("o3"),
            "nitrogen_dioxide": item.get("no2"),
            "us_aqi": us_aqi_val,
            "aqi": aqi_val,
            "aqi_change_rate": aqi_change_rate,
            "date": date_str or dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else "",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["target_aqi_d1"] = None
    df["target_aqi_d2"] = None
    df["target_aqi_d3"] = None
    return clean_features_df(df)


def compute_features_from_raw(raw: dict) -> pd.DataFrame:
    """
    Build feature rows from raw weather + air quality payload.
    Supports OpenWeather (source=openweather) and Open-Meteo.
    - Time-based: hour, day_of_week, month, is_weekend (converted to TIMEZONE)
    - Weather: temp, humidity, precipitation, wind
    - Pollutants: pm2_5, pm10, ozone, no2, us_aqi (or aqi 1-5)
    - Derived: aqi_change_rate
    - Targets: aqi_d1, aqi_d2, aqi_d3 — filled where possible
    """
    if raw.get("source") == "openweather":
        return compute_features_from_raw_openweather(raw)
    aq = raw.get("air_quality", {}).get("hourly", {})
    weather = raw.get("weather", {})
    date_str = raw.get("date", "")

    if not aq or "time" not in aq:
        return pd.DataFrame()

    times = aq["time"]
    us_aqi = aq.get("us_aqi", [None] * len(times))
    pm25 = aq.get("pm2_5", [None] * len(times))
    pm10 = aq.get("pm10", [None] * len(times))
    ozone = aq.get("ozone", [None] * len(times))
    no2 = aq.get("nitrogen_dioxide", [None] * len(times))

    # Daily weather (Open-Meteo daily = one value per day)
    daily = weather.get("daily", {})
    daily_times = daily.get("time", [date_str])
    idx = 0
    if date_str in daily_times:
        idx = daily_times.index(date_str)
    temp_max = daily.get("temperature_2m_max", [None])[idx] if daily.get("temperature_2m_max") else None
    temp_min = daily.get("temperature_2m_min", [None])[idx] if daily.get("temperature_2m_min") else None
    precip = daily.get("precipitation_sum", [None])[idx] if daily.get("precipitation_sum") else None
    humidity = daily.get("relative_humidity_2m_mean", [None])[idx] if daily.get("relative_humidity_2m_mean") else None
    wind = daily.get("wind_speed_10m_max", [None])[idx] if daily.get("wind_speed_10m_max") else None

    rows = []
    prev_aqi = None
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        is_weekend = 1 if day_of_week >= 5 else 0

        aqi_val = us_aqi[i] if i < len(us_aqi) else None
        if aqi_val is not None and aqi_val is not np.nan:
            try:
                aqi_val = float(aqi_val)
            except (TypeError, ValueError):
                aqi_val = None

        aqi_change_rate = None
        if prev_aqi is not None and aqi_val is not None and prev_aqi != 0:
            aqi_change_rate = (aqi_val - prev_aqi) / max(prev_aqi, 1e-6)
        if aqi_val is not None:
            prev_aqi = aqi_val

        rows.append({
            "timestamp": t,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "temperature_max": temp_max,
            "temperature_min": temp_min,
            "precipitation": precip,
            "humidity": humidity,
            "wind_speed": wind,
            "pm2_5": pm25[i] if i < len(pm25) else None,
            "pm10": pm10[i] if i < len(pm10) else None,
            "ozone": ozone[i] if i < len(ozone) else None,
            "nitrogen_dioxide": no2[i] if i < len(no2) else None,
            "us_aqi": aqi_val,
            "aqi_change_rate": aqi_change_rate,
            "date": date_str,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Targets: for this pipeline we use same-day AQI as "current"; next-day targets
    # are filled during backfill when we have multiple days. Here we add placeholder columns.
    df["target_aqi_d1"] = None
    df["target_aqi_d2"] = None
    df["target_aqi_d3"] = None
    return clean_features_df(df)


def add_future_targets(features_df: pd.DataFrame, raw_future: dict) -> pd.DataFrame:
    """Add target AQI for next day from a future raw payload (used in backfill)."""
    aq = raw_future.get("air_quality", {}).get("hourly", {})
    if not aq or "us_aqi" not in aq:
        return features_df
    aqi_list = aq["us_aqi"]
    # Use daily mean or last value as target for that day
    valid = [float(x) for x in aqi_list if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not valid:
        return features_df
    target_val = np.nanmean(valid)
    if "target_aqi_d1" in features_df.columns:
        features_df["target_aqi_d1"] = target_val
    return features_df


def run_feature_pipeline_for_date(
    date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    city: str = DEFAULT_CITY,
    fetch_if_missing: bool = True,
    raw_dir: Path = None,
) -> pd.DataFrame:
    """
    Run feature pipeline for one date: load raw (or fetch), compute features, return DataFrame.
    Does NOT save to DB (caller can call save_features).
    """
    raw_dir = raw_dir or RAW_DIR
    raw = load_raw_file(date, raw_dir)
    if raw is None and fetch_if_missing:
        raw = fetch_raw_for_date(date, lat=lat, lon=lon, save_dir=raw_dir)
    if raw is None:
        return pd.DataFrame()
    df = compute_features_from_raw(raw)
    return df


def run_feature_pipeline(
    date: str = None,
    save_to_store: bool = True,
    city: str = DEFAULT_CITY,
) -> pd.DataFrame:
    """
    Full step: run for date (default today), optionally save to Feature Store.
    """
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    df = run_feature_pipeline_for_date(date, city=city)
    if df.empty:
        print(f"No features for {date}")
        return df
    if save_to_store:
        n = save_features(df, city=city)
        print(f"Saved {n} feature rows for {date} to Feature Store")
    return df


def main():
    date = datetime.utcnow().strftime("%Y-%m-%d")
    if len(sys.argv) > 1:
        date = sys.argv[1]
    run_feature_pipeline(date=date, save_to_store=True)
    print(f"Feature pipeline done for {date}")


if __name__ == "__main__":
    main()
