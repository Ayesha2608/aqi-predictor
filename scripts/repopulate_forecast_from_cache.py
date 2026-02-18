import os
import json
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.db import save_features
from config.settings import DEFAULT_CITY, TIMEZONE

def repopulate_forecast():
    # Load the best available cache file that contains a forecast
    cache_file = Path("data/raw/raw_2026-02-17.json")
    if not cache_file.exists():
        print("Error: No cached raw file found at raw_2026-02-17.json")
        return

    with open(cache_file, "r") as f:
        raw = json.load(f)

    forecast_list = raw.get("forecast_list", [])
    if not forecast_list:
        print("Error: No forecast_list found in JSON.")
        return

    print(f"Processing {len(forecast_list)} forecast entries from {cache_file.name}")
    
    rows = []
    for item in forecast_list:
        dt_str = item.get("dt_txt") # Format: "2026-02-17 18:00:00" (usually UTC)
        if not dt_str:
            continue
        
        # Convert to Karachi time
        try:
            dt_utc = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            from zoneinfo import ZoneInfo
            dt_khi = dt_utc.astimezone(ZoneInfo(TIMEZONE))
        except Exception:
            # Fallback if zoneinfo missing
            dt_khi = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S") + timedelta(hours=5)

        main = item.get("main", {})
        wind = item.get("wind", {})
        
        row = {
            "timestamp": dt_khi.strftime("%Y-%m-%d %H:%M:%S+05:00"),
            "hour": dt_khi.hour,
            "day_of_week": dt_khi.weekday(),
            "month": dt_khi.month,
            "is_weekend": 1 if dt_khi.weekday() >= 5 else 0,
            "temperature_max": main.get("temp_max"),
            "temperature_min": main.get("temp_min"),
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "us_aqi": None, # Forecasted AQI is what we want the model to predict
            "date": dt_khi.strftime("%Y-%m-%d"),
            "city": DEFAULT_CITY
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        n = save_features(df, city=DEFAULT_CITY)
        print(f"--- SUCCESS: Injected {n} forecast weather rows for {DEFAULT_CITY} ---")
        print(f"Date range covered: {df['date'].min()} to {df['date'].max()}")
    else:
        print("No valid rows generated.")

if __name__ == "__main__":
    repopulate_forecast()
