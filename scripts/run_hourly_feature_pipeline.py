"""
HOURLY script: run every hour.
Step 1: Fetch from API (OpenWeather if key set, else Open-Meteo).
Step 2: Extract features from raw response.
Step 3: Store in Feature Store only.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DEFAULT_CITY,
    DEFAULT_LAT,
    DEFAULT_LON,
)
from scripts.db import save_features
from scripts.feature_pipeline import compute_features_from_raw
from scripts.fetch_raw_data import fetch_raw_range

def run_hourly():
    """
    Fetch current/latest data window (Today + next 3 days) → extract features → save to Feature Store.
    """
    # Fetch Today - 1 through Today + 4 (6 day window) to ensure all timezones and 3-day forecasts are covered
    start_dt = datetime.now(timezone.utc) - timedelta(days=1)
    end_dt = start_dt + timedelta(days=5)
    
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    
    try:
        raw = fetch_raw_range(start_date, end_date, lat=DEFAULT_LAT, lon=DEFAULT_LON)
        df = compute_features_from_raw(raw)
        if not df.empty:
            n = save_features(df, city=DEFAULT_CITY)
            print(f"Hourly Window: saved {n} feature rows for range {start_date} to {end_date} ({DEFAULT_CITY}).")
            return n
    except Exception as e:
        print(f"Hourly Window: failed for range {start_date} - {end_date}: {e}")
    return 0

def main():
    run_hourly()

if __name__ == "__main__":
    main()
