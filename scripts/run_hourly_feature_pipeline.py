"""
HOURLY script: run every hour.
Step 1: Fetch from API (OpenWeather if key set, else Open-Meteo).
Step 2: Extract features from raw response.
Step 3: Clean data (timezone Asia/Karachi, drop duplicates, fill missing).
Step 4: Store in Feature Store only (no CSV, no raw file).
Training runs on this live hourly data (full day) when daily training runs.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON
from scripts.fetch_raw_data import fetch_raw_for_date
from scripts.feature_pipeline import compute_features_from_raw
from scripts.db import save_features


def run_hourly():
    """
    Fetch current/latest data from API → extract features → clean → save to Feature Store only.
    No CSV. No raw file (save_to_disk=False).
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = fetch_raw_for_date(
        date,
        lat=DEFAULT_LAT,
        lon=DEFAULT_LON,
        save_to_disk=False,
    )
    df = compute_features_from_raw(raw)
    if df.empty:
        print("Hourly: no features computed.")
        return 0
    n = save_features(df, city=DEFAULT_CITY)
    print(f"Hourly: saved {n} feature rows for {date} ({DEFAULT_CITY}) to Feature Store.")
    return n


def main():
    run_hourly()


if __name__ == "__main__":
    main()
