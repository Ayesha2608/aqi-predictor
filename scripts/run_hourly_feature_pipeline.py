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
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
def run_hourly():
    """
    Fetch current/latest data from API → extract features → clean → save to Feature Store only.
    Fetches today + next 3 days to ensure forecasts are available.
    """
    total_saved = 0
    # Fetch today + next 5 days (6 days total buffer)
    for i in range(6):
        dt = datetime.now(timezone.utc) + timedelta(days=i)
        date = dt.strftime("%Y-%m-%d")
        try:
            raw = fetch_raw_for_date(
                date,
                lat=DEFAULT_LAT,
                lon=DEFAULT_LON,
                save_to_disk=False,
            )
            df = compute_features_from_raw(raw)
            if not df.empty:
                n = save_features(df, city=DEFAULT_CITY)
                total_saved += n
                print(f"Hourly: saved {n} feature rows for {date} ({DEFAULT_CITY}) to Feature Store.")
        except Exception as e:
            print(f"Hourly: failed for {date}: {e}")
    return total_saved


def main():
    run_hourly()


if __name__ == "__main__":
    main()
