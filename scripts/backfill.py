"""
Historical backfill: run feature pipeline for a range of past dates.
Fills target_aqi_d1 (and optionally d2, d3) from subsequent days' AQI.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DIR, DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON
from scripts.fetch_raw_data import fetch_raw_for_date
from scripts.feature_pipeline import compute_features_from_raw, pm25_to_us_aqi
from scripts.db import save_features


def backfill_date_range(
    start_date: str,
    end_date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    city: str = DEFAULT_CITY,
    save_to_store: bool = True,
) -> pd.DataFrame:
    """
    Backfill features for start_date..end_date (inclusive).
    Fetches raw for each date, computes features, fills targets from next days, saves to store.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_dfs = []
    raw_by_date = {}

    # 1) Fetch raw for all dates
    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        try:
            raw = fetch_raw_for_date(date_str, lat=lat, lon=lon, save_dir=RAW_DIR)
            raw_by_date[date_str] = raw
        except Exception as e:
            print(f"Fetch failed {date_str}: {e}")
        d += timedelta(days=1)

    # 2) Build features per date and attach targets from future dates
    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        raw = raw_by_date.get(date_str)
        if raw is None:
            d += timedelta(days=1)
            continue
        df = compute_features_from_raw(raw)
        if df.empty:
            d += timedelta(days=1)
            continue

        # Target D+1, D+2, D+3: mean US AQI (0-500) of next days so predictions match weather app (~92)
        def _us_aqi_values_for_day(raw_day):
            aq = raw_day.get("air_quality", {}).get("hourly", [])
            if not aq:
                return []
            # OpenWeather: list of dicts with pm2_5 and aqi (1-5) -> convert to US AQI
            if isinstance(aq, list) and aq and isinstance(aq[0], dict):
                out = []
                for x in aq:
                    pm25 = x.get("pm2_5")
                    if pm25 is not None:
                        try:
                            u = pm25_to_us_aqi(float(pm25))
                            if u is not None:
                                out.append(u)
                                continue
                        except (TypeError, ValueError):
                            pass
                    a = x.get("aqi")
                    if a is not None:
                        try:
                            out.append(float(a) * 50.0)  # 1-5 -> ~50-250
                        except (TypeError, ValueError):
                            pass
                return out
            # Open-Meteo: us_aqi list already 0-500
            aqi_list = aq if isinstance(aq, list) else aq.get("us_aqi", [])
            return [float(x) for x in aqi_list if x is not None and not (isinstance(x, float) and np.isnan(x))]
        d1 = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        if d1 in raw_by_date:
            valid1 = _us_aqi_values_for_day(raw_by_date[d1])
            if valid1:
                df["target_aqi_d1"] = np.nanmean(valid1)
        d2 = (d + timedelta(days=2)).strftime("%Y-%m-%d")
        if d2 in raw_by_date:
            valid2 = _us_aqi_values_for_day(raw_by_date[d2])
            if valid2:
                df["target_aqi_d2"] = np.nanmean(valid2)
        d3 = (d + timedelta(days=3)).strftime("%Y-%m-%d")
        if d3 in raw_by_date:
            valid3 = _us_aqi_values_for_day(raw_by_date[d3])
            if valid3:
                df["target_aqi_d3"] = np.nanmean(valid3)

        all_dfs.append(df)
        if save_to_store:
            n = save_features(df, city=city)
            print(f"Backfill saved {n} rows for {date_str}")
        d += timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def main():
    # Default: last 14 days
    end = datetime.utcnow()
    start = end - timedelta(days=14)
    end_str = end.strftime("%Y-%m-%d")
    start_str = start.strftime("%Y-%m-%d")
    if len(sys.argv) >= 2:
        start_str = sys.argv[1]
    if len(sys.argv) >= 3:
        end_str = sys.argv[2]
    print(f"Backfilling {start_str} to {end_str}")
    backfill_date_range(start_str, end_str, save_to_store=True)
    print("Backfill complete.")


if __name__ == "__main__":
    main()
