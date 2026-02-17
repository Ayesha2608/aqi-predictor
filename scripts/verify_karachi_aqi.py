"""
Verify Karachi AQI: fetch current US AQI from Open-Meteo (same source as pipeline)
and compare with Feature Store latest and model predictions.
Run: python scripts/verify_karachi_aqi.py
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON, USE_OPENMETEO_AQI
from scripts.fetch_raw_data import fetch_air_quality
from scripts.db import get_latest_features
from scripts.model_loader import load_model_for_day
import numpy as np
import pandas as pd


def get_latest_feature_row():
    df = get_latest_features(city=DEFAULT_CITY, n_days=1)
    if df.empty:
        return None
    return df.iloc[-1]


def main():
    print(f"=== Karachi AQI verification ({DEFAULT_CITY}) ===\n")
    print(f"USE_OPENMETEO_AQI = {USE_OPENMETEO_AQI} (pipeline uses Open-Meteo US AQI when True)\n")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        aq = fetch_air_quality(lat=DEFAULT_LAT, lon=DEFAULT_LON, start_date=today, end_date=today)
    except Exception as e:
        print(f"Open-Meteo fetch failed: {e}")
        return
    hourly = aq.get("hourly", {})
    times = hourly.get("time", [])
    us_aqi_list = hourly.get("us_aqi", [])
    if not times or not us_aqi_list:
        print("No hourly AQI in Open-Meteo response.")
        return
    valid_aqi = [float(x) for x in us_aqi_list if x is not None]
    current_openmeteo = float(np.nanmean(valid_aqi)) if valid_aqi else None
    latest_ts = times[-1] if times else ""
    print(f"Open-Meteo (live) current US AQI: {current_openmeteo:.0f} (last hour: {latest_ts})")

    row = get_latest_feature_row()
    if row is not None:
        store_aqi = row.get("us_aqi")
        if store_aqi is not None and (isinstance(store_aqi, float) or isinstance(store_aqi, (int,))):
            print(f"Feature Store latest us_aqi:    {float(store_aqi):.0f}")
            if current_openmeteo is not None:
                diff = abs(float(store_aqi) - current_openmeteo)
                print(f"  -> Match: {'Yes' if diff < 20 else 'No'} (diff = {diff:.0f})")
        else:
            print("Feature Store latest: no us_aqi (run hourly pipeline for fresh data)")
    else:
        print("Feature Store: no data (run backfill + hourly pipeline)")

    # Predictions
    def _prepare_x(row, feature_names, as_dataframe=True):
        X = []
        for c in feature_names:
            val = row[c] if c in row.index else 0.0
            try:
                X.append(float(val) if pd.notna(val) else 0.0)
            except (TypeError, ValueError):
                X.append(0.0)
        arr = np.array(X).reshape(1, -1)
        if as_dataframe:
            return pd.DataFrame(arr, columns=feature_names)
        return arr

    if row is not None:
        predictions = []
        for d in [1, 2, 3]:
            model, feature_names, is_keras = load_model_for_day(d)
            if model is None or not feature_names:
                predictions.append(None)
                continue
            X = _prepare_x(row, feature_names, as_dataframe=not is_keras)
            pred = model.predict(X, verbose=0) if is_keras else model.predict(X)
            pred = np.ravel(pred)[0]
            predictions.append(float(pred))
        print(f"\nModel predictions (US AQI): Day+1 = {predictions[0]:.0f}, Day+2 = {predictions[1]:.0f}, Day+3 = {predictions[2]:.0f}")
        if current_openmeteo is not None and predictions[0] is not None:
            print(f"  -> Day+1 vs current: diff = {abs(predictions[0] - current_openmeteo):.0f}")

    print("\nCompare with ground stations: https://aqicn.org/city/karachi/ or IQAir Karachi")
    print("(Open-Meteo uses CAMS model; local stations may differ slightly.)")


if __name__ == "__main__":
    main()
