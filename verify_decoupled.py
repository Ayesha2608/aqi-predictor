
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Avoid importing from app.dashboard to prevent side effects
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.db import get_latest_features
from scripts.model_loader import load_model_for_day
from config.settings import DEFAULT_CITY, KARACHI_REFERENCE_AQI, AQI_CALIBRATION_ENABLED

def calibrate_aqi(stored_aqi: float, reference_aqi: float = KARACHI_REFERENCE_AQI) -> float:
    if not AQI_CALIBRATION_ENABLED or stored_aqi is None:
        return stored_aqi
    stored_aqi = float(stored_aqi)
    if stored_aqi > reference_aqi * 1.3:
        excess = stored_aqi - reference_aqi
        calibrated = reference_aqi + (excess * 0.15)
        return max(0, min(500, calibrated))
    return stored_aqi

def prepare_inference_row(row: pd.Series, feature_names: list, as_dataframe: bool = True):
    X = []
    for c in feature_names:
        val = row.get(c, 0.0)
        try:
            X.append(float(val) if pd.notna(val) else 0.0)
        except (ValueError, TypeError):
            X.append(0.0)
    arr = np.array(X).reshape(1, -1)
    return pd.DataFrame(arr, columns=feature_names) if as_dataframe else arr

def verify():
    print("--- Decoupled Dashboard Verification ---", flush=True)
    
    # 1. Load Models
    print("DEBUG: Loading models...", flush=True)
    models = {}
    for day in [1, 2, 3]:
        models[day] = load_model_for_day(day)
        print(f"DEBUG: Day {day} model loaded: {type(models[day][0])}", flush=True)

    # 2. Fetch Data
    print("DEBUG: Fetching features from DB...", flush=True)
    df_features = get_latest_features(city=DEFAULT_CITY, n_days=5)
    print(f"DEBUG: Found {len(df_features)} feature rows.", flush=True)
    
    if df_features.empty:
        print("FAIL: No features in DB.")
        return

    ts_col = "timestamp" if "timestamp" in df_features.columns else "datetime_iso"
    df_features[ts_col] = pd.to_datetime(df_features[ts_col])
    now = datetime.now(df_features[ts_col].iloc[0].tzinfo if df_features[ts_col].iloc[0].tz else None)
    df_future = df_features[df_features[ts_col] >= now - timedelta(hours=1)].copy()
    print(f"DEBUG: Found {len(df_future)} future rows.", flush=True)

    # 3. Predict (First 10 rows)
    results = []
    test_rows = df_future.head(10)
    for i, (idx, row) in enumerate(test_rows.iterrows()):
        hour_offset = (row[ts_col] - now).total_seconds() / 3600
        target_day = 1 if hour_offset <= 24 else (2 if hour_offset <= 48 else 3)
        
        model_info = models.get(target_day)
        model, features, is_keras = model_info
        
        pred_val = None
        if model and features:
            X = prepare_inference_row(row, features, as_dataframe=not is_keras)
            pred = model.predict(X, verbose=0) if is_keras else model.predict(X)
            pred_val = float(np.ravel(pred)[0])
            pred_val = calibrate_aqi(pred_val)
        
        print(f"DEBUG: Row {i} | Time: {row[ts_col]} | Day: {target_day} | AQI: {pred_val}", flush=True)
        results.append(pred_val)

    # 4. Final Verification
    if len(results) > 0:
        std = np.std([r for r in results if r is not None])
        print(f"DEBUG: Result Std Dev: {std:.4f}", flush=True)
        if std > 0.0001:
            print("SUCCESS: Predictions show variance.", flush=True)
        else:
            print("WARNING: Predictions are flat.", flush=True)
        
        if len(df_future) >= 24:
            print(f"SUCCESS: Report will show {len(df_future)} hourly rows.", flush=True)
        else:
            print(f"WARNING: Only {len(df_future)} rows available.", flush=True)
    else:
        print("FAIL: No results generated.")

if __name__ == "__main__":
    verify()
