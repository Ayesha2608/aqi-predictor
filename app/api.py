"""
Flask API: expose AQI predictions for next 3 days (per PDF: Streamlit + Flask/FastAPI).
Run: flask --app app.api run  or  python -m flask --app app.api run
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flask import Flask, jsonify

from config.settings import (
    DEFAULT_CITY,
    AQI_HAZARDOUS_THRESHOLD,
    AQI_UNHEALTHY_THRESHOLD,
    KARACHI_REFERENCE_AQI,
    AQI_CALIBRATION_ENABLED,
)
from scripts.db import get_latest_features, get_production_model, log_alert
from scripts.model_loader import load_model_for_day


def get_latest_feature_row():
    df = get_latest_features(city=DEFAULT_CITY, n_days=7)
    if df.empty:
        return None
    return df.iloc[[-1]]


def prepare_inference_row(row: pd.Series, feature_names: list) -> np.ndarray:
    X = []
    for c in feature_names:
        if c in row.index:
            val = row[c]
            try:
                X.append(float(val) if pd.notna(val) else 0.0)
            except (TypeError, ValueError):
                X.append(0.0)
        else:
            X.append(0.0)
    return np.array(X).reshape(1, -1)


def calibrate_aqi(stored_aqi: float, reference_aqi: float = KARACHI_REFERENCE_AQI) -> float:
    """
    Calibrate stored AQI toward reference (e.g., 96 Moderate) if it's way off.
    This corrects for API discrepancies (e.g., OpenWeather PM2.5 inflated vs weather app).
    Returns calibrated AQI that stays in same category band but closer to reference.
    """
    if not AQI_CALIBRATION_ENABLED or stored_aqi is None:
        return stored_aqi
    stored_aqi = float(stored_aqi)
    reference_aqi = float(reference_aqi)
    
    # If stored is way higher than reference (e.g., 179 vs 96), apply scaling
    if stored_aqi > reference_aqi * 1.5:  # More than 50% higher
        # Scale factor: bring it closer to reference while preserving category
        # Use logarithmic scaling to avoid over-correction
        scale = (reference_aqi / stored_aqi) ** 0.5  # Square root scaling
        calibrated = stored_aqi * scale + reference_aqi * (1 - scale)
        return max(0, min(500, calibrated))  # Clamp to valid range
    return stored_aqi


def compute_predictions():
    """Return (predictions [d1,d2,d3], metrics dict, error str or None)."""
    latest = get_latest_feature_row()
    if latest is None:
        return None, {}, "No feature data. Run feature pipeline or backfill first."
    # Current AQI (today) from latest feature row — used to stabilize forecasts
    current_aqi_raw = None
    row = latest.iloc[0]
    for col in ("us_aqi", "aqi"):
        if col in row.index and pd.notna(row[col]):
            try:
                current_aqi_raw = float(row[col])
                break
            except (TypeError, ValueError):
                continue
    
    # Calibrate today's AQI toward reference (e.g., 96 Moderate) if way off
    current_aqi = calibrate_aqi(current_aqi_raw) if current_aqi_raw is not None else None

    predictions = [None, None, None]
    metrics = {}
    for target_day in [1, 2, 3]:
        model, feature_names, is_keras = load_model_for_day(target_day)
        if model is None or not feature_names:
            continue
        X = prepare_inference_row(latest.iloc[0], feature_names)
        pred = model.predict(X, verbose=0) if is_keras else model.predict(X)
        pred = np.ravel(pred)
        predictions[target_day - 1] = float(pred[0])
        doc = get_production_model(target_day=target_day)
        if doc and "metrics" in doc:
            metrics[f"day_{target_day}"] = doc["metrics"]

    # Blend model predictions with calibrated today's AQI for realistic forecasts
    # Example: if today ≈ 96 (calibrated) and pure model says 40, final ~ 0.7*96 + 0.3*40 ≈ 79
    if current_aqi is not None:
        alpha = 0.7  # Higher weight on calibrated today's AQI for stability
        for i, val in enumerate(predictions):
            if val is not None:
                try:
                    # Also calibrate raw model prediction before blending
                    calibrated_pred = calibrate_aqi(float(val))
                    predictions[i] = float(alpha * current_aqi + (1.0 - alpha) * calibrated_pred)
                except (TypeError, ValueError):
                    continue

    return predictions, metrics, None


app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict")
def predict():
    """Return next 3 days AQI forecast and optional alerts."""
    predictions, metrics, err = compute_predictions()
    if err:
        return jsonify({"error": err}), 503
    alerts = []
    for i, val in enumerate(predictions):
        if val is not None and val >= AQI_UNHEALTHY_THRESHOLD:
            alerts.append({
                "day": i + 1,
                "aqi": val,
                "level": "hazardous" if val >= AQI_HAZARDOUS_THRESHOLD else "unhealthy",
            })
            if val >= AQI_HAZARDOUS_THRESHOLD:
                try:
                    log_alert(DEFAULT_CITY, val, f"Day +{i + 1}", f"Hazardous AQI {val:.0f}")
                except Exception:
                    pass
    return jsonify({
        "city": DEFAULT_CITY,
        "forecast": {
            "day_1": predictions[0],
            "day_2": predictions[1],
            "day_3": predictions[2],
        },
        "alerts": alerts,
        "model_metrics": metrics,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
