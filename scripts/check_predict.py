"""Quick check: latest AQI and blended predictions (same as API/dashboard)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.db import get_latest_features
from scripts.model_loader import load_model_for_day
from config.settings import KARACHI_REFERENCE_AQI, AQI_CALIBRATION_ENABLED


def calibrate_aqi(stored_aqi: float, reference_aqi: float = KARACHI_REFERENCE_AQI) -> float:
    """Calibrate stored AQI toward reference if way off."""
    if not AQI_CALIBRATION_ENABLED or stored_aqi is None:
        return stored_aqi
    stored_aqi = float(stored_aqi)
    reference_aqi = float(reference_aqi)
    if stored_aqi > reference_aqi * 1.5:
        scale = (reference_aqi / stored_aqi) ** 0.5
        calibrated = stored_aqi * scale + reference_aqi * (1 - scale)
        return max(0, min(500, calibrated))
    return stored_aqi

def main():
    df = get_latest_features(city="Karachi", n_days=1)
    if df.empty:
        print("No features in Feature Store.")
        return
    row = df.iloc[-1]
    today_aqi = None
    for col in ("us_aqi", "aqi"):
        if col in row.index:
            v = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    today_aqi = float(v)
                    break
                except (TypeError, ValueError):
                    pass
    print("Today (stored):", round(today_aqi, 1) if today_aqi is not None else "N/A")
    
    # Calibrate today's AQI
    today_aqi_calibrated = calibrate_aqi(today_aqi) if today_aqi is not None else None
    if today_aqi_calibrated != today_aqi:
        print(f"Today (calibrated): {round(today_aqi_calibrated, 1)}")

    preds = []
    for d in [1, 2, 3]:
        model, names, is_k = load_model_for_day(d)
        if model is None or not names:
            preds.append(None)
            continue
        vals = []
        for c in names:
            v = row.get(c, 0)
            try:
                vals.append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                vals.append(0.0)
        X = np.array(vals).reshape(1, -1)
        p = model.predict(X, verbose=0) if is_k else model.predict(X)
        preds.append(float(np.ravel(p)[0]))

    print("Raw model (d1,d2,d3):", [round(p, 1) if p is not None else None for p in preds])
    
    # Apply calibration and blending (same as dashboard/API)
    if today_aqi_calibrated is not None:
        alpha = 0.7  # Higher weight on calibrated today
        blended = []
        for p in preds:
            if p is not None:
                calibrated_pred = calibrate_aqi(float(p))
                final = alpha * today_aqi_calibrated + (1 - alpha) * calibrated_pred
                blended.append(round(final, 1))
            else:
                blended.append(None)
        print(f"Final (calibrated + blended, alpha={alpha}):", blended)

if __name__ == "__main__":
    main()
