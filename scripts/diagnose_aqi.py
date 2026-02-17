"""
Diagnostic script to check AQI prediction issues:
- PM2.5 → US AQI conversion correctness
- Training data distribution (min/max/mean)
- Feature leakage check
- Calibration effectiveness
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import KARACHI_REFERENCE_AQI, DEFAULT_CITY
from scripts.db import get_latest_features, load_features
from scripts.feature_pipeline import pm25_to_us_aqi
from scripts.training_pipeline import build_shifted_targets, prepare_xy


def check_pm25_conversion():
    """Verify PM2.5 → US AQI conversion is correct."""
    print("=" * 60)
    print("STEP 1: PM2.5 -> US AQI Conversion Check")
    print("=" * 60)
    
    # Test cases from EPA breakpoints
    test_cases = [
        (9.0, 50),      # Upper bound of Good
        (35.4, 100),    # Upper bound of Moderate
        (55.4, 150),    # Upper bound of Unhealthy for Sensitive
        (96.16, None),  # Current Karachi value (should be ~179)
    ]
    
    for pm25, expected_max in test_cases:
        aqi = pm25_to_us_aqi(pm25)
        print(f"PM2.5: {pm25:6.2f} ug/m3 -> US AQI: {aqi:6.1f}", end="")
        if expected_max:
            print(f" (expected <= {expected_max})", end="")
        print()
    print()


def check_training_distribution():
    """Check training data AQI distribution."""
    print("=" * 60)
    print("STEP 2: Training Data Distribution")
    print("=" * 60)
    
    df = load_features(city=DEFAULT_CITY, limit=0)
    if df.empty:
        print("ERROR: No data in Feature Store!")
        return
    
    aqi_col = pd.to_numeric(df["us_aqi"], errors="coerce").dropna()
    if len(aqi_col) == 0:
        print("ERROR: No valid US AQI values!")
        return
    
    print(f"Total rows: {len(df)}")
    print(f"Valid US AQI rows: {len(aqi_col)}")
    print("\nUS AQI Statistics:")
    print(f"  Min:  {aqi_col.min():6.1f}")
    print(f"  Max:  {aqi_col.max():6.1f}")
    print(f"  Mean: {aqi_col.mean():6.1f}")
    print(f"  Median: {aqi_col.median():6.1f}")
    print(f"  Std:  {aqi_col.std():6.1f}")
    
    # Check PM2.5 distribution
    pm25_col = pd.to_numeric(df["pm2_5"], errors="coerce").dropna()
    if len(pm25_col) > 0:
        print("\nPM2.5 Statistics:")
        print(f"  Min:  {pm25_col.min():6.1f} ug/m3")
        print(f"  Max:  {pm25_col.max():6.1f} ug/m3")
        print(f"  Mean: {pm25_col.mean():6.1f} ug/m3")
        print(f"  Median: {pm25_col.median():6.1f} ug/m3")
    
    # Check target distribution
    df = build_shifted_targets(df)
    for day in [1, 2, 3]:
        target_col = pd.to_numeric(df[f"target_aqi_d{day}"], errors="coerce").dropna()
        if len(target_col) > 0:
            print(f"\nTarget AQI Day+{day} Statistics:")
            print(f"  Valid rows: {len(target_col)}")
            print(f"  Min:  {target_col.min():6.1f}")
            print(f"  Max:  {target_col.max():6.1f}")
            print(f"  Mean: {target_col.mean():6.1f}")
            print(f"  Median: {target_col.median():6.1f}")
    
    print()


def check_feature_leakage():
    """Check for feature leakage (predicting today using today's data)."""
    print("=" * 60)
    print("STEP 3: Feature Leakage Check")
    print("=" * 60)
    
    df = load_features(city=DEFAULT_CITY, limit=0)
    if df.empty:
        print("ERROR: No data!")
        return
    
    df = build_shifted_targets(df)
    X, y = prepare_xy(df, target_day=1)
    if X is None or y is None:
        print("ERROR: Cannot prepare X, y!")
        return
    
    # Check if current AQI is in features (would be leakage)
    if "us_aqi" in X.columns:
        print("WARNING: 'us_aqi' is in features! This could cause leakage.")
        print("   (Model might be using today's AQI to predict tomorrow)")
    else:
        print("OK: 'us_aqi' not in features (good - no direct leakage)")
    
    # Check correlation between features and target
    if len(X) > 10:
        corr = X.corrwith(y)
        high_corr = corr[abs(corr) > 0.8]
        if len(high_corr) > 0:
            print("\nWARNING: Features with very high correlation (>0.8) to target:")
            for feat, val in high_corr.items():
                print(f"   {feat}: {val:.3f}")
        else:
            print("OK: No features with suspiciously high correlation")
    
    print()


def check_calibration():
    """Test calibration function."""
    print("=" * 60)
    print("STEP 4: Calibration Check")
    print("=" * 60)
    
    from app.dashboard import calibrate_aqi
    
    test_values = [
        (179.5, "Current stored (high)"),
        (96.0, "Reference (Moderate)"),
        (50.0, "Good"),
        (150.0, "Unhealthy for Sensitive"),
    ]
    
    print(f"Reference AQI: {KARACHI_REFERENCE_AQI}")
    print("\nCalibration results:")
    for stored, label in test_values:
        calibrated = calibrate_aqi(stored)
        change = calibrated - stored
        print(f"  {label:30s}: {stored:6.1f} -> {calibrated:6.1f} (change: {change:+6.1f})")
    
    print()


def check_latest_predictions():
    """Check latest predictions vs stored AQI."""
    print("=" * 60)
    print("STEP 5: Latest Predictions vs Stored AQI")
    print("=" * 60)
    
    df = get_latest_features(city=DEFAULT_CITY, n_days=1)
    if df.empty:
        print("ERROR: No recent data!")
        return
    
    latest = df.iloc[-1]
    stored_aqi = pd.to_numeric(latest.get("us_aqi"), errors="coerce")
    pm25 = pd.to_numeric(latest.get("pm2_5"), errors="coerce")
    
    print(f"Latest timestamp: {latest.get('timestamp', 'N/A')}")
    if pd.notna(pm25):
        print(f"PM2.5: {pm25:.2f} ug/m3")
        calculated_aqi = pm25_to_us_aqi(float(pm25))
        print(f"Calculated US AQI (from PM2.5): {calculated_aqi:.1f}")
    if pd.notna(stored_aqi):
        print(f"Stored US AQI: {stored_aqi:.1f}")
    
    from app.dashboard import calibrate_aqi
    if pd.notna(stored_aqi):
        calibrated = calibrate_aqi(float(stored_aqi))
        print(f"Calibrated US AQI: {calibrated:.1f}")
        print(f"Reference (expected): {KARACHI_REFERENCE_AQI:.1f}")
        diff = abs(calibrated - KARACHI_REFERENCE_AQI)
        print(f"Difference from reference: {diff:.1f}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AQI Prediction Diagnostic Report")
    print("=" * 60 + "\n")
    
    check_pm25_conversion()
    check_training_distribution()
    check_feature_leakage()
    check_calibration()
    check_latest_predictions()
    
    print("=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)
