import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.dashboard import predict_next_3_days, calibrate_aqi
from scripts.db import get_production_model, get_latest_features
from config.settings import DEFAULT_CITY

print("="*70)
print("AQI PREDICTIONS - NEXT 3 DAYS")
print("="*70)

# Get predictions
preds, metrics, err = predict_next_3_days()

# Get Today's AQI from Feature Store
latest_df = get_latest_features(city=DEFAULT_CITY, n_days=1)
today_aqi = None
raw_aqi = None
if not latest_df.empty:
    for col in ("us_aqi", "aqi"):
        if col in latest_df.columns:
            raw_aqi = latest_df[col].iloc[-1]
            today_aqi = calibrate_aqi(raw_aqi)
            break

today_val = f"Today's AQI (Latest Real-time): {today_aqi:.1f}" if today_aqi is not None else "Today's AQI: N/A (Run feature pipeline)"
print("\n" + today_val)

# Write to file for verification
with open("predictions_report.txt", "w", encoding="utf-8") as f:
    f.write("AQI FORECAST REPORT\n")
    f.write("===================\n")
    f.write(f"{today_val}\n\n")
    f.write("Predicted AQI (Next 3 Days):\n")
    f.write(f"  Day +1 (Tomorrow):     {preds[0]:.1f}\n" if preds[0] else "  Day +1: N/A\n")
    f.write(f"  Day +2 (Day after):    {preds[1]:.1f}\n" if preds[1] else "  Day +2: N/A\n")
    f.write(f"  Day +3 (3 days ahead): {preds[2]:.1f}\n" if preds[2] else "  Day +3: N/A\n")

if err:
    print(f"Error: {err}")
else:
    print("\nPredicted AQI:")
    print(f"  Day +1 (Tomorrow):     {preds[0]:.1f}" if preds[0] else "  Day +1: N/A")
    print(f"  Day +2 (Day after):    {preds[1]:.1f}" if preds[1] else "  Day +2: N/A")
    print(f"  Day +3 (3 days ahead): {preds[2]:.1f}" if preds[2] else "  Day +3: N/A")

print("\n" + "="*70)
print("MODELS USED (BEST SELECTED)")
print("="*70)

for day in [1, 2, 3]:
    doc = get_production_model(target_day=day)
    if doc and doc.get("metrics"):
        m = doc["metrics"]
        print(f"\nDay +{day}:")
        print(f"  Model: {m.get('model', 'N/A')}")
        print(f"  RMSE:  {m.get('rmse', 0):.4f}")
        print(f"  MAE:   {m.get('mae', 0):.4f}")
        print(f"  R²:    {m.get('r2', 0):.4f}")
        
        # Show all models comparison
        if doc.get("all_models_comparison"):
            print("  All 3 models trained:")
            for model in doc["all_models_comparison"]:
                status = "[SELECTED]" if model["model"] == m.get("model") else ""
                print(f"    - {model['model']}: RMSE={model['rmse']:.4f} {status}")

print("\n" + "="*70)
