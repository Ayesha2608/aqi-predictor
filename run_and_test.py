"""
Run full pipeline and test: fetch → features → train → predict.
Use: python run_and_test.py
(Uses Python from your PATH; if venv has numpy issues, use system Python 3.11.)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("1. Fetch raw data (API)")
    print("=" * 60)
    try:
        from scripts.fetch_raw_data import fetch_raw_for_date
        from config.settings import DEFAULT_LAT, DEFAULT_LON, DEFAULT_CITY
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fetch_raw_for_date(date, lat=DEFAULT_LAT, lon=DEFAULT_LON, save_to_disk=False)
        print(f"   OK: Fetched raw data for {date} ({DEFAULT_CITY})")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    print("\n" + "=" * 60)
    print("2. Hourly feature pipeline (features -> Feature Store)")
    print("=" * 60)
    try:
        from scripts.run_hourly_feature_pipeline import run_hourly
        n = run_hourly()
        print(f"   OK: Saved {n} feature rows to Feature Store")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    print("\n" + "=" * 60)
    print("3. Daily training (train 3 models, best per day)")
    print("=" * 60)
    try:
        from scripts.training_pipeline import main as training_main
        training_main()
        print("   OK: Training pipeline complete")
    except Exception as e:
        print(f"   FAIL: {e}")
        return 1

    print("\n" + "=" * 60)
    print("4. Predict next 3 days AQI (same as dashboard)")
    print("=" * 60)
    try:
        from app.dashboard import predict_next_3_days
        predictions, metrics, err = predict_next_3_days()
        if err:
            print(f"   FAIL: {err}")
            return 1
        print(f"   Day +1 AQI: {predictions[0]}")
        print(f"   Day +2 AQI: {predictions[1]}")
        print(f"   Day +3 AQI: {predictions[2]}")
        for d, m in metrics.items():
            print(f"   {d}: model={m.get('model')}, rmse={m.get('rmse')}, r2={m.get('r2')}")
        print("   OK: Predictions ready for dashboard")
    except Exception as e:
        print(f"   FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("ALL STEPS PASSED. Run dashboard: streamlit run app/dashboard.py")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
