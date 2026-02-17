"""
Master Pipeline: The one-stop script for running all local tasks.
1. Checks for data (runs backfill if empty).
2. Runs feature pipeline for today's data.
3. Runs training pipeline only if needed (once a day).
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_CITY
from scripts.db import get_latest_features, get_db
from scripts.feature_pipeline import run_feature_pipeline
from scripts.training_pipeline import main as run_training
from scripts.backfill import backfill_date_range

def needs_backfill():
    """Check if we have enough historical data (at least 5 days)."""
    df = get_latest_features(city=DEFAULT_CITY, n_days=5)
    return len(df) < 100  # Expect ~120 rows for 5 days hourly

def needs_training():
    """Check if training has run in the last 20 hours."""
    last_doc = get_db()["models"].find_one(sort=[("created_at", -1)])
    if not last_doc or "created_at" not in last_doc:
        return True
    try:
        last_time = datetime.fromisoformat(last_doc["created_at"])
        elapsed = datetime.utcnow() - last_time
        return elapsed > timedelta(hours=20)
    except Exception:
        return True

def run_all():
    print("🚀 Starting Master Pipeline...")
    
    # 1. Backfill if database is new
    if needs_backfill():
        print("📥 Database looks empty or new. Running 14-day backfill...")
        end = datetime.utcnow()
        start = end - timedelta(days=14)
        backfill_date_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        print("✅ Backfill complete.")
    else:
        print("📊 Historical data exists. Skipping backfill.")

    # 2. Daily/Hourly Feature Fetch
    print("🔄 Fetching latest features for today...")
    run_feature_pipeline(save_to_store=True)
    print("✅ Feature pipeline complete.")

    # 3. Training (if needed)
    if needs_training():
        print("🏋️ Training models (this runs once per day or for first-time setup)...")
        run_training()
        print("✅ Training complete.")
    else:
        print("⏰ Training already up-to-date for today. Skipping.")

    print("\n✨ Master Pipeline finished successfully!")

if __name__ == "__main__":
    run_all()
