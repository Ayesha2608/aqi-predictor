import os
import json
import pandas as pd
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.db import save_features
from scripts.feature_pipeline import compute_features_from_raw

def restore():
    raw_dir = Path("data/raw")
    files = list(raw_dir.glob("raw_*.json"))
    total_rows = 0
    print(f"Found {len(files)} local raw files to restore history.")
    
    for f in files:
        try:
            with open(f, "r") as fr:
                raw = json.load(fr)
                df = compute_features_from_raw(raw)
                if not df.empty:
                    n = save_features(df, city="Karachi")
                    total_rows += n
                    print(f"Restored {n} rows from {f.name}")
        except Exception as e:
            print(f"Failed to process {f.name}: {e}")
    
    print(f"--- SUCCESS: Restored {total_rows} Karachi rows to MongoDB ---")

if __name__ == "__main__":
    restore()
