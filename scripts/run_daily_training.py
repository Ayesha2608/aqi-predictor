"""
DAILY script: run once per day.
Loads all features from Feature Store (hourly data for full days).
Trains 3 models (Ridge, Random Forest, TensorFlow); evaluates RMSE, MAE, R².
Best model per target (d1, d2, d3) is set as production automatically.
UI will always use the best model — no manual selection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.training_pipeline import main as training_main


def main():
    """Run full training pipeline: load from Feature Store → train → set best as production."""
    training_main()


if __name__ == "__main__":
    main()
