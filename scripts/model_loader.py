"""
Load production models from Model Registry (supports joblib and Keras).
Used by dashboard and Flask API.
"""
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROJECT_ROOT
from scripts.db import get_production_model


def load_model_for_day(target_day: int) -> Tuple[Optional[Any], Optional[list], bool]:
    """
    Load production model for target_day (1/2/3).
    Returns (model, feature_names, is_keras).
    """
    doc = get_production_model(target_day=target_day)
    if not doc or "path" not in doc:
        return None, None, False
    path = Path(doc["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None, None, False

    if path.suffix == ".keras":
        try:
            from tensorflow import keras
            model = keras.models.load_model(path)
            meta_path = path.with_suffix(".keras.json")
            feature_names = []
            if meta_path.exists():
                with open(meta_path) as f:
                    feature_names = json.load(f).get("feature_names", [])
            return model, feature_names, True
        except Exception:
            return None, None, True
    else:
        import joblib
        data = joblib.load(path)
        model = data.get("model")
        feature_names = data.get("feature_names", [])
        return model, feature_names, False
