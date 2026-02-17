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
    
    # Flag to indicate if we're loading from local vs binary
    load_from_binary = not path.exists() and "model_binary" in doc
    
    if not path.exists() and not load_from_binary:
        return None, None, False

    if path.suffix == ".keras":
        try:
            from tensorflow import keras
            import tempfile
            import os
            
            if load_from_binary:
                # Keras.models.load_model requires a file/path, can't load from bytes directly easily
                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                    tmp.write(doc["model_binary"])
                    tmp_path = tmp.name
                try:
                    model = keras.models.load_model(tmp_path)
                    # Feature names can be in doc directly or in metrics
                    feature_names = doc.get("feature_names") or doc.get("metrics", {}).get("feature_names", [])
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
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
        import io
        if load_from_binary:
            data = joblib.load(io.BytesIO(doc["model_binary"]))
        else:
            data = joblib.load(path)
        model = data.get("model")
        feature_names = data.get("feature_names", [])
        return model, feature_names, False
