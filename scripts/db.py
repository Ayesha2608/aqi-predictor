"""
MongoDB client and Feature Store / Model Registry helpers.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MONGODB_URI,
    MONGODB_DB,
    FEATURES_COLLECTION,
    MODELS_COLLECTION,
    ALERTS_COLLECTION,
)


def get_client():
    return MongoClient(MONGODB_URI)


def get_db():
    return get_client()[MONGODB_DB]


# ---------- Feature Store ----------


def save_features(features_df: pd.DataFrame, city: str = "London") -> int:
    """Insert feature rows into Feature Store with upsert (deduplication)."""
    db = get_db()
    coll = db[FEATURES_COLLECTION]
    from pymongo import UpdateOne
    
    records = features_df.to_dict(orient="records")
    ops = []
    for r in records:
        r["city"] = city
        for k, v in r.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                r[k] = v.isoformat() if hasattr(v, "isoformat") else str(v)
            elif isinstance(v, (float,)) and (pd.isna(v) or v != v):
                r[k] = None
        
        # Identity fields for deduplication
        query = {"timestamp": r["timestamp"], "city": city}
        ops.append(UpdateOne(query, {"$set": r}, upsert=True))
    
    if ops:
        result = coll.bulk_write(ops)
        return result.upserted_count + result.modified_count
    return 0


def load_features(
    city: str = "London",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 0,
) -> pd.DataFrame:
    """Load historical features (and targets) from Feature Store."""
    db = get_db()
    coll = db[FEATURES_COLLECTION]
    query = {"city": city}
    if start_date:
        query["timestamp"] = query.get("timestamp", {})
        query["timestamp"]["$gte"] = start_date
    if end_date:
        query["timestamp"] = query.get("timestamp", {})
        query["timestamp"]["$lte"] = end_date
    cursor = coll.find(query).sort("timestamp", ASCENDING)
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame(docs)


def get_latest_features(city: str = "London", n_days: int = 7) -> pd.DataFrame:
    """Get most recent feature rows for inference. High limit to handle history + forecast."""
    db = get_db()
    coll = db[FEATURES_COLLECTION]
    # Fetch a large buffer (e.g. 1000 rows) to ensure we get past any duplicates/overlaps
    cursor = coll.find({"city": city}).sort("timestamp", DESCENDING).limit(1000)
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # Final safety deduplication on read
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return df.reset_index(drop=True)


# ---------- Model Registry ----------


def save_model_metadata(
    model_path: str,
    metrics: dict,
    version: str = None,
    set_production: bool = True,
    target_day: Optional[int] = None,
    all_models_comparison: Optional[list] = None,
    model_binary: Optional[bytes] = None,
    feature_names: Optional[list] = None,
) -> str:
    """Register model in Model Registry. Stores best model + all_models_comparison for UI."""
    db = get_db()
    coll = db[MODELS_COLLECTION]
    version = version or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if set_production and target_day is not None:
        coll.update_many({"target_day": target_day}, {"$set": {"production": False}})
    elif set_production:
        coll.update_many({}, {"$set": {"production": False}})
    doc = {
        "version": version,
        "path": model_path,
        "metrics": metrics,
        "created_at": datetime.utcnow().isoformat(),
        "production": set_production,
        "target_day": target_day,
    }
    if all_models_comparison is not None:
        doc["all_models_comparison"] = all_models_comparison
    if model_binary is not None:
        from bson.binary import Binary
        doc["model_binary"] = Binary(model_binary)
    if feature_names is not None:
        doc["feature_names"] = feature_names
    coll.insert_one(doc)
    return version


def get_production_model(target_day: Optional[int] = None) -> Optional[dict]:
    """Get current production model metadata. If target_day given, return that day's model."""
    db = get_db()
    coll = db[MODELS_COLLECTION]
    query = {"production": True}
    if target_day is not None:
        query["target_day"] = target_day
    doc = coll.find_one(query)
    return doc


def get_latest_model() -> Optional[dict]:
    """Get latest model by created_at."""
    db = get_db()
    coll = db[MODELS_COLLECTION]
    doc = coll.find_one(sort=[("created_at", DESCENDING)])
    return doc


# ---------- Alerts ----------


def log_alert(city: str, aqi_value: float, forecast_date: str, message: str):
    """Log hazardous AQI alert."""
    db = get_db()
    coll = db[ALERTS_COLLECTION]
    coll.insert_one({
        "city": city,
        "aqi": aqi_value,
        "forecast_date": forecast_date,
        "message": message,
        "created_at": datetime.utcnow().isoformat(),
    })
