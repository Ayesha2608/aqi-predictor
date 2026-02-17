from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "aqi_predictor")

def verify_db():
    print(f"Connecting to: {MONGODB_URI.split('@')[-1]}")
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB]
    
    print("\n--- Collections ---")
    print(db.list_collection_names())
    
    # Check Features
    features_coll = db["aqi_features"]
    f_count = features_coll.count_documents({})
    print("\n--- Features ---")
    print(f"Total rows: {f_count}")
    if f_count > 0:
        latest = features_coll.find_one(sort=[("timestamp", -1)])
        print(f"Latest timestamp: {latest.get('timestamp')}")
        print(f"Latest city: {latest.get('city')}")

    # Check Models
    models_coll = db["models"]
    m_count = models_coll.count_documents({})
    print("\n--- Models ---")
    print(f"Total models: {m_count}")
    
    production_models = list(models_coll.find({"production": True}))
    print(f"Production models count: {len(production_models)}")
    
    for m in production_models:
        has_binary = "model_binary" in m
        binary_size = len(m["model_binary"]) if has_binary else 0
        print(f"Day {m.get('target_day')}: Has Binary={has_binary}, Size={binary_size} bytes")

if __name__ == "__main__":
    verify_db()
