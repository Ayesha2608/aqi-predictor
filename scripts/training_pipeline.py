"""
Training pipeline: load features from Feature Store, train and evaluate models
(Ridge, Random Forest, TensorFlow), save best model and register in Model Registry.
Optional SHAP importance.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DEFAULT_CITY,
    MODELS_DIR,
    METRICS_DIR,
    PROJECT_ROOT,
)
from scripts.db import load_features, save_model_metadata


# Feature columns (numeric) and target columns
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_max", "temperature_min", "precipitation", "humidity", "wind_speed",
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "us_aqi", "aqi_change_rate",
]
TARGET_COLS = ["target_aqi_d1", "target_aqi_d2", "target_aqi_d3"]

# Hours per day for shift-based targets (hourly data)
HOURS_PER_DAY = 24


def build_shifted_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build future targets as daily mean AQI so model predicts that day's AQI (~92) not point-in-time.
    target_aqi_d1 = mean AQI over next 24h, d2 = mean over 48-72h, d3 = mean over 72-96h.
    Requires sorted-by-time hourly data; needs at least 96 rows for d3.
    """
    if df.empty or len(df) < 96:
        return df
    aqi_col = None
    for c in ("us_aqi", "aqi"):
        if c in df.columns:
            aqi_col = pd.to_numeric(df[c], errors="coerce")
            break
    if aqi_col is None:
        return df
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime_iso"
    if ts_col not in df.columns:
        return df
    df = df.sort_values(ts_col).reset_index(drop=True)
    # Reindex aqi to match sorted df (aqi_col was from original df)
    aqi_col = pd.to_numeric(df["us_aqi"] if "us_aqi" in df.columns else df["aqi"], errors="coerce")
    roll24 = aqi_col.rolling(24, min_periods=24).mean()
    # At row i: d1 = mean(hours i+24..i+47), d2 = mean(i+48..i+71), d3 = mean(i+72..i+95)
    df["target_aqi_d1"] = roll24.shift(-47).values
    df["target_aqi_d2"] = roll24.shift(-71).values
    df["target_aqi_d3"] = roll24.shift(-95).values
    return df


def prepare_xy(df: pd.DataFrame, target_day: int = 1):
    """
    Prepare X and y from feature store DataFrame. target_day 1/2/3.
    Uses us_aqi (or aqi) as proxy when target is null (single-day data).
    When using that proxy we INCLUDE us_aqi so the model can predict close to current AQI
    (persistence-style forecast when no real future targets exist). Exclude only aqi_change_rate.
    When real target_aqi_d* exist (from backfill), use full features.
    """
    target_col = f"target_aqi_d{target_day}"
    y = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(dtype=float)
    use_proxy = y.isna().all()
    if use_proxy and "us_aqi" in df.columns:
        y = pd.to_numeric(df["us_aqi"], errors="coerce")
    if use_proxy and "aqi" in df.columns and y.isna().all():
        y = pd.to_numeric(df["aqi"], errors="coerce")

    # With proxy target, include us_aqi so prediction tracks current AQI (no aqi_change_rate)
    # BUT for real future targets, EXCLUDE us_aqi to avoid feature leakage (can't use today's AQI to predict tomorrow)
    if use_proxy:
        available = [c for c in FEATURE_COLS if c in df.columns and c != "aqi_change_rate"]
    else:
        # Exclude us_aqi and aqi_change_rate when predicting future (they leak current AQI)
        available = [c for c in FEATURE_COLS if c in df.columns and c not in ("us_aqi", "aqi", "aqi_change_rate")]
    if not available:
        return None, None
    X = df[available].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    X = X.fillna(0)
    mask = y.notna()
    X = X.loc[mask]
    y = y[mask]
    if len(X) < 10:
        return None, None
    return X, y


def evaluate(y_true, y_pred):
    """Return dict of RMSE, MAE, R². Caps R² for display when target has no variance or model is very bad."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
        # Constant target or single value: R² not meaningful
        var_ = np.var(y_true) if hasattr(y_true, "__iter__") else 0
        if var_ == 0 or (hasattr(y_true, "nunique") and y_true.nunique() <= 1):
            r2 = 0.0
        else:
            r2 = max(-1.0, min(1.0, r2))  # clamp to [-1, 1] for UI
    except Exception:
        r2 = 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_tensorflow_model(n_features: int, random_state: int = 42):
    """Build a small Keras Dense model for AQI regression."""
    if not HAS_TF:
        return None
    keras.utils.set_random_seed(random_state)
    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(n_features,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_lstm_model(n_features: int, random_state: int = 42):
    """Build LSTM model for time-series AQI prediction (3rd model)."""
    if not HAS_TF:
        return None
    keras.utils.set_random_seed(random_state)
    model = keras.Sequential([
        keras.layers.Reshape((1, n_features), input_shape=(n_features,)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_evaluate(
    city: str = DEFAULT_CITY,
    target_day: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load features, train Ridge + RandomForest, pick best by RMSE, save model and metrics.
    Returns (model, metrics_dict, feature_names).
    """
    df = load_features(city=city, limit=0)
    if df.empty or len(df) < 20:
        raise ValueError("Not enough data in Feature Store. Run backfill or hourly pipeline first.")

    # Build real future targets from time series (AQI 24h, 48h, 72h later) when we have enough rows
    df = build_shifted_targets(df)

    X, y = prepare_xy(df, target_day=target_day)
    if X is None or len(X) < 10:
        raise ValueError(f"Not enough valid rows for target_aqi_d{target_day}")

    y_var = float(np.var(y))
    y_unique = len(pd.Series(y).dropna().unique())
    if y_var < 1e-10 or y_unique <= 1:
        print(
            f"  [WARNING] Target d{target_day} has NO variance. "
            "Need 72+ hourly rows for shift-based future targets (run backfill for 3+ days)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    feature_names = list(X.columns)

    # Train exactly 3 models: Ridge, RandomForest, and LSTM
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
    }
    if HAS_TF:
        models["LSTM"] = build_lstm_model(X_train.shape[1], random_state=random_state)
    else:
        print("  [WARNING] TensorFlow not installed. Only Ridge and RandomForest will be trained.")
        print("  Install TensorFlow to enable LSTM model: pip install tensorflow>=2.13.0")

    best_model = None
    best_metrics = None
    best_name = None
    all_models_metrics = []  # Compare all 3 models; show on UI

    # Train all models and track metrics
    for name, model in models.items():
        if model is None:
            continue
        try:
            print(f"  Training {name}...")
            if name == "LSTM":
                # LSTM requires specific training parameters
                model.fit(
                    X_train, y_train,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                )
                pred = model.predict(X_test, verbose=0)
                pred = np.ravel(pred)
            else:
                # Ridge and RandomForest use standard fit
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            
            m = evaluate(y_test, pred)
            m["model"] = name
            all_models_metrics.append({k: v for k, v in m.items()})
            print(f"    {name}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R²={m['r2']:.3f}")
            
            # Track best model by RMSE (lower is better)
            if best_metrics is None or m["rmse"] < best_metrics["rmse"]:
                best_metrics = m
                best_model = model
                best_name = name
        except Exception as e:
            print(f"    [ERROR] {name} training failed: {e}")
            continue

    if best_model is None:
        raise ValueError("No models trained successfully. Check data and dependencies.")
    
    # Log best model selection
    print(f"\n  ✓ BEST MODEL SELECTED: {best_name} (RMSE={best_metrics['rmse']:.2f})")
    print(f"    Selected because it has the lowest RMSE among all {len(all_models_metrics)} models trained.\n")
    
    # Save best model (path stored relative to PROJECT_ROOT for portability)
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    is_tf = best_name == "LSTM"
    ext = ".keras" if is_tf else ".joblib"
    model_name = f"aqi_d{target_day}_{best_name}_{version}{ext}"
    model_path = MODELS_DIR / model_name
    if is_tf:
        best_model.save(model_path)
        meta_path = model_path.with_suffix(".keras.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_names": feature_names}, f)
    else:
        joblib.dump({"model": best_model, "feature_names": feature_names}, model_path)
    model_path_for_registry = str(model_path.relative_to(PROJECT_ROOT))

    # Save metrics JSON (best + all comparison)
    metrics_path = METRICS_DIR / f"metrics_d{target_day}_{version}.json"
    with open(metrics_path, "w") as f:
        json.dump({"best": best_metrics, "all_models_comparison": all_models_metrics}, f, indent=2)

    # Register in Model Registry (MongoDB) with all 3 models' comparison
    model_binary = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_binary = f.read()

    save_model_metadata(
        model_path=model_path_for_registry,
        metrics=best_metrics,
        version=version,
        set_production=True,
        target_day=target_day,
        all_models_comparison=all_models_metrics,
        model_binary=model_binary,
    )

    return best_model, best_metrics, feature_names


def run_shap_importance(model, X_sample, feature_names, save_path: Path = None):
    """Compute SHAP or coefficient-based feature importance and save JSON."""
    if X_sample is None or len(X_sample) == 0:
        return None
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = dict(zip(feature_names, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):
        imp = dict(zip(feature_names, np.abs(model.coef_).tolist()))
    if imp is None:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shv = explainer.shap_values(X_sample)
            if shv is not None:
                mean_abs = np.abs(shv).mean(axis=0)
                imp = dict(zip(feature_names, mean_abs.tolist()))
        except Exception:
            pass
    if imp and save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path.with_suffix(".json"), "w") as f:
            json.dump(imp, f, indent=2)
    return imp


def main():
    """Train for all three targets (d1, d2, d3) with 3 models each and select best."""
    print("="*70)
    print("AQI PREDICTION MODEL TRAINING PIPELINE")
    print("="*70)
    print("Training 3 models: Ridge, RandomForest, LSTM")
    print(f"TensorFlow available: {HAS_TF}")
    if not HAS_TF:
        print("WARNING: TensorFlow not installed. Only 2 models will be trained.")
        print("Run: pip install tensorflow>=2.13.0")
    print("="*70 + "\n")
    
    for target_day in [1, 2, 3]:
        try:
            print(f"\n{'='*70}")
            print(f"TRAINING FOR TARGET DAY {target_day} (AQI {target_day} day(s) ahead)")
            print(f"{'='*70}")
            model, metrics, feature_names = train_and_evaluate(
                city=DEFAULT_CITY, target_day=target_day
            )
            # SHAP on a sample (skip for LSTM; TreeExplainer is for tree models)
            if metrics.get("model") not in ["LSTM", "tensorflow"]:
                df = load_features(city=DEFAULT_CITY, limit=500)
                X, y = prepare_xy(df, target_day=target_day)
                if X is not None and len(X) > 0:
                    sample = X.sample(min(100, len(X)), random_state=42)
                    run_shap_importance(
                        model,
                        sample,
                        feature_names,
                        save_path=METRICS_DIR / f"shap_importance_d{target_day}.json",
                    )
        except Exception as e:
            print(f"Training failed for d{target_day}: {e}")
    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
