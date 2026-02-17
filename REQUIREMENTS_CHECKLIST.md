# Requirements Checklist — AQI Predictor

This document maps the project requirements to the implementation.

---

## Feature Pipeline Development

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Fetch raw weather and pollutant data from external APIs (AQICN or OpenWeather) | ✅ | `scripts/fetch_raw_data.py`, `scripts/fetch_openweather.py` — OpenWeather (weather + air pollution). Open-Meteo fallback when key invalid. AQICN can be added via config. |
| Compute features: time-based (hour, day, month) and derived (AQI change rate) | ✅ | `scripts/feature_pipeline.py` — hour, day_of_week, month, is_weekend, aqi_change_rate, weather, pollutants. |
| Store processed features in Feature Store (Hopsworks or Vertex AI) | ✅ | Feature Store = **MongoDB** (collection `aqi_features`). Same role as Hopsworks/Vertex. `scripts/db.py` → `save_features()`. |

---

## Historical Data Backfill

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Run feature pipeline for past dates to generate training data | ✅ | `scripts/backfill.py` — backfill_date_range(start_date, end_date). |
| Create comprehensive dataset for model training and evaluation | ✅ | Backfill fills target_aqi_d1, d2, d3 from next days; data stored in Feature Store. |

---

## Training Pipeline Implementation

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Fetch historical features and targets from Feature Store | ✅ | `scripts/training_pipeline.py` — `load_features()` from MongoDB. |
| Experiment with various ML models (Random Forest, Ridge, TensorFlow/PyTorch) | ✅ | Ridge, RandomForestRegressor, TensorFlow/Keras Dense model in `training_pipeline.py`. |
| Evaluate performance using RMSE, MAE, R² metrics | ✅ | `evaluate(y_true, y_pred)` returns rmse, mae, r2. |
| Store trained models in Model Registry | ✅ | Model Registry = **MongoDB** (collection `models`). `scripts/db.py` → `save_model_metadata()`. Models saved to `models/*.joblib` and `*.keras`. |

---

## Automated CI/CD Pipeline

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Feature pipeline runs every hour automatically | ✅ | `.github/workflows/feature_pipeline.yml` — cron `0 * * * *`. Script: `scripts/run_hourly_feature_pipeline.py`. |
| Training pipeline runs daily for model updates | ✅ | `.github/workflows/training_pipeline.yml` — cron `0 6 * * *`. Script: `scripts/run_daily_training.py`. |
| Use Apache Airflow or GitHub Actions | ✅ | **GitHub Actions** (Airflow optional). |

---

## Web Application Dashboard

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Load models and features from Feature Store | ✅ | Dashboard uses `get_latest_features()`, `get_production_model()`, `load_model_for_day()` from MongoDB. |
| Compute real-time predictions for next 3 days | ✅ | `predict_next_3_days()` — loads production models and latest features, returns [d1, d2, d3]. |
| Display interactive dashboard with Streamlit/Gradio and Flask/FastAPI | ✅ | **Streamlit:** `app/dashboard.py`. **Flask:** `app/api.py` (GET /predict, /health). |

---

## Advanced Analytics Features

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Perform EDA to identify trends | ✅ | `scripts/eda.py` — AQI over time, distribution, correlations. Output in `metrics/`. |
| Use SHAP or LIME for feature importance | ✅ | SHAP/coefficient importance in `scripts/training_pipeline.py` → `run_shap_importance()`. Saved to `metrics/shap_importance_d*.json`. Dashboard shows SHAP section. |
| Implement alerts for hazardous AQI levels | ✅ | Dashboard: warnings for AQI ≥ unhealthy threshold, error for hazardous. `log_alert()` to MongoDB `alerts` collection. |
| Support multiple forecasting models (statistical to deep learning) | ✅ | Ridge (statistical), Random Forest (ensemble), TensorFlow/Keras (deep learning). Best model auto-selected per target day. |

---

## UI: AQI Prediction Display

The dashboard now shows:

- **AQI Prediction for Next 3 Days** — section with Today, Day +1, Day +2, Day +3.
- **Date** for each day (e.g. Jan 30, Jan 31, Feb 1).
- **Predicted AQI value** and **level** (Good, Fair, Moderate, Poor, Very Poor for 1–5 scale).
- **Bar chart** of AQI (Today + next 3 days).
- **Alerts** for unhealthy/hazardous AQI.
- **Model performance** (RMSE, MAE, R²).
- **SHAP feature importance** (Advanced Analytics).
- **Sidebar** with requirements checklist.

Scale (1–5 or US 0–500) is shown; config via `AQI_SCALE_1_5` in `.env`.
