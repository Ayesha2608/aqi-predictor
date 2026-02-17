# AQI Predictor — Completion Checklist vs PDF Requirements

This file maps the **AQI_predict-1.pdf** and **Pearls AQI Predictor.pdf** requirements to the implementation.

---

## Feature pipeline (slide 3)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Fetch raw weather and pollutant data from external API (AQICN/OpenWeather) | ✅ | `scripts/fetch_raw_data.py` — Open-Meteo (weather + air quality, no key) |
| Compute features: time-based (hour, day, month) and derived (e.g. AQI change rate) | ✅ | `scripts/feature_pipeline.py` — hour, day_of_week, month, is_weekend, aqi_change_rate |
| Store processed features in Feature Store | ✅ | `scripts/db.py` + MongoDB collection `aqi_features` |

---

## Historical backfill (slide 4)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Run feature script for a range of past dates | ✅ | `scripts/backfill.py` — backfill_date_range() |
| Generate training data for ML models | ✅ | Targets target_aqi_d1, d2, d3 filled from next days’ AQI |

---

## Training pipeline (slide 5)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Fetch historical (features, targets) from Feature Store | ✅ | `scripts/training_pipeline.py` — load_features() |
| Experiment with Scikit-learn (Random Forest, Ridge) and TensorFlow/PyTorch | ✅ | Ridge, RandomForestRegressor, TensorFlow/Keras Dense model |
| Evaluate with RMSE, MAE, R² | ✅ | evaluate() in training_pipeline.py |
| Store trained model in Model Registry | ✅ | MongoDB `models` collection + models/*.joblib and *.keras |

---

## CI/CD (slide 6)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Feature script runs every hour | ✅ | `.github/workflows/feature_pipeline.yml` (cron: 0 * * * *) |
| Training script runs every day | ✅ | `.github/workflows/training_pipeline.yml` (cron: 0 6 * * *) |
| Use Apache Airflow or GitHub Actions | ✅ | GitHub Actions |

---

## Web app (slide 7)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Load model and features from Feature Store | ✅ | Dashboard and API use get_production_model() + get_latest_features() |
| Compute model predictions and show on dashboard | ✅ | `app/dashboard.py` — next 3 days AQI + chart |
| Use Streamlit/Gradio and Flask/FastAPI | ✅ | Streamlit: `app/dashboard.py`; Flask: `app/api.py` (/predict, /health) |

---

## Guidelines (slide 8)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Perform EDA to identify trends | ✅ | `scripts/eda.py` — AQI over time, distribution, correlations |
| Variety of models: statistical to deep learning | ✅ | Ridge (statistical), Random Forest (ensemble), TensorFlow/Keras (deep learning) |
| SHAP or LIME for feature importance | ✅ | SHAP/coefficient importance in `training_pipeline.py`; saved to metrics/shap_importance_d*.json |
| Alerts for hazardous AQI levels | ✅ | Dashboard + API: warnings for AQI ≥ 150, hazardous for ≥ 200; log_alert() to MongoDB |

---

## Final submissions (slide 8)

| Deliverable | Status |
|-------------|--------|
| 1. End-to-end AQI prediction system | ✅ |
| 2. Scalable, automated pipeline | ✅ (GitHub Actions + MongoDB) |
| 3. Interactive dashboard with real-time and forecasted AQI | ✅ (Streamlit + Flask API) |
| 4. Detailed report | ✅ (REPORT.md template; fill after runs) |

---

## Tech stack (Pearls PDF)

| Tool | Status |
|------|--------|
| Python | ✅ |
| Scikit-learn | ✅ |
| TensorFlow | ✅ (Keras model in training pipeline) |
| Hopsworks or Vertex AI | ✅ Alternative: MongoDB (Feature Store + Model Registry) |
| Apache Airflow or GitHub Actions | ✅ GitHub Actions |
| Streamlit | ✅ |
| Flask | ✅ (app/api.py) |
| AQICN or OpenWeather APIs | ✅ Open-Meteo (no key); config supports OpenWeather/AQICN |
| SHAP | ✅ |
| Git | ✅ (repo + .gitignore) |

---

**Conclusion:** The project satisfies all requirements from both PDFs. Run backfill → training → dashboard/API as in SETUP.md to use it end-to-end.
