# AQI Predictor — 3-Day Implementation Plan

## Objective (from AQI_predict-1.pdf & Pearls AQI Predictor.pdf)

Build a **100% serverless** end-to-end ML pipeline that:
- Predicts **Air Quality Index (AQI)** for your city for the **next 3 days**
- Uses automated data collection → feature engineering → model training → real-time predictions
- Exposes an interactive **web dashboard** with forecasts and alerts

---

## Tech Stack (Simplified for 3 Days)

| Component           | Primary choice        | Fallback / note                         |
|--------------------|------------------------|-----------------------------------------|
| Feature Store      | Hopsworks or Vertex AI| **MongoDB** (collections for features)   |
| Model Registry     | Hopsworks / Vertex    | **MongoDB** (store model artifacts/paths)|
| CI/CD              | **GitHub Actions**    | Apache Airflow (heavier; skip for 3 days)|
| Data APIs          | OpenWeather + AQICN (or Open-Meteo / WAQI) | Free tiers, minimal setup   |
| ML                 | Scikit-learn + TensorFlow (1 simple model) | Add more models if time     |
| Explainability     | **SHAP**              | LIME if SHAP is slow                    |
| Web app            | **Streamlit** (fast)   | Optional Flask/FastAPI for API only     |
| Version control    | **Git**               | —                                       |

---

## 3-Day Sprint Breakdown

### Day 1 — Data + Feature Pipeline + Storage

**Goal:** Raw data in, features and targets stored, backfill script ready.

1. **Data ingestion (2–3 h)**  
   - Pick one city (e.g. your city or a fixed one).  
   - Use **OpenWeather** (weather) + **AQICN** or **Open-Meteo Air Quality** (pollutants/AQI).  
   - Script: `scripts/fetch_raw_data.py` — fetch current + historical (past 30–60 days) if API allows.  
   - Save raw responses (e.g. JSON) to disk or a `raw/` folder; optional: push to MongoDB `raw` collection.

2. **Feature engineering (2–3 h)**  
   - Script: `scripts/feature_pipeline.py`.  
   - Input: raw weather + AQI/pollutant data.  
   - **Features:**  
     - Time: `hour`, `day_of_week`, `month`, `is_weekend`.  
     - Weather: temp, humidity, pressure, wind speed (and lags if easy).  
     - Pollutants: PM2.5, PM10, O3, etc., and **AQI change rate** (e.g. difference from previous period).  
   - **Target:** AQI for next 1, 2, 3 days (or next 24h/48h/72h); one target per horizon or one “next 3-day max/avg”.

3. **Storage (1–2 h)**  
   - If **Hopsworks/Vertex**: create feature group + write features.  
   - If **MongoDB**:  
     - Collection `features`: one document per (timestamp, city) with feature dict + targets.  
     - Optional: `metadata` collection for pipeline run info.  
   - Function: `save_features(features_df)` that works with either backend.

4. **Backfill (1 h)**  
   - Script: `scripts/backfill.py` — loop over past dates, call fetch + feature pipeline, save to Feature Store / MongoDB.  
   - Aim for at least 2–4 weeks of history for a minimal viable model.

**Day 1 deliverable:** Running `backfill.py` fills Feature Store / MongoDB with historical features and targets.

---

### Day 2 — Training Pipeline + Model Registry + CI/CD

**Goal:** Trained model, metrics, model stored; hourly/daily automation in place.

1. **Training pipeline (3–4 h)**  
   - Script: `scripts/training_pipeline.py`.  
   - Load historical (features + targets) from Feature Store / MongoDB.  
   - Train/compare:  
     - **Minimum:** Random Forest + Ridge (Scikit-learn).  
     - **If time:** one TensorFlow model (e.g. Dense small NN).  
   - Metrics: **RMSE, MAE, R²** (and optionally MAPE).  
   - Pick best model (e.g. by RMSE or R²), save artifact (e.g. `joblib`/`pickle` or TF SavedModel).  
   - Log metrics (e.g. in a `metrics/` JSON or in MongoDB `model_runs`).

2. **Model Registry (1 h)**  
   - If Hopsworks/Vertex: register model + version.  
   - If **MongoDB**:  
     - Collection `models`: `{ model_id, version, path_or_blob, metrics, created_at }`.  
     - Store file in disk (e.g. `models/`) and save path in DB; or use GridFS for binary.  
   - “Production” = latest model or the one marked `production: true`.

3. **CI/CD with GitHub Actions (1–2 h)**  
   - **Hourly:** `feature_pipeline.py` (fetch latest data → compute features → save).  
   - **Daily:** `training_pipeline.py` (retrain and register new model).  
   - Workflows: `.github/workflows/feature_pipeline.yml`, `.github/workflows/training_pipeline.yml`.  
   - Use repo secrets for API keys (OpenWeather, AQICN, MongoDB URI if needed).

**Day 2 deliverable:** Pushing to main runs feature pipeline hourly and training daily; new model appears in Model Registry / MongoDB.

---

### Day 3 — Dashboard + EDA + SHAP + Alerts + Report

**Goal:** Working app, explainability, alerts, and a short report.

1. **Web dashboard (2–3 h)**  
   - **Streamlit** app: `app/dashboard.py` (or `streamlit_app.py`).  
   - Load **latest model** from Model Registry / MongoDB.  
   - Load **latest features** from Feature Store / MongoDB.  
   - Run prediction for **next 3 days** (use same feature logic; for future dates use forecasted weather if available or last-known).  
   - Show:  
     - Current AQI + 3-day forecast (line/bar chart).  
     - Simple metrics (RMSE, MAE, R²) from last training run.  
   - Optional: Flask/FastAPI endpoint that returns JSON predictions for the same model (e.g. for alerts or external use).

2. **EDA (1 h)**  
   - Notebook or script: `notebooks/eda.ipynb` or `scripts/eda.py`.  
   - Plots: AQI over time, distribution, correlation with weather, seasonality.  
   - Save 3–5 key figures; reference them in the report.

3. **SHAP (1 h)**  
   - In training pipeline or a separate script: compute SHAP values for the chosen model (e.g. TreeExplainer for RF).  
   - Save feature importance plot; show in dashboard or in report.

4. **Alerts (1 h)**  
   - In dashboard or API: if predicted AQI > threshold (e.g. 150 or 200), show **alert** (badge/banner).  
   - Optional: log to MongoDB `alerts` or send a simple notification (e.g. email later).

5. **Report (1 h)**  
   - Short document: what you built, how to run pipelines and app, EDA summary, model comparison (RMSE/MAE/R²), SHAP summary, and how alerts work.  
   - Format: Markdown or PDF in repo (e.g. `REPORT.md` or `docs/REPORT.md`).

**Day 3 deliverable:** One-click (or one-command) dashboard, EDA + SHAP in repo, alerts on dashboard, and a concise report.

---

## Project Structure (Suggested)

```
internship/
├── .github/workflows/
│   ├── feature_pipeline.yml   # hourly
│   └── training_pipeline.yml  # daily
├── app/
│   └── dashboard.py           # Streamlit
├── scripts/
│   ├── fetch_raw_data.py
│   ├── feature_pipeline.py
│   ├── backfill.py
│   ├── training_pipeline.py
│   └── eda.py
├── notebooks/
│   └── eda.ipynb
├── models/                    # or Model Registry
├── data/raw/                  # optional raw cache
├── config/
│   └── settings.py            # API keys, city, MongoDB URI
├── requirements.txt
├── REPORT.md
├── IMPLEMENTATION_PLAN.md     # this file
├── AQI_predict-1.pdf
└── Pearls AQI Predictor.pdf
```

---

## MongoDB as Feature Store + Model Registry (Concrete)

- **Feature Store:**  
  - Collection `aqi_features`.  
  - Schema: `timestamp`, `city`, `feature_1`, …, `target_aqi_d1`, `target_aqi_d2`, `target_aqi_d3` (or one target column + horizon).  
  - Index on `(city, timestamp)` for fast range queries for training and “latest” for app.

- **Model Registry:**  
  - Collection `models`.  
  - Fields: `model_id`, `version`, `path` (e.g. `models/rf_v1.joblib`), `metrics` (RMSE, MAE, R²), `created_at`, `production` (boolean).  
  - App and training script both query “latest production” or “latest by date”.

- **Secrets:**  
  - Store MongoDB URI and API keys in GitHub Actions secrets; read via env in scripts.

---

## What to Cut If Time Is Short

- Second/third ML framework (e.g. only Scikit-learn, no TensorFlow).  
- Full Airflow; stick to GitHub Actions.  
- Multiple cities (single city is enough).  
- LIME (use SHAP only).  
- Flask/FastAPI if Streamlit is enough for the demo.

---

## What Must Be There (From PDFs)

- Feature pipeline with time-based + derived (e.g. AQI change rate) features.  
- Historical backfill and training data in Feature Store (or MongoDB).  
- At least 2 model types (e.g. Random Forest + Ridge), metrics RMSE, MAE, R².  
- Model in Model Registry (or MongoDB).  
- Automated runs: feature hourly, training daily.  
- Web app loading model + features and showing next 3 days AQI.  
- EDA, SHAP (or LIME), and hazardous AQI alerts.  
- Git + report.

---

## Next Steps

1. Create repo structure and `requirements.txt` (Python, pandas, scikit-learn, tensorflow, pymongo, streamlit, shap, requests, python-dotenv).  
2. Implement Day 1 (fetch → features → store → backfill).  
3. Implement Day 2 (training → registry → GitHub Actions).  
4. Implement Day 3 (Streamlit dashboard → EDA → SHAP → alerts → report).

If you want, the next step can be generating the initial `requirements.txt` and skeleton scripts (fetch, feature_pipeline, backfill, training_pipeline) so you can start coding immediately.
