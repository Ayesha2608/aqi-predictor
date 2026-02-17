# AQI Predictor — Next 3 Days (Serverless)

End-to-end ML pipeline to **predict Air Quality Index (AQI)** for your city for the **next 3 days**, using a serverless-friendly stack: Open-Meteo APIs, MongoDB as Feature Store & Model Registry, Scikit-learn + optional TensorFlow, GitHub Actions, and Streamlit.

## Features

- **Feature pipeline**: Fetch raw weather + air quality (Open-Meteo), compute time-based and derived features (e.g. AQI change rate), store in MongoDB.
- **Historical backfill**: Generate training data for past dates.
- **Training pipeline**: Random Forest, Ridge, and **LSTM (Deep Learning)** models; metrics RMSE/MAE/R²; model registry in MongoDB; **Auto-selection of best model**.
- **Smart Calibration**: Dynamic adjustment logic to align sensor readings with ground truth (e.g. correcting high PM2.5 bias).
- **CI/CD**: GitHub Actions — feature pipeline hourly, training pipeline daily.
- **Dashboard**: Streamlit app with next 3 days forecast, hazardous/unhealthy alerts, and **deployment-ready UI controls** (daily training limit).
- **EDA**: Script to plot AQI over time, distribution, and correlations.

## Quick start

For a detailed, step-by-step master guide, see **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)**.

1. **Setup** (see [SETUP.md](SETUP.md) for details):
   - Python 3.10+, `pip install -r requirements.txt`
   - MongoDB (local or Atlas), `.env` with `MONGODB_URI`, `AQI_CITY`, `AQI_LAT`, `AQI_LON`

2. **Backfill + train** (one-time):

   ```bash
   python scripts/backfill.py
   python scripts/training_pipeline.py
   ```

3. **Run dashboard**:

   ```bash
   streamlit run app/dashboard.py
   ```

## Project layout

```
.github/workflows/   # Feature pipeline (hourly), training (daily)
app/dashboard.py     # Streamlit dashboard
app/api.py           # Flask API (/predict, /health)
config/settings.py   # Config and env
scripts/
  fetch_raw_data.py # Open-Meteo weather + air quality
  feature_pipeline.py
  backfill.py
  training_pipeline.py
  db.py              # MongoDB Feature Store & Model Registry
  eda.py             # EDA plots and summary
data/raw/            # Raw API responses (optional)
models/              # Trained models (.joblib)
metrics/             # Metrics JSON, SHAP importance, EDA figures
```

## Manual setup

- **MongoDB**: Install locally or use [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) free tier; set `MONGODB_URI` in `.env`.
- **GitHub Actions**: In repo Settings → Secrets, add `MONGODB_URI` (and optionally `MONGODB_DB`, `AQI_CITY`, `AQI_LAT`, `AQI_LON` as variables).

Full steps: **[SETUP.md](SETUP.md)**.

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for a step-by-step guide on deploying to Streamlit Cloud and setting up CI/CD.

## Report

See `IMPLEMENTATION_PLAN.md` for the 3-day implementation plan and `REPORT.md` (to be added) for a short project report.
