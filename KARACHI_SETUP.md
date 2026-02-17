# Karachi AQI — Setup & Flow

## Summary

- **City:** Karachi (lat 24.8607, lon 67.0011).
- **AQI source:** **Open-Meteo** (CAMS) by default — **US AQI 0–500** so numbers match weather apps (e.g. 92). If Open-Meteo is unreachable (DNS/network error), the pipeline **automatically falls back to OpenWeather** (AQI 1–5) when `OPENWEATHER_API_KEY` is set. Weather can also come from OpenWeather when the key is set.
- **No hardcoded AQI:** Today = latest from Feature Store; Day +1/+2/+3 = model predictions only.
- **Timezone:** All timestamps converted to **Asia/Karachi** and cleaned before Feature Store.
- **No CSV:** Features go only to Feature Store (MongoDB). No CSV files.
- **Hourly script:** Runs **every hour** (GitHub Actions: `0 * * * *`) → fetch → features → Feature Store only.
- **Daily script:** Runs **once per day** (e.g. `0 6 * * *` UTC) → train 3 models → best model set as production.
- **UI:** Dark bluish weather theme, bold AQI cards, smooth scroll; shows next 3 days; best model auto-selected.

---

## Step-by-step flow

### 1. API → features (hourly)

- **Script:** `scripts/run_hourly_feature_pipeline.py`
- **Runs:** Every hour (e.g. GitHub Actions cron `0 * * * *`).
- **Does:** Fetches from OpenWeather (or Open-Meteo if no key) → extracts features → cleans (timezone Asia/Karachi, drop duplicates, fill missing) → saves **only to Feature Store** (no CSV, no raw file on disk for hourly).

### 2. Feature Store

- All feature rows are stored in MongoDB (collection `aqi_features`).
- No CSV. Raw is not written to disk in the hourly run.

### 3. Training (daily)

- **Script:** `scripts/run_daily_training.py`
- **Runs:** Once per day (e.g. cron `0 6 * * *`).
- **Does:** Loads all features from Feature Store (hourly data) → trains **3 models** (Ridge, Random Forest, TensorFlow) → evaluates RMSE, MAE, R² → **best model per target (d1, d2, d3) is set as production**. UI always uses this best model.

### 4. UI (the day you run it)

- **Dashboard:** `streamlit run app/dashboard.py`
- **Shows:** Next **3 days** AQI prediction from **today**.
- **Model:** Best model is **auto-selected** (production model from Model Registry). You do not select the model yourself.

---

## OpenWeather (AQI 1–5)

- If you use **only** OpenWeather: **no normalization** (AQI stays 1–5).
- If you use **multiple** APIs later: add normalization in the feature pipeline.
- Set `OPENWEATHER_API_KEY` in `.env`. Get key from https://openweathermap.org/api.

---

## Timezone & cleaning

- **Timezone:** All datetimes are converted to **Asia/Karachi** (`config.settings.TIMEZONE`).
- **Cleaning:** Done in `scripts/data_cleaning.py` and applied before saving to Feature Store:
  - Convert `timestamp` / `datetime_iso` to Asia/Karachi.
  - Drop duplicates (e.g. by timestamp, city).
  - Fill missing numeric values with median.
  - Clip AQI to valid range (1–5 for OpenWeather, 0–500 for US scale).

---

## Run full flow (one-time or after changes)

Use **Python 3.11** (or the interpreter where you installed `pymongo`, `streamlit`, `plotly`, etc.). From project root:

```powershell
# 0. (Optional) Backfill last 7–14 days so training has enough hourly rows for shift-based targets
python scripts/backfill.py

# 1. Fetch + features → Feature Store (hourly)
python scripts/run_hourly_feature_pipeline.py

# 2. Train models → Model Registry (best model per day)
python scripts/run_daily_training.py

# 3. Verify: compare Open-Meteo live vs Feature Store vs model
python scripts/verify_karachi_aqi.py

# 4. Start dashboard (then open http://localhost:8501 in browser)
python -m streamlit run app/dashboard.py
```

If your default `python` points to a venv with broken numpy, use the full path to your working Python, e.g.:

```powershell
& "C:\Users\...\Python\Python311\python.exe" scripts/run_hourly_feature_pipeline.py
```

---

## Scripts summary

| Script | When | What |
|--------|------|------|
| `run_hourly_feature_pipeline.py` | Every hour | API → features → clean → Feature Store only |
| `run_daily_training.py` | Once per day | Feature Store → train 3 models → set best as production |
| `backfill.py` | One-time (optional) | Historical dates → features → Feature Store (uses Open-Meteo for past dates if no OpenWeather history) |
| Dashboard | When you open UI | Load production model → show next 3 days AQI (best model auto) |

---

## Getting predictions that match Karachi (e.g. ~92)

1. **Use Open-Meteo for AQI** (default): `USE_OPENMETEO_AQI=true` so pipeline fetches **US AQI 0–500** (same scale as weather apps).
2. **Backfill then train:** Run backfill for the last few days, then daily training, so the model is trained on US AQI targets (shift-based: AQI 24h/48h/72h later).
3. **Hourly pipeline:** Run `python scripts/run_hourly_feature_pipeline.py` so Feature Store has fresh Open-Meteo AQI for “Today”.
4. **Verify:** Run `python scripts/verify_karachi_aqi.py` to compare Open-Meteo live vs Feature Store vs model predictions. Compare with [aqicn.org/city/karachi](https://aqicn.org/city/karachi/) or IQAir.

If you still see 1–5 or 5.0, the store/model were likely built with old OpenWeather (1–5) data. Re-run backfill + training with current code so everything is US AQI.

---

## Troubleshooting: Open-Meteo unreachable (getaddrinfo failed)

If you see `Failed to resolve 'air-quality.api.open-meteo.com'` (DNS/network/firewall):

- **Automatic fallback:** With `OPENWEATHER_API_KEY` set in `.env`, the pipeline will use **OpenWeather** for AQI (1–5 scale) when Open-Meteo fails, so the hourly script still runs.
- **Force OpenWeather only:** Set `USE_OPENMETEO_AQI=false` in `.env` so only OpenWeather is used (no Open-Meteo call). AQI will be 1–5; 92 on a weather app ≈ index 2 (Fair).

---

## .env for Karachi (Open-Meteo AQI + optional OpenWeather weather)

```env
AQI_CITY=Karachi
AQI_LAT=24.8607
AQI_LON=67.0011
AQI_TIMEZONE=Asia/Karachi
USE_OPENMETEO_AQI=true
OPENWEATHER_API_KEY=your_key_optional
MONGODB_URI=your_mongodb_uri
MONGODB_DB=aqi_predictor
```
