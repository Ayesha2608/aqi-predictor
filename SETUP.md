# AQI Predictor — Setup & Manual Steps

## 1. Python environment

- **Python 3.10+** (3.11 recommended).
- Create a virtual environment and install dependencies:

```bash
cd f:\Ayesha\internship
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

---

## 2. MongoDB (Feature Store + Model Registry)

You need a running MongoDB instance. Two options:

### Option A: Local MongoDB

- Install [MongoDB Community](https://www.mongodb.com/try/download/community) and start the server.
- Default URI: `mongodb://localhost:27017` (no setup needed if using default).

### Option B: MongoDB Atlas (free tier)

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a free account.
2. Create a free cluster (e.g. M0).
3. Create a database user (username + password).
4. In **Network Access**, add your IP (or `0.0.0.0/0` for “allow from anywhere” for dev only).
5. Get the connection string: **Connect → Drivers → Python** and copy the URI (e.g. `mongodb+srv://user:pass@cluster.mongodb.net/`).
6. Create a `.env` file (see below) and set `MONGODB_URI=<your-atlas-uri>` and `MONGODB_DB=aqi_predictor`.

---

## 3. Environment variables (`.env`)

Copy `.env.example` to `.env` and edit:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac
```

Edit `.env`:

- **MONGODB_URI** — required if not using `mongodb://localhost:27017` (e.g. Atlas URI).
- **MONGODB_DB** — database name (default: `aqi_predictor`).
- **AQI_CITY** — display name (e.g. `London`).
- **AQI_LAT**, **AQI_LON** — coordinates for Open-Meteo (required). Get from [Open-Meteo](https://open-meteo.com/) or Google Maps.

Example for another city (e.g. Delhi):

```env
AQI_CITY=Delhi
AQI_LAT=28.6139
AQI_LON=77.2090
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=aqi_predictor
```

No API keys are required for Open-Meteo (weather + air quality).

---

## 4. One-time: backfill and train

Run from the project root (`f:\Ayesha\internship`):

```bash
# 1) Backfill last 14 days of features (fetches raw data and writes to MongoDB)
python scripts/backfill.py

# 2) Train models (reads features from MongoDB, saves models to models/ and registry)
python scripts/training_pipeline.py

# 3) Optional: EDA and SHAP importance (writes to metrics/)
python scripts/eda.py
```

If `backfill.py` fails with “API fetch failed”, check internet and coordinates. If “Not enough data” in training, increase the backfill range (edit `backfill.py` or run with args: `python scripts/backfill.py 2025-01-01 2025-01-29`).

---

## 5. Run the dashboard and API locally

**Streamlit dashboard:**
```bash
streamlit run app/dashboard.py
```
Browser will open. You should see “Next 3 Days” forecast and alerts if AQI is above thresholds.

**Flask API (optional):**
```bash
flask --app app.api run
```
Then open `http://127.0.0.1:5000/predict` for JSON forecast; `http://127.0.0.1:5000/health` for health check.

---

## 6. GitHub Actions (CI/CD) — manual setup

To run the **feature pipeline (hourly)** and **training pipeline (daily)** in GitHub Actions:

1. Push the repo to GitHub.
2. In the repo: **Settings → Secrets and variables → Actions**.
3. Add **Secrets**:
   - **MONGODB_URI** — your MongoDB connection string (e.g. Atlas URI). Required for both workflows.
   - **MONGODB_DB** — optional; default in code is `aqi_predictor`.
4. Add **Variables** (optional, for city/coordinates):
   - **AQI_CITY** — e.g. `London`
   - **AQI_LAT** — e.g. `51.5074`
   - **AQI_LON** — e.g. `-0.1278`

Without `MONGODB_URI`, the workflows will fail when they try to read/write MongoDB. For a fully local demo, run the scripts manually (steps 4 and 5).

---

## 7. Optional: OpenWeather / AQICN

The project uses **Open-Meteo** by default (no keys). To use OpenWeather or AQICN:

1. Get API keys from [OpenWeather](https://openweathermap.org/api) and/or [AQICN](https://aqicn.org/api/).
2. Add to `.env`: `OPENWEATHER_API_KEY=...`, `AQICN_API_KEY=...`.
3. Extend `scripts/fetch_raw_data.py` to call those APIs and merge with or replace Open-Meteo (same script, extra branches).

---

## 8. Quick checklist

- [ ] Python 3.10+ and venv created  
- [ ] `pip install -r requirements.txt`  
- [ ] MongoDB running (local or Atlas)  
- [ ] `.env` created with `MONGODB_URI`, `AQI_CITY`, `AQI_LAT`, `AQI_LON`  
- [ ] `python scripts/backfill.py`  
- [ ] `python scripts/training_pipeline.py`  
- [ ] `streamlit run app/dashboard.py`  
- [ ] (Optional) GitHub repo + `MONGODB_URI` secret for Actions  

If something fails, check: MongoDB reachable, coordinates valid, and backfill date range contains data (Open-Meteo has historical air quality for past days).
