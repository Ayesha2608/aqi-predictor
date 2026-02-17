# 📘 Complete Project Guide (From Scratch)

This guide covers everything you need to set up, run, and deploy the **AQI Predictor** project.

---

## 🏗️ Phase 1: Retrieve Credentials
Since you already have GitHub, MongoDB, and OpenWeather accounts, you just need to get your keys ready:

1.  **MongoDB Connection String**:
    - Log in to [MongoDB Atlas](https://cloud.mongodb.com/).
    - Go to your Database -> Click **Connect**.
    - Click **Drivers** (Python).
    - **Copy the connection string**. It looks like: `mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/`
    - *Replace `<password>` with your actual database user password.*

2.  **OpenWeather API Key**:
    - Log in to [OpenWeather](https://home.openweathermap.org/api_keys).
    - Copy your **active API Key**.

3.  **GitHub Repo**:
    - Create a **new repository** on GitHub (e.g. `aqi-predictor`).
    - Keep the URL handy (e.g. `https://github.com/YourUser/aqi-predictor.git`).

---

## 💻 Phase 2: Local Setup
Get the project running on your machine first.

### 1. Clone & Install
```bash
# Clone the repository
git clone https://github.com/Ayesha/internship.git
cd internship

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Secrets
Create a file named `.env` in the root folder and paste this content (fill in your real keys):
```ini
# Database (CRITICAL)
MONGODB_URI="your_mongodb_connection_string_here"
MONGODB_DB="aqi_db"

# Location (Karachi Defaults)
AQI_CITY="Karachi"
AQI_LAT="24.86"
AQI_LON="67.00"

# APIs
OPENWEATHER_API_KEY="your_openweather_key"
```

---

## ⚙️ Phase 3: Data Initialization (The "Cold Start")
The system needs historical data to train the models.

### 1. Run Backfill
Fetch past 14 days of weather/AQI data to build a training dataset.
```bash
python scripts/backfill.py
```
*Output: "Backfill saved X rows..."*

### 2. Train Models (First Run)
Train Ridge, RandomForest, and LSTM models on this new data.
```bash
python scripts/training_pipeline.py
```
*Output: "Training complete. Best model: LSTM..."*

---

## 🖥️ Phase 4: Local Testing
Verify everything works before deploying.

### 1. Run Feature Pipeline (Simulate Hourly Job)
```bash
python scripts/run_hourly_feature_pipeline.py
```

### 2. Launch Dashboard
```bash
streamlit run app/dashboard.py
```
- A browser tab will open at `http://localhost:8501`.
- Check if you see the **Forecast Table**, **Graphs**, and **Map**.

---

## 🚀 Phase 5: Deployment (Go Live)
Move from "Local" to "Cloud".

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Configure GitHub Actions (CI/CD)
**IMPORTANT:** Your local `.env` file is **ignored** by Git for security. GitHub cannot see it. You MUST add these secrets manually:

- Go to your GitHub Repo -> **Settings** -> **Secrets and variables** -> **Actions**.
- Click **"New repository secret"**.
- Add each key exactly as it is in your `.env` file:
  - `MONGODB_URI`
  - `MONGODB_DB`
  - `OPENWEATHER_API_KEY`
- **Done!** Now GitHub can run the pipelines.

### 3. Deploy Dashboard to Streamlit Cloud
- Go to [share.streamlit.io](https://share.streamlit.io/).
- Click **"New app"**.
- Select your repo.
- **CRITICAL STEP:** Streamlit also cannot see your `.env`.
- Click **"Advanced settings"** -> **Secrets**.
- Copy the content of your local `.env` and paste it here.
- Click **"Deploy"**.

---

## ✅ Phase 6: Maintenance
- **Monitor**: Check GitHub "Actions" tab to ensure pipelines are green.
- **Update**: If you change code, just `git push`. Streamlit updates automatically.
- **Check**: Visit your Streamlit URL to see the latest forecasts.

**That's it! Your AQI Prediction System is fully autonomous.** 🚀
