# 🚀 Deployment Guide

This project is designed for **Serverless Deployment**. You do not need to keep your local laptop running. The system runs entirely in the cloud.

## 1. Backend Pipelines (Automation)
**Tool:** GitHub Actions  
**What it does:** Automatically runs scripts to fetch data and train models.

### How it works:
You don't need to "deploy" this manually. It is already configured in the `.github/workflows/` folder.
- **Hourly:** Every hour, GitHub starts a virtual runner, fetches data from OpenWeather/Open-Meteo, and saves it to MongoDB.
- **Daily:** Every day at 6:00 AM UTC, GitHub starts a runner, retrains the models (Ridge, RandomForest, LSTM), and saves the best one.

### Verification:
1. Go to your GitHub Repository.
2. Click on the **"Actions"** tab.
3. You will see "Feature pipeline" and "Training pipeline" workflows listed.
4. Green checkmarks ✅ mean they are running successfully.

---

## 2. Frontend Dashboard (The Web App)
**Tool:** Streamlit Community Cloud (Free)  
**What it does:** Hosts the interactive website (`dashboard.py`) so anyone can access it via a URL.

### Steps to Deploy:
1. **Push your code** to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository: `Ayesha/internship`.
5. Set the **Main file path** to: `app/dashboard.py`.
6. **CRITICAL STEP:** Click "Advanced settings..." to add your Secrets (Environment Variables).
   Copy these exactly from your local `.env` file:
   ```env
   # Database connection
   MONGODB_URI = "your_mongodb_connection_string_here"
   MONGODB_DB = "aqi_db"

   # App Configuration
   AQI_CITY = "Karachi"
   AQI_LAT = "24.86"
   AQI_LON = "67.00"
   
   # API Keys
   OPENWEATHER_API_KEY = "your_openweather_key"
   ```
7. Click **"Deploy"**.

### Result:
Streamlit will install the libraries from `requirements.txt` and verify the code. In rarely 2-3 minutes, you will get a public URL (e.g., `https://aqi-predictor-karachi.streamlit.app/`) that you can share.

---

## 3. Architecture Summary (For External Explanation)
If you need to explain this to a supervisor, say:
> "We implemented a **Serverless MLOps Architecture**.
> 1.  **Data Ingestion**: Handled by **GitHub Actions** (Cron jobs) which fetch weather/AQI data hourly.
> 2.  **Storage**: Data is persisted in **MongoDB Atlas** (Cloud Database), serving as our Feature Store.
> 3.  **Training**: A daily CI/CD pipeline retrains our models (including LSTM) on the latest data.
> 4.  **Serving**: The user interface is a **Streamlit Web App** connected to the Cloud Feature Store, providing real-time 72-hour forecasts without manual intervention."
