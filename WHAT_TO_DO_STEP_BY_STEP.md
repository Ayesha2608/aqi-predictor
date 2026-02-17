# What to Do — Step by Step (Karachi AQI)

Do these steps **in order**. Use the project folder: `f:\Ayesha\internship`.

---

## Step 1: Install Python

1. Open **Command Prompt** or **PowerShell**.
2. Type: `py --version` or `python --version`
3. If you see **Python 3.10** or **3.11** (or higher), go to **Step 2**.
4. If not:
   - Go to https://www.python.org/downloads/
   - Download **Python 3.11**
   - Run the installer
   - **Tick "Add Python to PATH"**
   - Finish and close the terminal, then open a new one.

---

## Step 2: Create a virtual environment and install packages

1. Open **Command Prompt** or **PowerShell**.
2. Go to the project folder:
   ```
   cd f:\Ayesha\internship
   ```
3. Create the virtual environment:
   ```
   py -m venv .venv
   ```
   (If `py` doesn’t work, try: `python -m venv .venv`)
4. Activate it:
   - **PowerShell:** `.venv\Scripts\Activate.ps1`
   - **Command Prompt:** `.venv\Scripts\activate.bat`
   You should see `(.venv)` at the start of the line.
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Wait until it finishes.

---

## Step 3: Set up MongoDB

Choose **one** option.

### Option A — MongoDB on your PC

1. Go to: https://www.mongodb.com/try/download/community
2. Download the **Windows MSI** installer and run it.
3. Use default settings and finish. MongoDB will run as a service.
4. You will use: `mongodb://localhost:27017` (no password).
5. Go to **Step 4**.

### Option B — MongoDB Atlas (cloud, free)

1. Go to: https://www.mongodb.com/cloud/atlas
2. Click **Try Free** and create an account.
3. Create a **free cluster** (e.g. M0), then **Create**.
4. **Database Access** → **Add New Database User** → set username and password → **Add User** (save the password).
5. **Network Access** → **Add IP Address** → **Allow Access from Anywhere** (for testing) → **Confirm**.
6. **Database** → **Connect** on your cluster → **Connect your application** → copy the connection string.
7. In the string, replace `<password>` with your real password (no angle brackets).
8. You will paste this in Step 4 as `MONGODB_URI`.

---

## Step 4: Get OpenWeather API key

1. Go to: https://openweathermap.org/api
2. Sign up (free) and get an **API key**.
3. The **Air Pollution API** is included in the free plan.
4. Copy the key; you will put it in `.env` in Step 5.

---

## Step 5: Create the `.env` file

1. Open File Explorer and go to: `f:\Ayesha\internship`
2. Find **`.env.example`**.
3. Copy it and rename the copy to **`.env`** (remove `.example`).
   - Or in the terminal (with project folder as current directory):
     ```
     copy .env.example .env
     ```
4. Open **`.env`** in Notepad or any editor.
5. Edit these lines (replace with your values):

   ```
   AQI_CITY=Karachi
   AQI_LAT=24.8607
   AQI_LON=67.0011
   AQI_TIMEZONE=Asia/Karachi

   OPENWEATHER_API_KEY=paste_your_openweather_api_key_here

   MONGODB_URI=mongodb://localhost:27017
   ```
   - If you use **Atlas**: replace `MONGODB_URI` with your Atlas connection string (e.g. `mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/`).
   - Keep `MONGODB_DB=aqi_predictor` (or add it if missing).

6. Save and close `.env`.

---

## Step 6: Run the hourly feature pipeline (first time)

This fetches data from the API and saves features to the Feature Store (MongoDB). No CSV files.

1. Open **Command Prompt** or **PowerShell**.
2. Go to the project folder and activate the venv:
   ```
   cd f:\Ayesha\internship
   .venv\Scripts\Activate.ps1
   ```
3. Run the hourly script:
   ```
   python scripts/run_hourly_feature_pipeline.py
   ```
4. You should see something like: `Hourly: saved … feature rows for … (Karachi) to Feature Store.`
5. If you see an error about **API** or **key**: check `OPENWEATHER_API_KEY` in `.env`.
6. If you see an error about **MongoDB**: check that MongoDB is running and `MONGODB_URI` in `.env` is correct.

**Tip:** Run this script a few times (e.g. once now, once after an hour) so you have more hourly data before training. Or use the backfill script once (see Step 6b).

### Step 6b (optional): Backfill more history

If you want more past data for training (e.g. last 14 days with Open-Meteo when no OpenWeather key, or to fill gaps):

```
python scripts/backfill.py
```

You can also run: `python scripts/backfill.py 2025-01-01 2025-01-29` (use your desired start and end dates).

---

## Step 7: Run the daily training (first time)

This loads features from the Feature Store, trains 3 models, and sets the best one as “production”. The UI will use this best model automatically.

1. Same terminal, same folder, venv still active.
2. Run:
   ```
   python scripts/run_daily_training.py
   ```
3. Wait until it finishes. You should see lines like `Target d1: …`, `Target d2: …`, `Target d3: …` and `Training pipeline complete.`
4. If you see **“Not enough data”**: run **Step 6** again (and optionally Step 6b) to add more feature data, then run Step 7 again.

---

## Step 8: Run the dashboard (UI)

1. Same terminal (or a new one with venv activated), same folder.
2. Run:
   ```
   streamlit run app/dashboard.py
   ```
3. A browser tab should open with the **AQI Predictor** dashboard.
4. You should see:
   - **City: Karachi**
   - **Next 3 days** AQI prediction from today
   - Best model is used automatically (no model selection)
5. To stop the dashboard: press **Ctrl+C** in the terminal.

---

## Step 9 (optional): Run the Flask API

1. Open a **new** terminal (or stop the dashboard with Ctrl+C).
2. Go to the project folder and activate the venv:
   ```
   cd f:\Ayesha\internship
   .venv\Scripts\Activate.ps1
   ```
3. Run:
   ```
   flask --app app.api run
   ```
4. In the browser open: **http://127.0.0.1:5000/predict** to get the next 3 days AQI as JSON.

---

## Step 10 (optional): Automate with GitHub Actions

If the project is in a **GitHub** repo and you want the **hourly** and **daily** scripts to run automatically:

1. Push your code to GitHub.
2. In the repo: **Settings** → **Secrets and variables** → **Actions**.
3. **New repository secret:**
   - Name: `MONGODB_URI`
   - Value: your MongoDB URI (same as in `.env`; use Atlas URI for cloud).
4. **New repository secret** (for hourly pipeline):
   - Name: `OPENWEATHER_API_KEY`
   - Value: your OpenWeather API key.
5. **Variables** (optional): `AQI_CITY` = Karachi, `AQI_LAT` = 24.8607, `AQI_LON` = 67.0011.
6. The workflows in `.github/workflows/` will run:
   - **Hourly:** `run_hourly_feature_pipeline.py` every hour.
   - **Daily:** `run_daily_training.py` once per day (e.g. 06:00 UTC).

---

## Quick checklist

| Step | What to do |
|------|------------|
| 1 | Install Python 3.10+ (add to PATH) |
| 2 | `cd f:\Ayesha\internship` → create venv → activate → `pip install -r requirements.txt` |
| 3 | Install MongoDB (local or Atlas) and get connection URI |
| 4 | Get OpenWeather API key from openweathermap.org |
| 5 | Copy `.env.example` to `.env` and set Karachi, OpenWeather key, MongoDB URI |
| 6 | `python scripts/run_hourly_feature_pipeline.py` (and optionally `backfill.py`) |
| 7 | `python scripts/run_daily_training.py` |
| 8 | `streamlit run app/dashboard.py` → see Karachi next 3 days AQI |
| 9 | (Optional) `flask --app app.api run` → use /predict |
| 10 | (Optional) Add GitHub secrets/variables for hourly and daily runs |

---

## If something fails

- **“No module named …”** → Run Step 2 again: activate venv and `pip install -r requirements.txt`.
- **“OPENWEATHER_API_KEY” or API error** → Check Step 4 and Step 5; key must be in `.env` with no extra spaces.
- **“MongoDB” or connection error** → Check Step 3 and Step 5; MongoDB must be running and `MONGODB_URI` correct in `.env`.
- **“Not enough data” in training** → Run Step 6 again (and Step 6b if you want more history), then Step 7 again.
- **Dashboard shows “No feature data”** → Run Step 6 first, then Step 7, then Step 8.
