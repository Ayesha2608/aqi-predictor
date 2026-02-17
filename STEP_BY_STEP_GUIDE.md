# AQI Predictor — Step-by-Step Guide

Do these steps **in order** from the project folder: `f:\Ayesha\internship`.

---

## Step 1: Install Python (if needed)

1. Check if Python is installed:
   - Open **Command Prompt** or **PowerShell**.
   - Type: `py --version` or `python --version`
   - If you see something like `Python 3.11.x`, you’re good. Skip to Step 2.
2. If Python is not found:
   - Go to https://www.python.org/downloads/
   - Download **Python 3.11** (or 3.10+).
   - Run the installer.
   - **Important:** Tick **“Add Python to PATH”**.
   - Finish the install, then close and reopen the terminal.

---

## Step 2: Create a virtual environment and install packages

1. Open **Command Prompt** or **PowerShell**.
2. Go to the project folder:
   ```text
   cd f:\Ayesha\internship
   ```
3. Create the virtual environment:
   ```text
   py -m venv .venv
   ```
   (If `py` doesn’t work, try: `python -m venv .venv`.)
4. Activate it:
   - **PowerShell:**
     ```text
     .venv\Scripts\Activate.ps1
     ```
   - **Command Prompt (cmd):**
     ```text
     .venv\Scripts\activate.bat
     ```
   When it’s active, you’ll see `(.venv)` at the start of the line.
5. Install the project dependencies:
   ```text
   pip install -r requirements.txt
   ```
   Wait until it finishes without errors.

---

## Step 3: Set up MongoDB

You need a running MongoDB. Choose **one** option.

### Option A: MongoDB on your PC (local)

1. Go to: https://www.mongodb.com/try/download/community
2. Select **Windows**, download the **MSI** installer.
3. Run the installer. Use default settings; you can leave “Install as Service” checked.
4. After install, MongoDB should be running. You’ll use: `mongodb://localhost:27017` (no username/password).
5. You’re done with Step 3. Go to Step 4.

### Option B: MongoDB Atlas (cloud, free)

1. Go to: https://www.mongodb.com/cloud/atlas
2. Click **“Try Free”** and create an account (email or Google).
3. Create a **free cluster**:
   - Choose a cloud provider and region (e.g. AWS, closest to you).
   - Cluster name can stay default (e.g. Cluster0).
   - Click **Create**.
4. Create a **database user**:
   - In the left menu: **Database Access** → **Add New Database User**.
   - Choose **Password**; set a username (e.g. `aqiuser`) and a strong password. Save the password somewhere safe.
   - Click **Add User**.
5. Allow network access:
   - In the left menu: **Network Access** → **Add IP Address**.
   - For testing you can click **“Allow Access from Anywhere”** (adds `0.0.0.0/0`). Then **Confirm**.
6. Get the connection string:
   - Go back to **Database** → **Connect** on your cluster.
   - Choose **“Connect your application”**.
   - Copy the URI. It looks like:
     ```text
     mongodb+srv://aqiuser:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
     ```
   - Replace `<password>` with the real password (no angle brackets).
   - You’ll paste this into `.env` in Step 4 as `MONGODB_URI`.

---

## Step 4: Create the `.env` file

1. In File Explorer, go to: `f:\Ayesha\internship`
2. Find the file **`.env.example`**.
3. Copy it and rename the copy to **`.env`** (no “.example”).
   - Or in Command Prompt/PowerShell (with project folder as current directory):
     ```text
     copy .env.example .env
     ```
4. Open **`.env`** in Notepad or your editor.
5. Fill in or check these lines:

   **If you use local MongoDB (Option A):**
   ```text
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB=aqi_predictor
   AQI_CITY=London
   AQI_LAT=51.5074
   AQI_LON=-0.1278
   ```

   **If you use Atlas (Option B):**  
   Use your Atlas URI and keep the rest similar:
   ```text
   MONGODB_URI=mongodb+srv://YOUR_USER:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   MONGODB_DB=aqi_predictor
   AQI_CITY=London
   AQI_LAT=51.5074
   AQI_LON=-0.1278
   ```

6. To use **another city**, change `AQI_CITY`, `AQI_LAT`, and `AQI_LON`. You can look up coordinates on https://www.latlong.net/ or Google Maps.
7. Save and close `.env`.

---

## Step 5: Run the backfill (fetch data and build features)

1. Open Command Prompt or PowerShell.
2. Go to the project folder and activate the venv (if it’s not already active):
   ```text
   cd f:\Ayesha\internship
   .venv\Scripts\Activate.ps1
   ```
   (Use `activate.bat` if you use cmd.)
3. Run the backfill script (it fetches past days from Open-Meteo and writes features to MongoDB):
   ```text
   python scripts/backfill.py
   ```
4. Wait until it finishes. You should see lines like “Backfill saved … rows for 2025-xx-xx” for several dates.
5. If you see **“API fetch failed”**: check internet and that `AQI_LAT` / `AQI_LON` in `.env` are correct.
6. If you see **“Not enough data”** later in training, you can run a longer range (replace dates with the range you want):
   ```text
   python scripts/backfill.py 2025-01-01 2025-01-29
   ```

---

## Step 6: Train the models

1. Same terminal, same folder, venv still active.
2. Run:
   ```text
   python scripts/training_pipeline.py
   ```
3. Wait for it to finish. You should see lines like “Target d1: …”, “Target d2: …”, “Target d3: …” and “Training pipeline complete.”
4. If you see **“Not enough data in Feature Store”**: run Step 5 again (backfill) and/or use a longer date range as above.
5. After this, the `models` folder will have `.joblib` or `.keras` files, and MongoDB will have the “production” models registered.

---

## Step 7: Run the Streamlit dashboard

1. Same terminal, same folder, venv still active.
2. Run:
   ```text
   streamlit run app/dashboard.py
   ```
3. A browser tab should open with the AQI Predictor dashboard (next 3 days forecast and alerts).
4. If it doesn’t open, look in the terminal for a URL like `http://localhost:8501` and open it in your browser.
5. To stop the dashboard: in the terminal press **Ctrl+C**.

---

## Step 8 (optional): Run the Flask API

1. Open a **new** Command Prompt or PowerShell (or stop the dashboard with Ctrl+C first).
2. Go to the project folder and activate the venv:
   ```text
   cd f:\Ayesha\internship
   .venv\Scripts\Activate.ps1
   ```
3. Run:
   ```text
   flask --app app.api run
   ```
4. In the browser go to:
   - `http://127.0.0.1:5000/predict` — JSON with next 3 days AQI.
   - `http://127.0.0.1:5000/health` — health check.
5. To stop: **Ctrl+C** in the terminal.

---

## Step 9 (optional): Run EDA

1. Terminal in project folder, venv active.
2. Run:
   ```text
   python scripts/eda.py
   ```
3. It will create plots and CSV in the `metrics` folder (e.g. `eda_aqi_over_time.png`, `eda_summary_stats.csv`). You can use these in your report.

---

## Step 10 (optional): GitHub Actions (CI/CD)

Only if you want the **feature pipeline every hour** and **training every day** to run on GitHub:

1. Push your project to a GitHub repository.
2. In the repo on GitHub: **Settings** → **Secrets and variables** → **Actions**.
3. Click **New repository secret**.
   - Name: `MONGODB_URI`
   - Value: your MongoDB URI (same as in `.env` — e.g. Atlas URI or `mongodb://localhost:27017` if you use a tunnel; for cloud, Atlas URI is typical).
4. Optionally add **Variables** (not secrets): `AQI_CITY`, `AQI_LAT`, `AQI_LON` with the same values as in `.env`.
5. Save. The workflows in `.github/workflows/` will run on schedule (and you can also run them manually from the **Actions** tab).

---

## Quick reference: order of steps

| Step | What |
|------|------|
| 1 | Install Python (if needed), add to PATH |
| 2 | `cd f:\Ayesha\internship` → create venv → activate → `pip install -r requirements.txt` |
| 3 | Install MongoDB locally **or** create MongoDB Atlas cluster + user + get URI |
| 4 | Copy `.env.example` to `.env` and set `MONGODB_URI`, `AQI_CITY`, `AQI_LAT`, `AQI_LON` |
| 5 | `python scripts/backfill.py` |
| 6 | `python scripts/training_pipeline.py` |
| 7 | `streamlit run app/dashboard.py` |
| 8 | (Optional) `flask --app app.api run` |
| 9 | (Optional) `python scripts/eda.py` |
| 10 | (Optional) Add `MONGODB_URI` (and vars) in GitHub repo for Actions |

If something fails at a step, check the error message (e.g. “MONGODB_URI” not set → Step 4; “Not enough data” → Step 5 with a longer date range).
