# MongoDB Atlas Setup — Step-by-Step Guide

This guide walks you through setting up **MongoDB Atlas** (free cloud database) and connecting it to your AQI project.

---

## Step 1: Create an Atlas account

1. Open your browser and go to: **https://www.mongodb.com/cloud/atlas**
2. Click the green **"Try Free"** or **"Get started free"** button.
3. Sign up using:
   - **Email** (enter your email and a password), or
   - **Google** (click "Sign in with Google" and use your Google account).
4. Fill in any extra details if asked (e.g. name, company — you can skip or use "Personal").
5. Click **Create your Atlas account** (or similar). You’re now logged in.

---

## Step 2: Create a free cluster

1. After login you’ll see the **Atlas dashboard**.
2. Choose **"Build a Database"** or **"Create"** → **"Create a cluster"**.
3. Select the **FREE** tier (usually **"M0 Sandbox"** or **"Shared"** — 512 MB, free forever).
4. **Cloud Provider & Region:**
   - Choose **AWS** (or **Google Cloud** / **Azure** if you prefer).
   - Pick a **region** close to you (e.g. **Mumbai** or **Singapore** for Pakistan).
5. **Cluster Name:** Leave default (e.g. **Cluster0**) or type something like **aqi-cluster**.
6. Click **"Create"** or **"Create Cluster"**.
7. Wait 1–3 minutes until the cluster status shows **Available** (green).

---

## Step 3: Create a database user (username + password)

You need a user so your app can connect to the database.

1. A popup may appear: **"Security Quickstart"** or **"Create Database User"**.
   - If you see it, use it. Otherwise go to **Security** → **Database Access** in the left menu.
2. Click **"+ Add New Database User"** (or **"Create Database User"**).
3. **Authentication Method:** leave **"Password"**.
4. **Username:** type something you’ll remember, e.g. **aqiuser** (no spaces).
5. **Password:** click **"Autogenerate Secure Password"** — **copy and save this password** somewhere safe (e.g. Notepad). You’ll need it for the connection string.
   - Or choose your own strong password and remember it.
6. **Database User Privileges:** leave **"Atlas admin"** or choose **"Read and write to any database"**.
7. Click **"Add User"** (or **"Create Database User"**).

---

## Step 4: Allow network access (so your app can reach Atlas)

By default Atlas blocks all IPs. You need to allow your computer (and later GitHub Actions) to connect.

1. In the left menu click **"Network Access"** (under **Security**).
2. Click **"+ Add IP Address"** (or **"Add IP Address"**).
3. For **testing from your PC** you have two options:
   - **Option A (easiest for testing):** Click **"Allow Access from Anywhere"**.
     - This adds IP **0.0.0.0/0** (any IP). Fine for learning; for production you’d restrict IPs later.
   - **Option B:** Click **"Add Current IP Address"** so only your current IP is allowed.
4. Optionally add a **comment** like "My PC" or "AQI project".
5. Click **"Confirm"** (or **"Add IP Address"**).
6. Wait until the new entry shows status **Active** (can take a minute).

---

## Step 5: Get your connection string

This is the URI you’ll put in `.env` as `MONGODB_URI`.

1. In the left menu go back to **"Database"** (or **"Database Deployments"**).
2. Find your cluster (e.g. **Cluster0**) and click **"Connect"** next to it.
3. Choose **"Connect your application"** (or **"Drivers"**).
4. **Driver:** Python. **Version:** 3.12 or 3.6+ (any is fine).
5. You’ll see a connection string like:
   ```text
   mongodb+srv://aqiuser:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
6. **Copy** this whole string (Ctrl+C).
7. **Important:** Replace **`<password>`** with the **real password** you created in Step 3.
   - Example: if your password is `MyPass123`, the string becomes:
     ```text
     mongodb+srv://aqiuser:MyPass123@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
     ```
   - If your password has special characters (e.g. `#`, `@`, `%`), they must be **URL-encoded**:
     - `@` → `%40`
     - `#` → `%23`
     - `%` → `%25`
     - Or use the **Autogenerate Secure Password** from Atlas (usually no special chars).
8. This final string is your **MONGODB_URI**. You’ll paste it in Step 6.

---

## Step 6: Put the connection string in your `.env` file

1. Open **File Explorer** and go to: **f:\Ayesha\internship**
2. Find the file **`.env`** (if you don’t have it, copy **`.env.example`** and rename the copy to **`.env`**).
3. Open **`.env`** in Notepad or any editor.
4. Find the line that says **MONGODB_URI** (or add it if missing).
5. Set it to your connection string **with your real password** (no quotes, no spaces around `=`):

   ```text
   MONGODB_URI=mongodb+srv://aqiuser:YOUR_REAL_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   MONGODB_DB=aqi_predictor
   ```

   Replace:
   - **YOUR_REAL_PASSWORD** with the database user password from Step 3.
   - **aqiuser** with your username if you used a different one.
   - **cluster0.xxxxx.mongodb.net** with the host from your own connection string (from Step 5).
6. Make sure **MONGODB_DB=aqi_predictor** is there (same line or next line).
7. **Save** the file (Ctrl+S) and close it.

---

## Step 7: Test the connection from your project

1. Open **Command Prompt** or **PowerShell**.
2. Go to your project folder and activate the virtual environment:
   ```text
   cd f:\Ayesha\internship
   .venv\Scripts\Activate.ps1
   ```
3. Run the hourly script (it will connect to Atlas and write to the Feature Store):
   ```text
   python scripts/run_hourly_feature_pipeline.py
   ```
4. If it runs without errors and you see something like **"Hourly: saved … feature rows … to Feature Store"**, your Atlas connection is working.
5. If you get an error:
   - **"Authentication failed"** → Wrong username or password in `MONGODB_URI`; check Step 3 and Step 6. Encode special characters in the password (see Step 5).
   - **"Could not connect"** or **"timed out"** → Check Step 4 (Network Access); make sure **0.0.0.0/0** or your current IP is allowed.
   - **"No module named pymongo"** → Activate venv and run: `pip install -r requirements.txt`

---

## Quick checklist (Atlas only)

| Step | What you did |
|------|----------------|
| 1 | Create account at mongodb.com/cloud/atlas |
| 2 | Create free M0 cluster, choose region |
| 3 | Create database user, save password |
| 4 | Network Access → Add IP → Allow from Anywhere (or current IP) |
| 5 | Database → Connect → Connect your application → copy connection string, replace `<password>` |
| 6 | Put final URI in `.env` as MONGODB_URI=... and MONGODB_DB=aqi_predictor |
| 7 | Run `python scripts/run_hourly_feature_pipeline.py` to test |

---

## Where to find things in Atlas later

- **Dashboard / Clusters:** https://cloud.mongodb.com/ (after login).
- **Change password:** Security → Database Access → your user → Edit → Password.
- **Connection string again:** Database → your cluster → Connect → Connect your application.
- **See data:** Database → your cluster → **Browse Collections** (after your app has written data; database name will be **aqi_predictor**, collections **aqi_features**, **models**).

If you tell me the exact error message you get (e.g. after Step 7), I can tell you exactly what to fix.
