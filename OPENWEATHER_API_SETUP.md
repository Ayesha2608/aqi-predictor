# OpenWeather API Setup — Step-by-Step Guide

This guide walks you through getting an **OpenWeather API key** and using it in your AQI project (Karachi). The **Air Pollution API** (AQI 1–5) is included in the free plan.

---

## Step 1: Go to OpenWeather

1. Open your browser and go to: **https://openweathermap.org**
2. In the top menu, click **"API"** or go directly to: **https://openweathermap.org/api**
3. You’ll see the list of APIs (Current Weather, Forecast, Air Pollution, etc.). **Air Pollution** is what we use for AQI.

---

## Step 2: Sign up (create an account)

1. On the API page, click **"Sign In"** (top right) or **"Sign Up"**.
2. If you don’t have an account, click **"Create an Account"** or go to: **https://home.openweathermap.org/users/sign_up**
3. Fill in:
   - **Username** (choose a username, e.g. your email or a nickname)
   - **Email** (your real email)
   - **Password** (choose a strong password)
4. Accept the terms and conditions (tick the box if shown).
5. Click **"Create Account"**.
6. Check your email and **verify your account** (click the link OpenWeather sends you). You must verify before the API key works.

---

## Step 3: Get your API key

1. After signing in, go to: **https://home.openweathermap.org/api_keys**
   - Or: **OpenWeather home** → click your **username** (top right) → **"My API keys"**.
2. You’ll see a section **"API keys"**.
3. If there’s no key yet:
   - Click **"Generate"** or **"Create Key"**.
   - You can give it a name (e.g. **AQI Project**) or leave default.
   - Click **"Generate"**.
4. Your **API key** (a long string of letters and numbers) will appear.
5. **Copy** the key (Ctrl+C) and save it somewhere safe (e.g. Notepad). You’ll paste it into `.env` in the next step.
6. **Note:** New keys can take **10 minutes to a few hours** to activate. If you get "Invalid API key" at first, wait a bit and try again.

---

## Step 4: Confirm Air Pollution API is available

1. The **Air Pollution API** is included in the **free** plan (no payment needed).
2. On **https://openweathermap.org/api/air-pollution** you can see:
   - **Current air pollution** (real-time AQI and pollutants)
   - **Forecast** (up to 4 days, hourly)
3. Your same API key works for **weather** and **air pollution**; no extra signup.

---

## Step 5: Put the API key in your `.env` file

1. Open **File Explorer** and go to: **f:\Ayesha\internship**
2. Open the file **`.env`** in Notepad or any editor.
3. Find the line:
   ```text
   OPENWEATHER_API_KEY=your_openweather_api_key_here
   ```
4. **Replace** `your_openweather_api_key_here` with your **actual API key** (paste it).  
   Example (use your own key, not this one):
   ```text
   OPENWEATHER_API_KEY=a1b2c3d4e5f6789012345678abcdef12
   ```
5. Rules:
   - **No quotes** around the key
   - **No spaces** before or after the `=`
   - **No spaces** at the start or end of the key
6. **Save** the file (Ctrl+S) and close it.

---

## Step 6: Test that it works

1. Open **Command Prompt** or **PowerShell**.
2. Go to your project folder and activate the virtual environment:
   ```text
   cd f:\Ayesha\internship
   .venv\Scripts\Activate.ps1
   ```
3. Run the hourly feature pipeline (it will call OpenWeather and save features to MongoDB):
   ```text
   python scripts/run_hourly_feature_pipeline.py
   ```
4. If it works, you’ll see something like: **"Hourly: saved … feature rows for … (Karachi) to Feature Store."**
5. If you see an error:
   - **"Invalid API key"** or **401** → Key wrong in `.env`, or key not activated yet (wait 10–60 minutes after creating it). Check Step 3 and Step 5.
   - **"OPENWEATHER_API_KEY"** or **KeyError** → Make sure the line in `.env` is exactly `OPENWEATHER_API_KEY=your_key` with no typos.
   - **"No module named …"** → Activate venv and run: `pip install -r requirements.txt`

---

## Quick checklist

| Step | What you did |
|------|----------------|
| 1 | Go to openweathermap.org/api |
| 2 | Sign up and **verify your email** |
| 3 | My API keys → Generate → **Copy** the key |
| 4 | Confirm Air Pollution API is on free plan (no extra step) |
| 5 | In `.env`: `OPENWEATHER_API_KEY=paste_your_key_here` and save |
| 6 | Run `python scripts/run_hourly_feature_pipeline.py` to test |

---

## Where to find things later

- **Sign in:** https://home.openweathermap.org  
- **API keys:** https://home.openweathermap.org/api_keys  
- **Air Pollution API docs:** https://openweathermap.org/api/air-pollution  
- **Pricing (free tier):** https://openweathermap.org/price — free plan includes Air Pollution; no credit card for free tier.

If you get a specific error message when running the script, paste it here and I can tell you exactly what to fix.
