# What’s Actually Happening — Simple Explanation

## What this project does (one sentence)

**It predicts Karachi’s AQI for the next 3 days** using weather + air-quality data and a trained model.

---

## The big picture (4 steps)

```
  [1] FETCH          [2] FEATURES         [3] TRAIN           [4] PREDICT
  ─────────          ───────────          ────────            ──────────
  OpenWeather   →    Save to MongoDB  →   Train 3 models  →   API/Dashboard
  (or Open-Meteo)   (Feature Store)      (best one wins)      shows Day+1, +2, +3
```

1. **Fetch** – Scripts get raw weather + air quality (PM2.5, etc.) from an API (OpenWeather or Open-Meteo).
2. **Features** – That raw data is turned into numbers (hour, temperature, PM2.5, US AQI, etc.) and **saved in MongoDB** (the “Feature Store”). No CSV for the main flow.
3. **Train** – Once per day, the training script **reads all those features from MongoDB**, builds targets (e.g. “tomorrow’s average AQI”), trains Ridge, Random Forest, and TensorFlow, and **saves the best model** for Day+1, Day+2, Day+3.
4. **Predict** – When you open the **dashboard** or call the **Flask API** (`/predict`), the app takes the **latest row** from the Feature Store, runs it through the 3 saved models, and returns **3 numbers**: predicted AQI for Day+1, Day+2, Day+3.

So: **same data pipeline → same Feature Store → same models → same predictions** whether you use the API or the dashboard.

---

## What each script does (one line each)

| Script | When it runs | What it does |
|--------|----------------|--------------|
| `run_hourly_feature_pipeline.py` | Every hour (or when you run it) | Fetches new data → computes features → **saves rows to MongoDB** (Feature Store). |
| `run_daily_training.py` | Once per day (or when you run it) | Loads features from MongoDB → **trains models** → saves best model for d1, d2, d3 to disk + Model Registry. |
| `backfill.py` | When you want history | Fetches past dates → same features → **fills “next day’s AQI” targets** so training has real targets (not just today’s AQI). |
| Flask API (`app.api`) | When you start the server | Serves **/predict**: reads **latest feature row** from MongoDB, runs the 3 production models, returns Day+1, Day+2, Day+3 AQI. |
| Dashboard (`app/dashboard.py`) | When you run Streamlit | Same as API: reads latest features, runs same models, **shows the same 3 predictions** in a UI. |

So “what’s happening” when you run the pipeline: **hourly script fills the Feature Store, daily script trains on that store and updates the models, and the API/dashboard just “latest row → models → 3 numbers”.**

---

## Why the numbers might look confusing

- **“Today’s AQI”** (e.g. 106) = **last stored value** in the Feature Store (from the hourly pipeline). It’s the **current/latest** US AQI we computed (e.g. from PM2.5).
- **“Predicted Day+1, Day+2, Day+3”** (e.g. 40, 50, 51) = **model output** from the **same latest row**. The model was trained to predict **daily average AQI** for the next 3 days. It doesn’t “know” that today is 106; it only sees the feature row (weather, PM2.5, hour, etc.) and outputs what it learned from past data.

So you can get:

- **Today (stored):** 106  
- **Predicted tomorrow:** 40  

That’s normal: the model is not “copying” today’s 106; it’s predicting based on patterns it learned. If in the past, when features looked like this, the **next day’s average** was often lower, it will predict lower. So “what’s happening” is: **stored value = latest observation, predictions = model’s guess for the next 3 days’ average AQI**, and they don’t have to match.

If you want predictions “around 92” (or around today’s value), we’d have to change the model or add a step that nudges predictions toward recent observed AQI (e.g. blending with today’s value).

---

## Quick checklist: “Is everything connected?”

1. **MongoDB** – Is it running and `MONGODB_URI` in `.env` correct? (Feature Store + Model Registry live here.)
2. **OpenWeather key** – In `.env`, `OPENWEATHER_API_KEY=...` so the hourly script can fetch data.
3. **Hourly pipeline** – Run `python scripts/run_hourly_feature_pipeline.py` → you should see “saved X feature rows … to Feature Store”.
4. **Daily training** – Run `python scripts/run_daily_training.py` → you should see “Training pipeline complete” and RMSE/R² for d1, d2, d3.
5. **API** – Run `python -m flask --app app.api run` → open `http://127.0.0.1:5000/predict` → you should get JSON with `forecast.day_1`, `day_2`, `day_3`.
6. **Dashboard** – Run `streamlit run app/dashboard.py` → you should see the same 3 numbers as the API.

If any step fails, the next step will be wrong (e.g. no data in MongoDB → training has nothing to learn from; no models → API returns error or empty).

---

## Summary

- **What’s happening:** Data is fetched → turned into features and stored in MongoDB → a daily job trains models on that store → the API and dashboard use the **latest feature row** and the **saved models** to show 3 AQI predictions.
- **Why it can look weird:** “Today” is the **last stored AQI**; “Day+1, +2, +3” are **model predictions** that don’t have to match today. They’re two different things: one is observed, the other is forecast.

If you tell me what’s confusing (e.g. “why is prediction 40 when today is 106?” or “where does the 92 come from?”), I can point to the exact place in the code and suggest a concrete change.
