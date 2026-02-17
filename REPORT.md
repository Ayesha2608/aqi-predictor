# 🌫️ Project Report: Karachi AQI Forecast Dashboard

## 1. Executive Summary
The **Karachi AQI Forecast Dashboard** is an end-to-end MLOps solution designed to monitor and predict air quality in Karachi, Pakistan. The system provides real-time monitoring and a high-precision **72-hour forecast** for PM2.5 levels, enabling citizens to make informed health decisions.

## 2. Technical Objective
The primary goal was to build a robust, automated pipeline that:
- Fetches real-time weather and air quality data.
- Processes features using a professional Feature Store approach.
- Trains and evaluates multiple ML models.
- Deploys a glassmorphic, mobile-responsive dashboard with automated daily retraining.

## 3. Technology Stack
- **Dashboard**: [Streamlit](https://streamlit.io/) (with custom CSS for Glassmorphism).
- **Database**: [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (Feature Store & Model Registry).
- **Automation**: [GitHub Actions](https://github.com/features/actions) (Serverless CI/CD and MLOps).
- **Machine Learning**: 
  - `scikit-learn`: Random Forest & Ridge Regression.
  - `TensorFlow/Keras`: LSTM (Long Short-Term Memory) for temporal sequences.
  - `Joblib`: Model serialization.
- **Data Source**: [OpenWeatherMap API](https://openweathermap.org/api).

---

## 4. Algorithms & Techniques

### Algorithms Used
| Algorithm | Role | Why? |
| :--- | :--- | :--- |
| **Random Forest** | Main Regressor | Handles non-linear relationships and feature interactions best (Current Top Performer). |
| **LSTM (Deep Learning)** | Sequence Learner | Captures long-term temporal dependencies in air quality trends. |
| **Ridge Regression** | Baseline | Provides a stable, linear baseline to ensure ML models are actually adding value. |

### Advanced Techniques
1. **Multi-Point Inference**: Instead of a single prediction, the model runs a recursive loop to generate a continuous 72-hour trend.
2. **Rolling Median Filter**: Applied a window=3 median filter to the final forecast to suppress spurious single-point anomalies (spikes).
3. **Sensor Calibration**: Implemented logic to align raw satellite/model data with ground-truth local monitoring stations.
4. **Time-Based Feature Engineering**: Extracted hour, day, and cyclic (sin/cos) features to capture daily air quality rhythms.

---

## 5. MLOps Pipelines

### 🔄 Feature Pipeline (Hourly)
- Triggered every hour via GitHub Actions.
- Fetches weather (Temp, Humidity, Wind) + current PM2.5.
- Computes derived features and upserts into MongoDB.

### 🧠 Training Pipeline (Daily)
- Triggered every morning or on-demand.
- Automates model selection: Trains Ridge, RF, and LSTM.
- Evaluates scores (RMSE, MAE, R²) and promotes the best model to "Production" in MongoDB.

### 🧪 CI Pipeline (Quality Control)
- Runs on every Pull Request or Push.
- Executes `ruff` for linting and `verify_decoupled.py` for headless prediction testing.

---

## 6. Challenges & Technical Resolutions

Building an automated ML system for a city like Karachi came with significant hurdles. Below is a detailed breakdown of the technical challenges faced and how they were overcome:

### 🛠️ ML Training & Prediction Challenges
- **Early-Stage Overfitting**:
    - *Problem*: Initial models showed 99% accuracy on training data but performed poorly on test data (RMSE was high).
    - *Resolution*: Implemented **Regularization** (Ridge Regression) and **Pruning** (limiting Max Depth in Random Forest). Also introduced **Temporal Cross-Validation** to ensure the model was learning trends, not memorizing specific data points.
- **The "Cold Start" Data Problem**: 
    - *Problem*: At the start, there was no historical data in MongoDB to train the model.
    - *Resolution*: Developed a **Backfill Script** (`backfill.py`) that fetched months of historical data from Open-Meteo's archive, creating a robust foundation for the initial training.
- **Complexity of Recursive Inference**: 
    - *Problem*: Predicting 72 hours ahead isn't possible with a single "predict" call. Early attempts led to "flat lines" or divergent values.
    - *Resolution*: Developed a **Sliding Window Inference logic**. The output of Hour 1 is used as a "Lag Feature" to predict Hour 2, and so on. This maintains the physical consistency of the forecast.

### 🌐 Deployment & Infrastructure Challenges
- **Data Spike Suppression**: 
    - *Problem*: Hardware sensors or API glitches occasionally reported PM2.5 jumps from 100 to 400 and back to 100 in one hour, creating visual "spikes".
    - *Resolution*: Integrated a **Rolling Median Filter (Window=3)**. This statistical technique discards single-point anomalies while preserving the legitimate rising/falling trends.
- **GitHub Branching & Push Rejections**: 
    - *Problem*: Confusion between the legacy `master` branch and modern `main` branch caused numerous push failures.
    - *Resolution*: Standardized the repository to the `main` branch and utilized forced updates (`--force`) after cleaning the git history to ensure a professional production repository.
- **Secrets & API Security**: 
    - *Problem*: Managing multiple keys (MongoDB, OpenWeather) across Local, GitHub Actions, and Streamlit Cloud.
    - *Resolution*: Centralized secret management using `config/settings.py`. This allows the code to seamlessly switch between local `.env` and Cloud Environment Variables without hardcoding.

### 🎨 UI/UX Challenges
- **The "N/A" Display Logic**:
    - *Problem*: Before we had automated pipelines, the dashboard would often show "Calibration Data Missing" errors.
    - *Resolution*: Implemented **Robust Fallbacks**. If live data fails, a cached version is used. If prediction fails, we display the last known trend with a "Data Refreshing" status, preventing a broken user experience.
- **Mobile Responsiveness**:
    - *Problem*: Complex glassmorphic cards looked cluttered on small screens.
    - *Resolution*: Refactored the layout using a **Column-to-Grid approach** in Streamlit, ensuring that the 3-Day Forecast grid scales elegantly on mobile devices.

---

## 7. Model Performance (Current Benchmarks)
The system currently selects the **Random Forest** model for production due to its superior variance capture.

| Metric | Score (D1 Forecast) | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | **0.950** | Excellent (95% of variance explained). |
| **RMSE** | **9.54** | Very Low error relative to the AQI range. |
| **MAE** | **6.01** | Highly accurate average prediction. |

---

## 8. Final Results
The project is now fully **Production-Ready**. 
- **User Interface**: Premium Glassmorphic design with real-time health precautions.
- **Automation**: 100% hands-free data fetching and model training.
- **Reliability**: Statistical smoothing and error handling ensure a clean user experience.

**Developer**: Ayesha (Intern)  
**System**: Karachi AQI Predictor v2.0
