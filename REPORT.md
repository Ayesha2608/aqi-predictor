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

## 6. Challenges & Resolutions

| Challenge | Impact | Resolution |
| :--- | :--- | :--- |
| **The "Flat Line" Issue** | Forecast showed a single repeating value. | Fixed the inference loop where the feature vector wasn't being updated with the previous prediction's result. |
| **Graph Spikes** | Random 300+ AQI spikes in a single hour. | Implemented a **Rolling Median Filter** to smooth out data noise without losing the overall trend. |
| **Data Integrity (N/A)** | "N/A" values when API was down or laggy. | Developed a robust cleaning function that uses the last known valid value (interpolation) instead of failing. |
| **Layout Spacing** | Dashboard looked cluttered on mobile. | Refactored Streamlit `st.columns` and injected custom CSS for responsive container padding. |

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
