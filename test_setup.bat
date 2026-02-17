@echo off
echo ============================================================
echo Testing MongoDB Connection
echo ============================================================
echo.

set MONGODB_URI=mongodb+srv://aqiuser:JmSm8SxX4EFzyC56@cluster0.ujo97dt.mongodb.net/?appName=Cluster0
set MONGODB_DB=aqi_predictor
set AQI_CITY=Karachi

python test_connection.py
