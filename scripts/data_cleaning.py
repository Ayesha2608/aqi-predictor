"""
Data cleaning and timezone conversion for AQI pipeline.
- Convert all timestamps to Asia/Karachi (or config TIMEZONE).
- Drop duplicates, fill missing values, optional outlier handling.
- No CSV; output is DataFrame for Feature Store only.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import TIMEZONE


def to_local_tz(dt_series_or_value):
    """Convert datetime(s) to TIMEZONE (e.g. Asia/Karachi)."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(TIMEZONE)
    except Exception:
        return dt_series_or_value
    if hasattr(dt_series_or_value, "dt"):
        return dt_series_or_value.dt.tz_convert(tz) if dt_series_or_value.dt.tz is not None else dt_series_or_value
    if isinstance(dt_series_or_value, (pd.Timestamp, pd.DatetimeTZDtype)):
        if getattr(dt_series_or_value, "tz", None) is not None:
            return dt_series_or_value.tz_convert(tz)
        return dt_series_or_value
    return dt_series_or_value


def clean_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature DataFrame before storing in Feature Store.
    - Normalize timestamp column to Asia/Karachi
    - Drop duplicates by (timestamp, city)
    - Fill missing numeric with median
    - Optional: clip extreme outliers (e.g. AQI > 500 or < 0)
    """
    if df.empty:
        return df
    df = df.copy()
    # Normalize timestamp to local timezone
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.dt.tz is not None:
            try:
                from zoneinfo import ZoneInfo
                df["timestamp"] = ts.dt.tz_convert(ZoneInfo(TIMEZONE)).astype(str)
            except Exception:
                df["timestamp"] = ts.astype(str)
        else:
            df["timestamp"] = ts.astype(str)
    if "datetime_iso" in df.columns:
        ts = pd.to_datetime(df["datetime_iso"], errors="coerce")
        if ts.dt.tz is not None:
            try:
                from zoneinfo import ZoneInfo
                df["timestamp"] = ts.dt.tz_convert(ZoneInfo(TIMEZONE)).astype(str)
            except Exception:
                df["timestamp"] = ts.astype(str)
        else:
            df["timestamp"] = ts.astype(str)
    # Drop duplicates (keep first)
    key_cols = [c for c in ["timestamp", "city", "date"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    # Fill missing numeric with median
    numeric = df.select_dtypes(include=[np.number]).columns
    for c in numeric:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    # Clip AQI-like columns: 1-5 (OpenWeather) or 0-500 (US)
    for col in ["us_aqi", "aqi"]:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        if (valid <= 5).all() and (valid >= 1).all():
            df[col] = df[col].clip(lower=1, upper=5)
        else:
            df[col] = df[col].clip(lower=0, upper=500)
    return df.reset_index(drop=True)
