"""
Exploratory Data Analysis: load features from Feature Store, plot trends and correlations.
Saves figures to metrics/ or a chosen output dir for the report.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFAULT_CITY, METRICS_DIR
from scripts.db import load_features


def run_eda(city: str = DEFAULT_CITY, output_dir: Path = None):
    """Load features, generate EDA plots and summary stats."""
    output_dir = output_dir or METRICS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(city=city, limit=0)
    if df.empty or len(df) < 10:
        print("Not enough data for EDA. Run backfill first.")
        return

    # Parse timestamp
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts")
    else:
        print("No timestamp column.")
        return

    # 1) AQI over time
    if "us_aqi" in df.columns:
        aqi = df[["ts", "us_aqi"]].copy()
        aqi["us_aqi"] = pd.to_numeric(aqi["us_aqi"], errors="coerce")
        aqi = aqi.dropna()
        if len(aqi) > 0:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(aqi["ts"], aqi["us_aqi"], alpha=0.7)
                ax.set_title("AQI over time")
                ax.set_xlabel("Time")
                ax.set_ylabel("AQI (US)")
                fig.tight_layout()
                fig.savefig(output_dir / "eda_aqi_over_time.png", dpi=100)
                plt.close()
                print(f"Saved {output_dir / 'eda_aqi_over_time.png'}")
            except Exception as e:
                print(f"Plot failed: {e}")

    # 2) AQI distribution
    if "us_aqi" in df.columns:
        aqi_vals = pd.to_numeric(df["us_aqi"], errors="coerce").dropna()
        if len(aqi_vals) > 0:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(aqi_vals, bins=30, edgecolor="black", alpha=0.7)
                ax.set_title("AQI distribution")
                ax.set_xlabel("AQI")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(output_dir / "eda_aqi_distribution.png", dpi=100)
                plt.close()
                print(f"Saved {output_dir / 'eda_aqi_distribution.png'}")
            except Exception as e:
                print(f"Plot failed: {e}")

    # 3) Summary stats
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.describe().round(2)
    summary_path = output_dir / "eda_summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # 4) Correlation with AQI (if present)
    if "us_aqi" in df.columns:
        numeric = df.select_dtypes(include=[np.number])
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr()
        if "us_aqi" in corr.columns:
            aqi_corr = corr["us_aqi"].drop("us_aqi", errors="ignore").sort_values(key=lambda x: x.abs(), ascending=False)
            corr_path = output_dir / "eda_correlation_with_aqi.csv"
            aqi_corr.to_csv(corr_path)
            print(f"Saved {corr_path}")

    print("EDA complete.")


def main():
    run_eda()


if __name__ == "__main__":
    main()
