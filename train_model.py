"""
train_model.py
----------------
Generates a realistic synthetic hourly electricity-load dataset (shaped like
the AEP hourly-consumption pattern: seasonal swings, a daily double-hump
curve, weekday/weekend effects, and temperature sensitivity), trains a
RandomForestRegressor to predict hourly load, and saves:
  - model_store/peak_model.pkl   (trained sklearn model)
  - model_store/metrics.json     (held-out evaluation metrics)
  - model_store/monthly_peaks.json (avg monthly peak load, for the results chart)

Swap generate_dataset() for a loader that reads your real AEP_hourly.csv
(or any hourly load CSV with a datetime + load column) and the rest of the
pipeline (features, training, saving) stays the same.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "model_store")
os.makedirs(STORE, exist_ok=True)


def generate_dataset(start="2019-01-01", periods_years=4, seed=42):
    """Synthetic hourly load data with realistic seasonal + daily + weekly structure."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods_years * 365 * 24, freq="h")
    df = pd.DataFrame({"timestamp": idx})
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["doy"] = df.timestamp.dt.dayofyear
    df["is_weekend"] = (df.dow >= 5).astype(int)

    # Synthetic daily temperature (°F): coldest in Jan, hottest in Jul, plus daily noise
    seasonal_temp = 55 - 25 * np.cos(2 * np.pi * (df.doy - 15) / 365)
    df["temperature"] = seasonal_temp + rng.normal(0, 6, len(df))

    # Base load + winter/summer peaks (electric heating in Jan, AC in Jul)
    base = 12000
    winter_bump = 3200 * np.exp(-((df.month - 1) ** 2) / 6)
    summer_bump = 4200 * np.exp(-((df.month - 7) ** 2) / 6)

    # Daily shape: morning ramp + evening peak (shifts later in summer)
    evening_peak_hour = np.where(df.month.isin([6, 7, 8, 9]), 19, 18)
    daily_shape = (
        2600 * np.exp(-((df.hour - evening_peak_hour) ** 2) / 10)
        + 1400 * np.exp(-((df.hour - 8) ** 2) / 6)
    )

    weekend_cut = np.where(df.is_weekend == 1, 0.90, 1.0)
    temp_sensitivity = np.where(
        df.temperature > 75, (df.temperature - 75) * 110,
        np.where(df.temperature < 45, (45 - df.temperature) * 70, 0),
    )
    noise = rng.normal(0, 350, len(df))

    df["load_mw"] = (
        (base + winter_bump + summer_bump + daily_shape) * weekend_cut
        + temp_sensitivity
        + noise
    ).round(0)

    return df


def build_features(df):
    X = df[["hour", "dow", "month", "is_weekend", "temperature"]].copy()
    y = df["load_mw"]
    return X, y


def train():
    df = generate_dataset()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=120, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)

    metrics = {
        "mae_mw": round(mae, 1),
        "rmse_mw": round(rmse, 1),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Monthly peak load (max load reached in each calendar month, averaged across years)
    tmp = df.copy()
    tmp["_year"] = tmp.timestamp.dt.year
    tmp["_month"] = tmp.timestamp.dt.month
    monthly = tmp.groupby(["_year", "_month"])["load_mw"].max().reset_index()
    monthly_avg = monthly.groupby("_month")["load_mw"].mean().round(0)
    monthly_peaks = {int(m): float(v) for m, v in monthly_avg.items()}

    joblib.dump(model, os.path.join(STORE, "peak_model.pkl"), compress=3)
    with open(os.path.join(STORE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(STORE, "monthly_peaks.json"), "w") as f:
        json.dump(monthly_peaks, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    return model, metrics


if __name__ == "__main__":
    train()
