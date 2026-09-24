"""
history.py
----------
Loads the REAL AEP transmission-zone dataset (Oct 2012 - Nov 2017) and the
two published RandomForestClassifier models (peak-day, peak-hour) from the
reference research project, and exposes a Flask Blueprint that powers the
calendar / "pick any date" analysis on the dashboard.

This is real historical data and real trained models — not a simulation.
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "real_data")
MODELS = os.path.join(HERE, "real_models")

# ---- load once at import time ----
daily = pd.read_csv(os.path.join(DATA, "daily_predictions.csv"))
daily["Date"] = pd.to_datetime(daily["Date"])

hourly = pd.read_csv(os.path.join(DATA, "hourly_data.csv"))
hourly["Datetime"] = pd.to_datetime(hourly["Datetime"])
hourly["Date"] = pd.to_datetime(hourly["Date"])

top_hours = pd.read_csv(os.path.join(DATA, "top_peak_hours.csv"))
top_hours["Datetime"] = pd.to_datetime(top_hours["Datetime"])
top_hours["Date"] = pd.to_datetime(top_hours["Date"])

peak_day_model = joblib.load(os.path.join(MODELS, "peak_day_model.pkl"))
peak_hour_model = joblib.load(os.path.join(MODELS, "peak_hour_model.pkl"))

DAY_FEATURES = ["Daily_Max_Load", "Max_Temperature", "Avg_Humidity", "Month", "IsWeekend"]
HOUR_FEATURES = ["Load", "Temperature", "Humidity", "Hour", "Month", "IsWeekend"]

MIN_DATE = daily["Date"].min().strftime("%Y-%m-%d")
MAX_DATE = daily["Date"].max().strftime("%Y-%m-%d")

# Published headline metrics (from the paper / README) — real, not placeholders
PUBLISHED_METRICS = {
    "peak_day_model": {
        "accuracy": 0.950, "precision": 0.348, "recall": 0.667,
        "f1": 0.457, "roc_auc": 0.907, "threshold": 0.15,
        "test_n": 378, "test_positive_n": 12,
    },
    "peak_hour_model": {
        "roc_auc": 0.9104, "accuracy": 0.9581,
        "mean_top_prob": 0.6924, "max_prob": 0.9624,
    },
    "dataset": {
        "source": "AEP transmission zone (PJM Interconnection)",
        "range": f"{MIN_DATE} to {MAX_DATE}",
        "daily_obs": int(len(daily)),
        "hourly_obs": int(len(hourly)),
    },
}

bp = Blueprint("history", __name__, url_prefix="/api/history")


def risk_label(prob):
    if prob > 0.60:
        return "High"
    if prob > 0.30:
        return "Moderate"
    return "Normal"


@bp.route("/date-range")
def date_range():
    return jsonify({"min": MIN_DATE, "max": MAX_DATE})


@bp.route("/calendar/<int:year>/<int:month>")
def calendar_month(year, month):
    """Peak-day probability for every date in a given month, for coloring the calendar grid."""
    mask = (daily["Date"].dt.year == year) & (daily["Date"].dt.month == month)
    rows = daily[mask].sort_values("Date")
    out = [
        {"date": d.strftime("%Y-%m-%d"), "prob": round(float(p), 4), "risk": risk_label(p)}
        for d, p in zip(rows["Date"], rows["Peak_Day_Prob"])
    ]
    return jsonify(out)


@bp.route("/day/<date>")
def day_analysis(date):
    try:
        target = pd.to_datetime(date)
    except Exception:
        return jsonify({"error": "invalid date"}), 400

    day_row = daily[daily["Date"] == target]
    if day_row.empty:
        return jsonify({"error": "no data for this date", "min": MIN_DATE, "max": MAX_DATE}), 404
    day_row = day_row.iloc[0]
    prob = float(day_row["Peak_Day_Prob"])

    day_hourly = hourly[hourly["Date"] == target].copy().sort_values("Datetime")
    if day_hourly.empty:
        return jsonify({"error": "no hourly data for this date"}), 404

    # Cost risk index, same formula as the reference dashboard
    avg_load = day_hourly["Load"].mean()
    day_hourly["Load_Ratio"] = day_hourly["Load"] / avg_load
    day_hourly["Cost_Index"] = 1 + 0.6 * (day_hourly["Load_Ratio"] - 1)

    # Live inference with the real peak-hour classifier
    day_hourly["Hour"] = day_hourly["Datetime"].dt.hour
    day_hourly["Month"] = day_hourly["Datetime"].dt.month
    day_hourly["IsWeekend"] = (day_hourly["Datetime"].dt.weekday >= 5).astype(int)
    day_hourly["Peak_Hour_Prob"] = peak_hour_model.predict_proba(day_hourly[HOUR_FEATURES])[:, 1]

    top_today = top_hours[top_hours["Date"] == target].sort_values("Peak_Hour_Prob", ascending=False)
    top_list = [
        {"time": r["Datetime"].strftime("%H:%M"), "prob": round(float(r["Peak_Hour_Prob"]), 4)}
        for _, r in top_today.head(2).iterrows()
    ]

    critical = day_hourly[day_hourly["Peak_Hour_Prob"] > 0.70]
    critical_hours = [t.strftime("%H:%M") for t in critical["Datetime"]]

    return jsonify({
        "date": target.strftime("%Y-%m-%d"),
        "peak_day_prob": round(prob, 6),
        "risk": risk_label(prob),
        "daily_max_load": float(day_row["Daily_Max_Load"]),
        "max_temperature_k": float(day_row["Max_Temperature"]),
        "avg_humidity_pct": float(day_row["Avg_Humidity"]),
        "is_weekend": bool(day_row["IsWeekend"]),
        "hourly": {
            "time": day_hourly["Datetime"].dt.strftime("%H:%M").tolist(),
            "load": day_hourly["Load"].round(1).tolist(),
            "cost_index": day_hourly["Cost_Index"].round(3).tolist(),
            "peak_hour_prob": day_hourly["Peak_Hour_Prob"].round(4).tolist(),
        },
        "cost": {
            "avg_load": round(float(avg_load), 1),
            "avg_cost_index": round(float(day_hourly["Cost_Index"].mean()), 3),
            "peak_cost_index": round(float(day_hourly["Cost_Index"].max()), 3),
        },
        "top_peak_hours": top_list,
        "critical_hours": critical_hours,
    })


@bp.route("/feature-importance")
def feature_importance():
    importances = peak_hour_model.feature_importances_
    data = sorted(
        [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(HOUR_FEATURES, importances)],
        key=lambda x: -x["importance"],
    )
    return jsonify(data)


@bp.route("/metrics")
def metrics():
    return jsonify(PUBLISHED_METRICS)
