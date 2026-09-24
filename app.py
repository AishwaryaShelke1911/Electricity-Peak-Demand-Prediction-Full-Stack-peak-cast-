"""
app.py
------
Flask backend for PeakCast. Two prediction modes, both real ML:

  /api/whatif/*   A RandomForestRegressor trained on a synthetic-but-realistic
                   load curve (train_model.py). Use this to explore hypothetical
                   "what if it's this hot on a weekday in summer" scenarios.

  /api/history/*  The REAL AEP transmission-zone dataset (Oct 2012 - Nov 2017)
                   and the two published RandomForestClassifier models
                   (peak-day, peak-hour) from the reference research project.
                   Pick any date in range on the calendar to see what actually
                   happened, plus live peak-hour inference for that day.

Also serves the frontend (static HTML/CSS/JS) from ../frontend so the whole
app runs from a single `python app.py` command.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "model_store")
FRONTEND = os.path.join(HERE, "..", "frontend")

MODEL_PATH = os.path.join(STORE, "peak_model.pkl")
METRICS_PATH = os.path.join(STORE, "metrics.json")
MONTHLY_PATH = os.path.join(STORE, "monthly_peaks.json")

# Train the what-if model on first run if artifacts don't exist yet
if not os.path.exists(MODEL_PATH):
    from train_model import train
    train()

whatif_model = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    WHATIF_METRICS = json.load(f)
with open(MONTHLY_PATH) as f:
    WHATIF_MONTHLY_PEAKS = json.load(f)

app = Flask(__name__, static_folder=FRONTEND, static_url_path="")

# Register the real-data calendar/history blueprint (see history.py)
from history import bp as history_bp
app.register_blueprint(history_bp)

SEASON_MONTH = {"winter": 1, "summer": 7, "shoulder": 4}


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/whatif/metrics")
def whatif_metrics():
    return jsonify(WHATIF_METRICS)


@app.route("/api/whatif/monthly-peaks")
def whatif_monthly_peaks():
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data = [{"month": names[int(m) - 1], "peak_mw": v} for m, v in sorted(WHATIF_MONTHLY_PEAKS.items(), key=lambda kv: int(kv[0]))]
    return jsonify(data)


@app.route("/api/whatif/predict", methods=["POST"])
def whatif_predict():
    body = request.get_json(force=True) or {}
    season = body.get("season", "summer")
    day_type = body.get("day_type", "weekday")
    temperature = float(body.get("temperature", 85))

    month = SEASON_MONTH.get(season, 7)
    dow = 2 if day_type == "weekday" else 6
    is_weekend = 1 if day_type == "weekend" else 0

    hours = list(range(24))
    X = pd.DataFrame(
        [[h, dow, month, is_weekend, temperature] for h in hours],
        columns=["hour", "dow", "month", "is_weekend", "temperature"],
    )
    preds = whatif_model.predict(X)

    peak_idx = int(np.argmax(preds))
    result = {
        "hours": hours,
        "load_mw": [round(float(v), 1) for v in preds],
        "peak_hour": hours[peak_idx],
        "peak_load_mw": round(float(preds[peak_idx]), 1),
        "inputs": {"season": season, "day_type": day_type, "temperature": temperature},
    }
    return jsonify(result)


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
