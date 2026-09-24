# PeakCast — Electricity Peak Demand Prediction

A full-stack dashboard with **two prediction modes**, both backed by real,
trained machine-learning models:

1. **Calendar mode (real data)** — pick any date from Oct 2012 to Nov 2017
   and see the actual AEP transmission-zone load for that day, plus a live
   peak-day / peak-hour risk score from two published RandomForestClassifier
   models trained in the referenced research project.
2. **What-if mode (synthetic)** — sliders for season, day type, and
   temperature, run through a RandomForestRegressor, for exploring
   conditions outside the historical record.

> Based on: *Peak Electricity Demand Prediction Using Dual-Granularity
> Random Forest Classification: A Case Study on the AEP Transmission Zone* —
> Mayank, Aishwarya Shelke, Dr. Seema Shukla, Sharda University.

```
PeakCast/
├── backend/
│   ├── app.py              Flask app — serves both APIs + the frontend
│   ├── history.py           real-data calendar API (loads real_data/ + real_models/)
│   ├── train_model.py       trains the synthetic what-if regressor
│   ├── requirements.txt
│   ├── real_data/            the actual AEP dataset (daily + hourly + top peak hours)
│   ├── real_models/          the two published RandomForestClassifier models (.pkl)
│   └── model_store/          what-if regressor artifacts (created by train_model.py)
├── frontend/
│   ├── index.html            calendar + what-if simulator + results + architecture
│   ├── style.css
│   └── script.js
├── research/                 original training pipeline + analysis notebook, for reference
│   ├── src/step1..step5*.py
│   └── model_analysis.ipynb
└── README.md
```

## Run it

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. First run trains the what-if regressor
(~10–20s); after that, startup is instant.

## The calendar — how it works

`GET /api/history/calendar/<year>/<month>` returns each day's peak-day
probability so the calendar grid can color High/Moderate/Normal risk days.
Clicking a date calls `GET /api/history/day/<date>`, which:

- looks up that day's real `Daily_Max_Load`, `Max_Temperature`, `Avg_Humidity`
  from `real_data/daily_predictions.csv` and its pre-computed `Peak_Day_Prob`
- pulls that day's 24 real hourly load readings from `real_data/hourly_data.csv`
- computes the demand-based cost risk index (`C_h = 1 + 0.6 × (load_h / avg − 1)`)
- runs **live inference** with `real_models/peak_hour_model.pkl` on that
  day's hourly features to produce a full peak-hour probability curve,
  and flags any hour above 70% as critical

Try **July 15–19, 2013** — a real five-day heatwave cluster where the
model's peak-day probability climbs to ~69% and load tops 22,800 MW around
3–4pm each afternoon.

## Published results (real, not placeholders)

| Peak-day model | Peak-hour model |
|---|---|
| Accuracy 0.950 | Accuracy 0.958 |
| ROC-AUC 0.907 | ROC-AUC 0.910 |
| F1 (peak class) 0.457 | Max probability 0.962 |

Served live at `GET /api/history/metrics` and `GET /api/history/feature-importance`.

## What to say to the panel

- **Problem**: grids are provisioned for their single worst hour of the
  year; advance warning on that peak lets an operator pre-cool buildings,
  trigger demand-response contracts, or dispatch reserves before prices spike.
- **Data**: the AEP transmission zone's real hourly load (PJM Interconnection),
  merged with multi-city temperature and humidity, Oct 2012 – Nov 2017 —
  1,887 daily observations, 45,248 hourly observations.
- **Approach**: dual granularity — a daily classifier flags whether a day
  is likely to set a new peak (top-5th-percentile load), a separate hourly
  classifier localizes which hour within that day the peak lands on.
- **Why classification, not regression**: peak days are rare (3.3% of days),
  so the problem is framed as an imbalanced classification task with
  class-balanced Random Forests and a tuned decision threshold (0.15,
  chosen by F1 maximization) rather than predicting raw load directly.
- **Honest limitations** (from the paper): only 12 true peak days in the
  test set, so F1 is sensitive to individual errors; the model doesn't
  explicitly capture multi-day autocorrelation; retraining is needed for
  other regions.
- **Novelty / future work** (from the paper): lagged load features (prior-day
  max, 7-day rolling mean) to catch multi-day heat waves; SMOTE oversampling
  for the class imbalance; a Temporal Fusion Transformer for explicit
  sequence modeling; a live grid-telemetry connection for real-time alerts.

## Reproducing the models

`research/src/step1..step5` are the original pipeline scripts (merge data →
feature engineering → train peak-day model → train peak-hour model →
predict) and `research/model_analysis.ipynb` has the full EDA and evaluation.
The `.pkl` files already shipped in `backend/real_models/` are their output —
you don't need to re-run them to use the dashboard.

## Notes

- Flask's dev server is fine for a demo/viva. For real deployment, run behind
  `gunicorn`.
- `Max_Temperature` is stored in Kelvin (as in the original dataset); the
  dashboard converts it to °F for display.
