import pandas as pd
import joblib

# ============================================
# Load Models
# ============================================

peak_day_model = joblib.load("models/peak_day_model.pkl")
peak_hour_model = joblib.load("models/peak_hour_model.pkl")

print("Models loaded")


# ============================================
# Load Data
# ============================================

hourly_data = pd.read_csv("data/processed/hourly_data.csv")
daily_data = pd.read_csv("data/processed/daily_data.csv")

hourly_data["Datetime"] = pd.to_datetime(hourly_data["Datetime"])

print("Data loaded")


# ============================================
# Peak Day Probability
# ============================================

day_features = [
    "Daily_Max_Load",
    "Max_Temperature",
    "Avg_Humidity",
    "Month",
    "IsWeekend"
]

daily_data["Peak_Day_Prob"] = peak_day_model.predict_proba(
    daily_data[day_features]
)[:, 1]


# ============================================
# Peak Hour Probability
# ============================================

hour_features = [
    "Load",
    "Temperature",
    "Humidity",
    "Hour",
    "Month",
    "IsWeekend"
]

hourly_data["Peak_Hour_Prob"] = peak_hour_model.predict_proba(
    hourly_data[hour_features]
)[:, 1]


# ============================================
# Select TOP 2 Hours Per Day (Correct Way)
# ============================================

hourly_data["Date"] = hourly_data["Datetime"].dt.date

top_hours = (
    hourly_data
    .sort_values(["Date", "Peak_Hour_Prob"], ascending=[True, False])
    .groupby("Date")
    .head(2)
    .reset_index(drop=True)
)

print("\nSample Top Hours:")
print(top_hours.head())


# ============================================
# Save Outputs
# ============================================

daily_data.to_csv("data/processed/daily_predictions.csv", index=False)
top_hours.to_csv("data/processed/top_peak_hours.csv", index=False)

print("\nSaved:")
print("data/processed/daily_predictions.csv")
print("data/processed/top_peak_hours.csv")
