import pandas as pd

# ============================================
# STEP 1: Load Processed Hourly Data
# ============================================

data_path = "data/processed/hourly_data.csv"
data = pd.read_csv(data_path)

data["Datetime"] = pd.to_datetime(data["Datetime"])

print("Data loaded:", data.shape)


# ============================================
# STEP 2: Time Features
# ============================================

data["Hour"] = data["Datetime"].dt.hour
data["Day"] = data["Datetime"].dt.day
data["Month"] = data["Datetime"].dt.month
data["Year"] = data["Datetime"].dt.year
data["Weekday"] = data["Datetime"].dt.weekday

# Weekend flag
data["IsWeekend"] = data["Weekday"].isin([5, 6]).astype(int)

print("Time features added")


# ============================================
# STEP 3: Create Peak Hour Label
# Peak hour = max load within each day
# ============================================

data["Date"] = data["Datetime"].dt.date

daily_peak_load = data.groupby("Date")["Load"].transform("max")

data["Peak_Hour"] = (data["Load"] == daily_peak_load).astype(int)

print("Peak hour label created")


# ============================================
# STEP 4: Create Daily Dataset for Peak Day Model
# ============================================

daily_data = data.groupby("Date").agg({
    "Load": "max",
    "Temperature": "max",
    "Humidity": "mean",
    "Month": "first",
    "Year": "first",
    "IsWeekend": "max"
}).reset_index()

daily_data.columns = [
    "Date",
    "Daily_Max_Load",
    "Max_Temperature",
    "Avg_Humidity",
    "Month",
    "Year",
    "IsWeekend"
]

print("Daily dataset shape:", daily_data.shape)


# ============================================
# STEP 5: Peak Day Label
# Peak day = max daily load within each month
# ============================================

daily_data["YearMonth"] = pd.to_datetime(daily_data["Date"]).dt.to_period("M")

monthly_peak_load = daily_data.groupby("YearMonth")["Daily_Max_Load"].transform("max")

daily_data["Peak_Day"] = (daily_data["Daily_Max_Load"] == monthly_peak_load).astype(int)

print("Peak day label created")


# ============================================
# STEP 6: Save Files
# ============================================

hourly_output = "data/processed/hourly_data.csv"
daily_output = "data/processed/daily_data.csv"

data.to_csv(hourly_output, index=False)
daily_data.to_csv(daily_output, index=False)

print("\nSaved hourly data:", hourly_output)
print("Saved daily data:", daily_output)


# ============================================
# STEP 7: Sanity Check
# ============================================

print("\nPeak hour distribution:")
print(data["Peak_Hour"].value_counts())

print("\nPeak day distribution:")
print(daily_data["Peak_Day"].value_counts())
