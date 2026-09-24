import pandas as pd

# ============================================
# STEP 1: Load Energy Data
# ============================================

energy_path = "data/raw/energy/AEP_hourly.csv"

energy = pd.read_csv(energy_path)

# Rename columns
energy.columns = ["Datetime", "Load"]

# Convert to datetime
energy["Datetime"] = pd.to_datetime(energy["Datetime"])

# Sort
energy = energy.sort_values("Datetime")

print("\nEnergy data loaded:", energy.shape)


# ============================================
# STEP 2: Load Weather Data
# ============================================

temp_path = "data/raw/weather/temperature.csv"
humidity_path = "data/raw/weather/humidity.csv"

temp = pd.read_csv(temp_path)
humidity = pd.read_csv(humidity_path)

# Convert datetime
temp["datetime"] = pd.to_datetime(temp["datetime"])
humidity["datetime"] = pd.to_datetime(humidity["datetime"])

# Select one city (New York works well with PJM regions)
temp = temp[["datetime", "New York"]]
humidity = humidity[["datetime", "New York"]]

# Rename columns
temp.columns = ["Datetime", "Temperature"]
humidity.columns = ["Datetime", "Humidity"]

# Merge weather
weather = temp.merge(humidity, on="Datetime")

print("Weather data loaded:", weather.shape)


# ============================================
# STEP 3: Merge Energy + Weather
# ============================================

data = energy.merge(weather, on="Datetime", how="inner")

print("\nMerged data shape:", data.shape)


# ============================================
# STEP 4: Handle Missing Values
# ============================================

print("\nMissing values BEFORE cleaning:")
print(data.isna().sum())

# Fill missing weather values
data["Temperature"] = data["Temperature"].ffill().bfill()
data["Humidity"] = data["Humidity"].ffill().bfill()

# Drop any remaining NaNs (safety)
data = data.dropna()

print("\nMissing values AFTER cleaning:")
print(data.isna().sum())


# ============================================
# STEP 5: Basic Sanity Checks
# ============================================

print("\nData preview:")
print(data.head())

print("\nDate range:")
print(data["Datetime"].min(), "to", data["Datetime"].max())


# ============================================
# STEP 6: Save Processed File
# ============================================

output_path = "data/processed/hourly_data.csv"

data.to_csv(output_path, index=False)

print("\nSaved merged data to:", output_path)
print("Final dataset shape:", data.shape)
