import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# STEP 1: Load Hourly Data
# ============================================

data_path = "data/processed/hourly_data.csv"
data = pd.read_csv(data_path)

print("Data loaded:", data.shape)


# ============================================
# STEP 2: Feature Selection
# ============================================

features = [
    "Load",
    "Temperature",
    "Humidity",
    "Hour",
    "Month",
    "IsWeekend"
]

X = data[features]
y = data["Peak_Hour"]

print("Features selected")


# ============================================
# STEP 3: Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# ============================================
# STEP 4: Train Random Forest
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model trained")


# ============================================
# STEP 5: Evaluation
# ============================================

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# STEP 6: Save Model
# ============================================

model_path = "models/peak_hour_model.pkl"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
