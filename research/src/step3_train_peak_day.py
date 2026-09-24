import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = pd.read_csv("data/processed/daily_data.csv")
print("Data loaded:", data.shape)

features = [
    "Daily_Max_Load",
    "Max_Temperature",
    "Avg_Humidity",
    "Month",
    "IsWeekend"
]

X = data[features]
y = data["Peak_Day"]

print("Peak day distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=3,
    random_state=42,
    class_weight={0: 1, 1: 15}
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/peak_day_model.pkl")
print("Peak day model saved")
