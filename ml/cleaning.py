import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1. LOAD DATA
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "household_data_15min_singleindex.csv")

print("Loading data from:", file_path)

df = pd.read_csv(file_path)

# ==============================
# 2. BASIC PREPROCESSING
# ==============================

# Convert timestamp
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df.set_index('utc_timestamp', inplace=True)

print("Dataset shape:", df.shape)

# ==============================
# 3. FEATURE SELECTION
# ==============================

grid_cols = [c for c in df.columns if "grid_import" in c]
pv_cols = [c for c in df.columns if "_pv" in c]

print("Grid columns:", len(grid_cols))
print("PV columns:", len(pv_cols))

# ==============================
# 4. TARGET VARIABLE
# ==============================

df["total_energy"] = df[grid_cols].sum(axis=1)

# Fill missing values
df = df.fillna(0)

# ==============================
# 5. TIME FEATURES
# ==============================

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# ==============================
# 6. LAG FEATURES
# ==============================

df["lag_1"] = df["total_energy"].shift(1)
df["lag_4"] = df["total_energy"].shift(4)     # 1 hour ago
df["lag_96"] = df["total_energy"].shift(96)   # 1 day ago

# Drop NaNs caused by lagging
df = df.dropna()

print("After feature engineering:", df.shape)

# ==============================
# 7. DEFINE FEATURES
# ==============================

features = ["lag_1", "lag_4", "lag_96", "hour", "day_of_week", "month"]

X = df[features]
y = df["total_energy"]

# ==============================
# 8. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# 9. TRAIN MODEL
# ==============================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 10. PREDICTIONS
# ==============================

preds = model.predict(X_test)

# ==============================
# 11. EVALUATION
# ==============================

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

# ==============================
# 12. PLOT RESULTS
# ==============================

plt.figure(figsize=(12,5))
plt.plot(y_test.values[:500], label="Actual")
plt.plot(preds[:500], label="Predicted")

plt.title("Energy Forecasting (Combined Dataset)")
plt.xlabel("Time steps")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 13. FEATURE IMPORTANCE
# ==============================

importance = pd.Series(model.feature_importances_, index=features)

plt.figure(figsize=(8,4))
importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.tight_layout()
plt.show()

print("\nFeature Importance:")
print(importance.sort_values(ascending=False))