import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# LOAD DATA
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "household_data_15min_singleindex.csv")

print("Loading data...")
df = pd.read_csv(file_path)

# ==============================
# PREPROCESSING
# ==============================

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df.set_index('utc_timestamp', inplace=True)

# ==============================
# FIND RESIDENTIAL HOUSEHOLDS
# ==============================

residential_cols = [c for c in df.columns if "residential" in c and "grid_import" in c]

print("\nResidential households found:")
for col in residential_cols:
    print("-", col)

# ==============================
# FUNCTION TO RUN MODEL
# ==============================

def run_model(df, target_col):
    print(f"\n========== {target_col} ==========")

    # Select single household
    df_res = df[[target_col]].copy()

    # Drop missing values
    df_res = df_res.dropna()

    # Rename target
    df_res["total_energy"] = df_res[target_col]

    # ==============================
    # FEATURE ENGINEERING
    # ==============================

    df_res["hour"] = df_res.index.hour
    df_res["day_of_week"] = df_res.index.dayofweek
    df_res["month"] = df_res.index.month

    # Lag features
    df_res["lag_1"] = df_res["total_energy"].shift(1)
    df_res["lag_4"] = df_res["total_energy"].shift(4)
    df_res["lag_96"] = df_res["total_energy"].shift(96)

    df_res = df_res.dropna()

    # ==============================
    # MODEL DATA
    # ==============================

    features = ["lag_1", "lag_4", "lag_96", "hour", "day_of_week", "month"]

    X = df_res[features]
    y = df_res["total_energy"]

    # Train/test split (NO shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ==============================
    # MODEL
    # ==============================

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # ==============================
    # EVALUATION
    # ==============================

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================

    importance = pd.Series(model.feature_importances_, index=features)

    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))


# ==============================
# RUN FOR EACH HOUSEHOLD
# ==============================

for col in residential_cols:
    run_model(df, col)