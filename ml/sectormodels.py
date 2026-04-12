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
# DEFINE SECTORS
# ==============================

sectors = {
    "industrial": [c for c in df.columns if "industrial" in c and "grid_import" in c],
    "residential": [c for c in df.columns if "residential" in c and "grid_import" in c],
    "public": [c for c in df.columns if "public" in c and "grid_import" in c],
}

# ==============================
# FUNCTION TO RUN MODEL
# ==============================

def run_model(df, grid_cols, sector_name):
    print(f"\n========== {sector_name.upper()} ==========")

    # Create target
    df_sector = df.copy()
    df_sector["total_energy"] = df_sector[grid_cols].sum(axis=1)

    # Fill missing values
    df_sector = df_sector.fillna(0)

    # Time features
    df_sector["hour"] = df_sector.index.hour
    df_sector["day_of_week"] = df_sector.index.dayofweek
    df_sector["month"] = df_sector.index.month

    # Lag features
    df_sector["lag_1"] = df_sector["total_energy"].shift(1)
    df_sector["lag_4"] = df_sector["total_energy"].shift(4)
    df_sector["lag_96"] = df_sector["total_energy"].shift(96)

    df_sector = df_sector.dropna()

    # Features
    features = ["lag_1", "lag_4", "lag_96", "hour", "day_of_week", "month"]
    X = df_sector[features]
    y = df_sector["total_energy"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=features)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))


# ==============================
# RUN FOR EACH SECTOR
# ==============================

results = {}

for sector_name, cols in sectors.items():
    if len(cols) == 0:
        print(f"No columns for {sector_name}")
        continue

    run_model(df, cols, sector_name)

#notes from the data: residential dxata is the most consistent and realistic based on the error
#metrics so we will work with residential data for the rest of the project. 
#The industrial and public data is very sparse and has a lot of zeroes which is likely why the model performs poorly on it.