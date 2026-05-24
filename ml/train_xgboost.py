import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
df = pd.read_csv(r"C:\Users\test\OneDrive\Documents\GitHub\DataScienceCapstone\data\residential3_cleaned.csv")
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.set_index('utc_timestamp').sort_index()

# Time features
df['hour'] = df.index.hour
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df = df.drop(columns=['hour', 'month', 'day_of_week'])

# ==========================================
# 2. MULTI-TARGET FEATURE ENGINEERING
# ==========================================
# Target 1: Grid Import Lags
df['import_lag_1'] = df['DE_KN_residential3_grid_import'].shift(1)
df['import_lag_4'] = df['DE_KN_residential3_grid_import'].shift(4)
df['import_lag_96'] = df['DE_KN_residential3_grid_import'].shift(96)
df['cons_roll_mean_4'] = df['DE_KN_residential3_grid_import'].rolling(window=4).mean()

# Target 2: Solar (PV) Lags
df['pv_lag_1'] = df['DE_KN_residential3_pv'].shift(1)
df['pv_lag_4'] = df['DE_KN_residential3_pv'].shift(4)
df['pv_roll_mean_4'] = df['DE_KN_residential3_pv'].rolling(window=4).mean()

# Drop NaN rows caused by shifting
df_xgb = df.dropna().copy()

# ==========================================
# 3. DEFINE X AND y (MULTI-OUTPUT)
# ==========================================
target_cols = ['DE_KN_residential3_pv', 'DE_KN_residential3_grid_import']
features = [col for col in df_xgb.columns if col not in target_cols]

X = df_xgb[features]
y = df_xgb[target_cols] # y is now a 2D dataframe!

# Chronological split
split_index = int(len(df_xgb) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# ==========================================
# 4. TRAIN MULTI-OUTPUT XGBOOST
# ==========================================
# Define the base model
base_xgb = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Wrap it to handle multiple targets
multi_model = MultiOutputRegressor(base_xgb)

print("Training Multi-Output XGBoost...")
multi_model.fit(X_train, y_train)
print("Training completed.")

# ==========================================
# 5. EVALUATE BOTH TARGETS
# ==========================================
y_pred = multi_model.predict(X_test)

# Slice the predictions (Column 0 is PV, Column 1 is Grid Import)
pv_actual = y_test['DE_KN_residential3_pv'].values
pv_pred = y_pred[:, 0]

# Zero-floor hack for PV (Sun doesn't generate negative power)
pv_pred[pv_pred < 0] = 0

grid_actual = y_test['DE_KN_residential3_grid_import'].values
grid_pred = y_pred[:, 1]

print("\n--- Solar (PV) Forecast ---")
print(f"MAE:  {mean_absolute_error(pv_actual, pv_pred):.3f} kW")
print(f"R²:   {r2_score(pv_actual, pv_pred):.3f}")

print("\n--- Grid Import Forecast ---")
print(f"MAE:  {mean_absolute_error(grid_actual, grid_pred):.3f} kW")
print(f"R²:   {r2_score(grid_actual, grid_pred):.3f}")