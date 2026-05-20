import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# LOAD CLEANED DATA
# ==============================

df = pd.read_csv("data/residential3_cleaned.csv")

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values("utc_timestamp")

df.set_index("utc_timestamp", inplace=True)
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["season"] = df.index.month.map(get_season)


df = pd.get_dummies(df, columns=["season"], drop_first=True)

# ==============================
# FEATURE ENGINEERING
# ==============================

# Time feature
df["hour"] = df.index.hour

# Lag features (important baseline signal)
df["lag_1"] = df["DE_KN_residential3_grid_import"].shift(1)
df["lag_4"] = df["DE_KN_residential3_grid_import"].shift(4)
df["lag_96"] = df["DE_KN_residential3_grid_import"].shift(96)

# Drop NaNs created by lagging
df = df.dropna()

# ==============================
# FEATURES & TARGET
# ==============================

season_cols = [col for col in df.columns if col.startswith("season_")]

features = [
    "lag_1",
    "lag_4",
    "lag_96",
    "hour",
    "DE_KN_residential3_pv",
    "DE_KN_residential3_washing_machine",
    "DE_KN_residential3_dishwasher",
    "DE_KN_residential3_grid_export",
]+season_cols

target = "DE_KN_residential3_grid_import"

X = df[features]
y = df[target]

# ==============================
# TRAIN / TEST SPLIT (time-based)
# ==============================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# MODEL
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# ==============================
# COEFFICIENTS (interpretation)
# ==============================

coeffs = pd.Series(model.coef_, index=features)
print("\nFeature Coefficients:")
print(coeffs.sort_values(ascending=False))