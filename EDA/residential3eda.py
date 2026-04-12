import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================
# LOAD DATA
# ==============================

csv_name = "household_data_15min_singleindex.csv"

csv_path = None
for root in [Path.cwd(), *Path.cwd().parents]:
    candidate = root / "data" / csv_name
    if candidate.exists():
        csv_path = candidate
        break

if csv_path is None:
    raise FileNotFoundError("Dataset not found")

df = pd.read_csv(csv_path)

# ==============================
# PREPROCESSING
# ==============================

df["cet_cest_timestamp"] = pd.to_datetime(df["cet_cest_timestamp"], utc=True)
df = df.sort_values("cet_cest_timestamp")

# Select residential3
cols = [c for c in df.columns if c.startswith("DE_KN_residential3")]
res = df[["cet_cest_timestamp"] + cols].copy()

res.set_index("cet_cest_timestamp", inplace=True)
res.index = res.index.tz_convert("Europe/Berlin").tz_localize(None)

# Drop missing values
res = res.dropna()

# ==============================
# DEFINE KEY VARIABLES
# ==============================

grid = "DE_KN_residential3_grid_import"
pv = "DE_KN_residential3_pv"

# ==============================
# 1. TIME SERIES (Consumption)
# ==============================

plt.figure(figsize=(12,5))
res[grid].plot()
plt.title("Residential3 Electricity Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.show()

# ==============================
# 2. HOURLY PATTERN
# ==============================

res["hour"] = res.index.hour
hourly = res.groupby("hour")[grid].mean()

plt.figure(figsize=(8,5))
hourly.plot(marker='o')
plt.title("Average Consumption by Hour")
plt.xlabel("Hour")
plt.ylabel("kWh")
plt.show()

# ==============================
# 3. DAILY PATTERN
# ==============================

res["day"] = res.index.dayofweek
daily = res.groupby("day")[grid].mean()

plt.figure(figsize=(8,5))
daily.plot(kind="bar")
plt.title("Average Consumption by Day of Week")
plt.xlabel("Day (0=Mon)")
plt.ylabel("kWh")
plt.show()

# ==============================
# 4. PV vs CONSUMPTION
# ==============================

plt.figure(figsize=(8,5))
plt.scatter(res[pv], res[grid], alpha=0.3)
plt.title("PV Generation vs Grid Consumption")
plt.xlabel("PV (kWh)")
plt.ylabel("Grid Import (kWh)")
plt.show()

# ==============================
# 5. CORRELATION HEATMAP
# ==============================

subset = res[[grid, pv]].copy()

plt.figure(figsize=(6,4))
sns.heatmap(subset.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation: PV vs Consumption")
plt.show()

# ==============================
# 6. DISTRIBUTION
# ==============================

plt.figure(figsize=(8,5))
sns.histplot(res[grid], bins=50, kde=True)
plt.title("Distribution of Electricity Consumption")
plt.xlabel("kWh")
plt.show()

# ==============================
# 7. ROLLING TREND (SMOOTHING)
# ==============================

rolling = res[grid].rolling(window=96).mean()  # 1 day

plt.figure(figsize=(12,5))
rolling.plot()
plt.title("Smoothed Consumption (Daily Rolling Average)")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.show()