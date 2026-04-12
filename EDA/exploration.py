import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

file_path = "data\\household_data_15min_singleindex.csv"

df = pd.read_csv(file_path)

df.head()

print("Shape:", df.shape)
df.info()
df.describe()

df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
df.set_index("utc_timestamp", inplace=True)

df.head()

missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])


print("\nInterpolated values:")
print(df["interpolated"].value_counts().head())

grid_cols = [c for c in df.columns if "grid_import" in c]
pv_cols = [c for c in df.columns if "_pv" in c]

industrial_cols = [c for c in grid_cols if "industrial" in c]
residential_cols = [c for c in grid_cols if "residential" in c]
public_cols = [c for c in grid_cols if "public" in c]

print("Industrial:", len(industrial_cols))
print("Residential:", len(residential_cols))
print("Public:", len(public_cols))

df["industrial_total"] = df[industrial_cols].sum(axis=1)
df["residential_total"] = df[residential_cols].sum(axis=1)
df["public_total"] = df[public_cols].sum(axis=1)

df["total_grid"] = df[grid_cols].sum(axis=1)
df["total_pv"] = df[pv_cols].sum(axis=1)

df[["industrial_total", "residential_total", "public_total"]] \
    .resample("D").mean() \
    .plot(figsize=(12, 5))

plt.title("Daily Average Energy Consumption by Sector")
plt.ylabel("kWh")
plt.show()

df[["total_grid", "total_pv"]] \
    .resample("D").sum() \
    .plot(figsize=(12, 5))

plt.title("Grid Import vs Solar PV Generation")
plt.ylabel("kWh")
plt.show()

df["hour"] = df.index.hour

hourly_usage = df.groupby("hour")["total_grid"].mean()

hourly_usage.plot(figsize=(10, 4))
plt.title("Average Energy Usage by Hour")
plt.xlabel("Hour")
plt.ylabel("kWh")
plt.show()

total_usage = df[grid_cols].sum().sort_values(ascending=False)

print("Top 10 consumers:")
print(total_usage.head(10))

plt.figure(figsize=(10, 8))
sns.heatmap(df[grid_cols].corr(), cmap="coolwarm")
plt.title("Correlation Between Households")
plt.show()
