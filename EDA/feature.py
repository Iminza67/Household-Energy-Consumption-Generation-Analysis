import pandas as pd
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
    raise FileNotFoundError(f"Could not find {csv_name}")

print("Loading from:", csv_path)
df = pd.read_csv(csv_path)

# Convert timestamp
df["cet_cest_timestamp"] = pd.to_datetime(df["cet_cest_timestamp"], utc=True)

# ==============================
# ANALYZE RESIDENTIAL HOUSEHOLDS
# ==============================

results = []

for i in range(1, 7):
    prefix = f"DE_KN_residential{i}"
    cols = [c for c in df.columns if c.startswith(prefix)]

    if not cols:
        continue

    subset = df[cols].copy()

    # 1. Number of features
    num_features = len(cols)

    # 2. Missing values
    total_missing = subset.isnull().sum().sum()

    # 3. Percentage missing
    total_values = subset.shape[0] * subset.shape[1]
    missing_pct = (total_missing / total_values) * 100

    # 4. Variability (std of grid_import)
    grid_col = f"{prefix}_grid_import"
    std_val = subset[grid_col].std() if grid_col in subset.columns else None

    # 5. Time coverage (non-null rows for grid_import)
    non_null_count = subset[grid_col].notnull().sum() if grid_col in subset.columns else 0

    results.append({
        "household": f"residential{i}",
        "num_features": num_features,
        "missing_values": total_missing,
        "missing_%": round(missing_pct, 2),
        "std_dev": round(std_val, 2) if std_val else None,
        "non_null_rows": non_null_count
    })

# ==============================
# CREATE RESULTS TABLE
# ==============================

results_df = pd.DataFrame(results)

print("\n=== Residential Comparison ===")
print(results_df.sort_values(by="missing_%"))

# ==============================
# BEST HOUSEHOLD (simple ranking)
# ==============================

print("\n=== Suggested Best Household ===")

best = results_df.sort_values(
    by=["missing_%", "num_features", "std_dev"],
    ascending=[True, False, False]
).iloc[0]

print(best)

#we will use residential3 as it has the lowest missing percentage,
# a good number of features, and decent variability.