import pandas as pd
import os

# Path to your 1-min dataset (relative to project root)
input_file = "data\\household_data_1min_singleindex.csv"

# Load the dataset
df = pd.read_csv(input_file, parse_dates=["utc_timestamp"])

# Project root: go **one level up** from EDA folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder to save 1-min household data in root-level data folder
output_folder = os.path.join(project_root, "data", "1min")

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get household prefixes
households = set([col.split("_")[2] for col in df.columns if col.startswith("DE_KN")])

# Save each household separately
for h in households:
    cols = [c for c in df.columns if f"_{h}_" in c or c in ["utc_timestamp", "cet_cest_timestamp", "interpolated"]]
    output_file = os.path.join(output_folder, f"{h}_1min.csv")
    df[cols].to_csv(output_file, index=False)
    print(f"Saved {output_file}")