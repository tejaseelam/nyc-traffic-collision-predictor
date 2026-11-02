
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "nyc_crashes_full.csv"
OUTPUT = ROOT / "data" / "processed" / "crashes_full_clean.parquet"

df = pd.read_csv(INPUT, low_memory=False)

print("Original records:", len(df))
original_records = len(df)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Preserve collision identifiers as strings. NYC Open Data currently returns
# opaque identifiers, while older exports may contain numeric identifiers.
df["collision_id"] = df["collision_id"].astype("string").str.strip()
missing_collision_id = df["collision_id"].isna() | df["collision_id"].eq("")
df = df.loc[~missing_collision_id].copy()
df = df.drop_duplicates(subset=["collision_id"])

if df.empty:
    raise ValueError("No valid collision IDs remain after cleaning")

# Parse dates and times
df["crash_date"] = pd.to_datetime(
    df["crash_date"], errors="coerce"
)

df["crash_datetime"] = pd.to_datetime(
    df["crash_date"].dt.strftime("%Y-%m-%d")
    + " "
    + df["crash_time"].fillna("").astype(str),
    errors="coerce"
)

df["valid_datetime"] = df["crash_datetime"].notna()

# Convert coordinates
df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
)
df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
)

# Broad NYC geographic bounds
df["valid_location"] = (
    df["latitude"].between(40.4, 41.0)
    & df["longitude"].between(-74.3, -73.6)
)

# Standardize borough
df["borough"] = (
    df["borough"].fillna("UNKNOWN")
    .astype(str)
    .str.strip()
    .str.upper()
)

# Create time-based features
dt = df["crash_datetime"]

df["year"] = dt.dt.year
df["month"] = dt.dt.month
df["day_of_week"] = dt.dt.day_of_week
df["hour"] = dt.dt.hour
df["is_weekend"] = df["day_of_week"].isin([5, 6])

# Convert collision counts
count_cols = [
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_cyclist_injured",
    "number_of_motorist_injured",
]

for col in count_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Save processed dataset
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT, index=False)

print("\nCLEANING SUMMARY")
print("Cleaned records:", len(df))
print("Records without collision IDs:", int(missing_collision_id.sum()))
print("Duplicate collision IDs removed:", original_records - int(missing_collision_id.sum()) - len(df))
print("Duplicate IDs remaining:", df["collision_id"].duplicated().sum())
print("Valid timestamps:", df["valid_datetime"].sum())
print("Valid locations:", df["valid_location"].sum())
print("Invalid locations:", (~df["valid_location"]).sum())
print("\nRecords by borough:")
print(df["borough"].value_counts())
print("\nSaved to:", OUTPUT)
