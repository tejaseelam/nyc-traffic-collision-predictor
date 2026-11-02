
from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "crashes_full_clean.parquet"
BOUNDARIES = ROOT / "data" / "raw" / "nyc_borough_boundaries.geojson"
OUTPUT = ROOT / "data" / "processed" / "crashes_geo_enriched.parquet"

df = pd.read_parquet(INPUT)
boroughs = gpd.read_file(BOUNDARIES)

print("Total collision records:", len(df))
print("Unknown boroughs before:", (df["borough"] == "UNKNOWN").sum())

boroughs = boroughs.to_crs("EPSG:4326")
boroughs["boroname"] = boroughs["boroname"].str.upper()

eligible = (
    (df["borough"] == "UNKNOWN")
    & df["valid_location"]
)

points = gpd.GeoDataFrame(
    df.loc[eligible, ["collision_id", "latitude", "longitude"]].copy(),
    geometry=gpd.points_from_xy(
        df.loc[eligible, "longitude"],
        df.loc[eligible, "latitude"]
    ),
    crs="EPSG:4326"
)

matched = gpd.sjoin(
    points,
    boroughs[["boroname", "geometry"]],
    how="left",
    predicate="within"
)

matched = matched.drop_duplicates(subset=["collision_id"])

borough_lookup = matched.set_index("collision_id")["boroname"]

df["borough_original"] = df["borough"]
df["borough_inferred"] = df["collision_id"].map(borough_lookup)

df["borough_final"] = df["borough_original"]
missing = df["borough_original"] == "UNKNOWN"

df.loc[missing, "borough_final"] = (
    df.loc[missing, "borough_inferred"]
    .fillna("UNKNOWN")
)

df["borough_source"] = "ORIGINAL"
df.loc[missing, "borough_source"] = "UNKNOWN"
df.loc[
    missing & df["borough_inferred"].notna(),
    "borough_source"
] = "SPATIAL_JOIN"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT, index=False)

print("\nGEOGRAPHIC ENRICHMENT SUMMARY")
print("Recovered boroughs:", (df["borough_source"] == "SPATIAL_JOIN").sum())
print("Unknown boroughs remaining:", (df["borough_final"] == "UNKNOWN").sum())

print("\nFinal borough distribution:")
print(df["borough_final"].value_counts())

print("\nSaved to:", OUTPUT)
