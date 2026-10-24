
from pathlib import Path
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "nyc_borough_boundaries.geojson"

URL = "https://data.cityofnewyork.us/resource/gthc-hcne.geojson"


print("Downloading NYC borough boundaries...")

boroughs = gpd.read_file(URL)

print("\nColumns:")
print(boroughs.columns.tolist())

print("\nNumber of geographic features:", len(boroughs))
print("Coordinate reference system:", boroughs.crs)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
boroughs.to_file(OUTPUT, driver="GeoJSON")

print("\nSaved to:", OUTPUT)
