
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "nyc_crashes_sample.csv"

params = {
    "$where": (
        "crash_date >= '2021-01-01T00:00:00' "
        "AND crash_date < '2025-01-01T00:00:00'"
    ),
    "$order": "crash_date ASC, collision_id ASC",
    "$limit": 50000
}

response = requests.get(
    BASE_URL,
    params=params,
    timeout=120
)

response.raise_for_status()

df = pd.DataFrame(response.json())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"Downloaded {len(df):,} records")
print(f"Saved to: {OUTPUT}")
print(df.head())
