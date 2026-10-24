
from pathlib import Path
import time
import pandas as pd
import requests

BASE_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BATCH_DIR = RAW_DIR / "crash_batches"
OUTPUT = RAW_DIR / "nyc_crashes_full.csv"

BATCH_SIZE = 50000

RAW_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()

where = (
    "crash_date >= '2021-01-01T00:00:00' "
    "AND crash_date < '2025-01-01T00:00:00'"
)

offset = 0
batch_number = 1

while True:
    batch_file = BATCH_DIR / f"batch_{batch_number:04d}.csv"

    if batch_file.exists():
        df = pd.read_csv(batch_file, low_memory=False)
        print(f"Using existing batch {batch_number}: {len(df):,} rows")
    else:
        params = {
            "$where": where,
            "$order": "crash_date ASC, collision_id ASC",
            "$limit": BATCH_SIZE,
            "$offset": offset
        }

        for attempt in range(5):
            try:
                response = session.get(
                    BASE_URL,
                    params=params,
                    timeout=180
                )
                response.raise_for_status()

                batch_file.write_bytes(response.content)
                df = pd.read_csv(batch_file, low_memory=False)
                break

            except (requests.RequestException, pd.errors.ParserError):
                batch_file.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

        print(f"Downloaded batch {batch_number}: {len(df):,} rows")

    if len(df) == 0:
        batch_file.unlink(missing_ok=True)
        break

    offset += len(df)

    if len(df) < BATCH_SIZE:
        break

    batch_number += 1
    time.sleep(0.5)

print("\nCombining batches...")

batch_files = sorted(BATCH_DIR.glob("batch_*.csv"))

with OUTPUT.open("wb") as outfile:
    for i, file in enumerate(batch_files):
        with file.open("rb") as infile:
            if i > 0:
                infile.readline()

            while True:
                chunk = infile.read(1024 * 1024)
                if not chunk:
                    break
                outfile.write(chunk)

print(f"\nSaved full dataset to: {OUTPUT}")
print(f"Total records downloaded: {offset:,}")
print(f"Total batches: {len(batch_files)}")
