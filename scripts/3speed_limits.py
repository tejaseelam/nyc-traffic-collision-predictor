from pathlib import Path
import argparse
import json
import pandas as pd
import requests
import shapely


ROOT = Path(__file__).resolve().parent.parent
URL = "https://data.cityofnewyork.us/resource/5mad-ntua.json"


def normalize(streets):
    return (streets.fillna("").astype(str).str.upper().str.strip().str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())


def download():
    batches = []
    offset = 0
    while True:
        response = requests.get(URL, params={
            "$select": ":id AS source_row_id,the_geom,street,postvz_sl,postvz_sg,shape_leng",
            "$order": ":id", "$limit": 25000, "$offset": offset,
        }, timeout=120)
        response.raise_for_status()
        batch = pd.DataFrame(response.json())

        if batch.empty:
            break
        batch["the_geom"] = batch["the_geom"].map(json.dumps)

        batches.append(batch)

        offset += len(batch)

        print(f"Speed limits: {offset:,} roads")

        if len(batch) < 25000:
            break
    return pd.concat(batches, ignore_index=True)


def clean(raw):
    roads = raw.copy()
    geometry_text = roads["the_geom"].astype(object).where(roads["the_geom"].notna(), None)
    lines = shapely.from_geojson(geometry_text.to_numpy(), on_invalid="ignore")
    bounds = shapely.bounds(lines)

    valid = shapely.is_valid(lines) & ~shapely.is_empty(lines)
    valid &= pd.Series(shapely.get_type_id(lines)).isin([1, 5]).to_numpy()
    valid &= (bounds[:, 0] >= -74.3) & (bounds[:, 2] <= -73.65)
    valid &= (bounds[:, 1] >= 40.45) & (bounds[:, 3] <= 40.95)

    roads = roads.loc[valid].copy()
    lines = shapely.normalize(lines[valid])

    roads["speed_geometry_wkb"] = list(shapely.to_wkb(lines))
    roads["speed_street"] = roads["street"].astype("string").str.strip().str.upper()
    roads["speed_street_normalized"] = normalize(roads["street"])
    mph = pd.to_numeric(roads["postvz_sl"], errors="coerce")

    roads["speed_limit_snapshot_mph"] = mph.where(mph.gt(0) & mph.le(85) & mph.mod(1).eq(0)).astype("Int64")
    roads["speed_signed"] = (roads["postvz_sg"].astype("string").str.upper().map({"YES": True, "NO": False}).astype("boolean"))
    roads["speed_source_row_id"] = roads["source_row_id"].astype("string")

    columns = ["speed_source_row_id", "speed_street", "speed_street_normalized",
               "speed_limit_snapshot_mph", "speed_signed", "speed_geometry_wkb"]
    roads = roads[columns]

    return roads.drop_duplicates([
        "speed_street_normalized", "speed_limit_snapshot_mph",
        "speed_signed", "speed_geometry_wkb",
    ]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-download", action="store_true")

    args = parser.parse_args()
    raw_path = args.root / "data/raw/nyc_speed_limits_snapshot.parquet"
    output = args.root / "data/processed/speed_limits_clean.parquet"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download().to_parquet(raw_path, index=False)
    clean(pd.read_parquet(raw_path)).to_parquet(output, index=False)
    print("done")


if __name__ == "__main__":
    main()
