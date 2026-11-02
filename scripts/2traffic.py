from pathlib import Path
import argparse
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
URL = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
COLUMNS = ["requestid", "boro", "yr", "m", "d", "hh", "mm", "vol", "segmentid",
           "wktgeom", "street", "fromst", "tost", "direction"]


def download():
    batches = []
    offset = 0
    where = "(yr between 2021 and 2024) OR (yr between 21 and 24)"
    while True:
        response = requests.get(URL, params={
            "$where": where, "$select": ",".join(COLUMNS),
            "$order": ":id", "$limit": 50000, "$offset": offset,
        }, timeout=120)
        response.raise_for_status()
        batch = pd.DataFrame(response.json()).reindex(columns=COLUMNS)
        if batch.empty:
            break
        batches.append(batch.astype("string"))
        offset += len(batch)
        print(f"Traffic: {offset:,} records")
        if len(batch) < 50000:
            break
    return pd.concat(batches, ignore_index=True)


def clean(raw):
    traffic = raw.copy()
    numbers = ["requestid", "segmentid", "yr", "m", "d", "hh", "mm", "vol"]
    for column in numbers:
        traffic[column] = pd.to_numeric(traffic[column], errors="coerce")

    valid = traffic[numbers].notna().all(axis=1) & traffic[numbers].mod(1).eq(0).all(axis=1)
    valid &= traffic["hh"].between(0, 23) & traffic["mm"].isin([0, 15, 30, 45])
    valid &= traffic["vol"].ge(0) & traffic["segmentid"].gt(0) & traffic["requestid"].ge(0)
    traffic.loc[traffic["yr"].between(0, 99), "yr"] += 2000
    valid &= traffic["yr"].between(2021, 2024)

    for column in ["boro", "street", "fromst", "tost", "direction"]:
        traffic[column] = traffic[column].fillna("").str.strip().str.upper()
    valid &= traffic["direction"].ne("")
    valid &= traffic["boro"].isin(["MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"])
    traffic = traffic.loc[valid].copy()

    traffic["traffic_datetime"] = pd.to_datetime(dict(
        year=traffic["yr"], month=traffic["m"], day=traffic["d"]
    ), errors="coerce")
    traffic["traffic_datetime"] += pd.to_timedelta(traffic["hh"], unit="h")
    traffic["traffic_datetime"] += pd.to_timedelta(traffic["mm"], unit="m")
    traffic = traffic.dropna(subset=["traffic_datetime"])
    traffic[["requestid", "segmentid"]] = traffic[["requestid", "segmentid"]].astype("int64")

    quarter = ["requestid", "segmentid", "boro", "direction", "traffic_datetime"]
    conflicting = traffic.groupby(quarter)["vol"].transform("nunique").gt(1)
    traffic = traffic.loc[~conflicting].drop_duplicates(quarter).copy()
    traffic["traffic_hour"] = traffic["traffic_datetime"].dt.floor("h")

    direction_hour = ["requestid", "segmentid", "boro", "direction", "traffic_hour"]
    directions = traffic.groupby(direction_hour, as_index=False).agg(
        volume=("vol", "sum"), quarters=("mm", "nunique"))
    site_hour = ["requestid", "segmentid", "boro", "traffic_hour"]

    incomplete = directions.loc[directions["quarters"] != 4, site_hour].drop_duplicates()

    hours = directions.groupby(site_hour, as_index=False).agg(
        traffic_nearby_segment_volume=("volume", "sum"),
        traffic_directions_observed=("direction", "nunique"))
    
    hours = hours.merge(incomplete.assign(incomplete=True), on=site_hour, how="left")
    hours = hours.loc[hours["incomplete"].isna()].drop(columns="incomplete")
    hours = hours.loc[~hours.duplicated(["segmentid", "traffic_hour"], keep=False)]

    site = ["requestid", "segmentid", "boro"]

    roads = traffic[site + ["street", "fromst", "tost"]].drop_duplicates()

    roads = roads.loc[~roads.duplicated(site, keep=False)]
    hours = hours.merge(roads, on=site, how="inner")

    return hours.rename(columns={
        "requestid": "traffic_request_id", "boro": "traffic_borough",
        "street": "traffic_street", "fromst": "traffic_from_street",
        "tost": "traffic_to_street",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    raw_path = args.root / "data/raw/nyc_traffic_2021_2024.parquet"
    output = args.root / "data/processed/traffic_clean.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download().to_parquet(raw_path, index=False)
    clean(pd.read_parquet(raw_path)).to_parquet(output, index=False)
    print("done")


if __name__ == "__main__":
    main()
