from pathlib import Path
from io import StringIO
import argparse
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
STATION = "USW00094728"
URL = "https://www.ncei.noaa.gov/access/services/data/v1"
FIELDS = {
    "TMAX": "weather_tmax_c",
    "TMIN": "weather_tmin_c",
    "PRCP": "weather_precip_mm",
    "SNOW": "weather_snowfall_mm",
    "SNWD": "weather_snow_depth_mm",
    "AWND": "weather_wind_mps",
}


def download():
    years = []
    for year in range(2021, 2025):
        response = requests.get(URL, params={
            "dataset": "daily-summaries", "stations": STATION,
            "startDate": f"{year}-01-01", "endDate": f"{year}-12-31",
            "dataTypes": ",".join(FIELDS), "units": "metric",
            "format": "csv", "includeAttributes": "true",
        }, timeout=90)
        response.raise_for_status()
        years.append(pd.read_csv(StringIO(response.text), dtype=str))
        print(f"Weather {year}: {len(years[-1]):,} days")
    return pd.concat(years, ignore_index=True)


def clean(raw):
    raw = raw.dropna(subset=["DATE", "STATION"]).drop_duplicates().copy()
    raw["weather_date"] = pd.to_datetime(raw["DATE"])
    raw = raw.drop_duplicates("weather_date")

    weather = raw[["weather_date"]].copy()
    weather["weather_station"] = raw["STATION"]
    weather["weather_record_present"] = True

    for code, name in FIELDS.items():
        attributes = raw[f"{code}_ATTRIBUTES"].fillna("").str.split(",")
        quality_flag = attributes.str[1].fillna("").str.strip().ne("")
        values = pd.to_numeric(raw[code], errors="coerce").astype("Float64")
        weather[name] = values.mask(values.eq(-9999) | quality_flag)
        if code in {"PRCP", "SNOW", "SNWD", "AWND"}:
            weather[name] = weather[name].mask(weather[name].lt(0))
        weather[name + "_quality_flagged"] = quality_flag
        if code in {"PRCP", "SNOW"}:
            weather[name + "_trace"] = attributes.str[0].fillna("").eq("T") & ~quality_flag

    inverted = weather["weather_tmin_c"] > weather["weather_tmax_c"]
    weather.loc[inverted.fillna(False), ["weather_tmin_c", "weather_tmax_c"]] = pd.NA
    weather["weather_temp_midpoint_c"] = (weather["weather_tmax_c"] + weather["weather_tmin_c"]) / 2
    weather["weather_precip_day"] = weather["weather_precip_mm"].gt(0) | weather["weather_precip_mm_trace"]
    weather["weather_snow_day"] = weather["weather_snowfall_mm"].gt(0) | weather["weather_snowfall_mm_trace"]

    days = pd.date_range("2021-01-01", "2024-12-31", name="weather_date")
    weather = weather.set_index("weather_date").reindex(days).reset_index()
    weather["weather_record_present"] = weather["weather_record_present"].fillna(False)
    weather["weather_station"] = STATION
    weather["weather_any_missing"] = weather[list(FIELDS.values())].isna().any(axis=1)
    return weather


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    raw_path = args.root / "data/raw/nyc_weather_2021_2024.parquet"
    output = args.root / "data/processed/weather_clean.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download().to_parquet(raw_path, index=False)
    clean(pd.read_parquet(raw_path)).to_parquet(output, index=False)
    print("done")


if __name__ == "__main__":
    main()
