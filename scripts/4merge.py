from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent.parent


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def normalize(streets):
    return (streets.fillna("").astype(str).str.upper().str.strip().str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())


def merge_weather(crashes, weather):
    require_columns(crashes, ["crash_date"], "Crash data")
    require_columns(
        weather,
        ["weather_date", "weather_record_present", "weather_any_missing"],
        "Weather data",
    )

    crashes = crashes.copy()
    weather = weather.copy()
    crashes["weather_date"] = pd.to_datetime(
        crashes["crash_date"], errors="coerce"
    ).dt.normalize()
    weather["weather_date"] = pd.to_datetime(
        weather["weather_date"], errors="coerce"
    ).dt.normalize()
    weather = weather.dropna(subset=["weather_date"])

    if weather["weather_date"].duplicated().any():
        raise ValueError("Weather data contains duplicate dates")

    crashes = crashes.merge(weather, on="weather_date", how="left", validate="many_to_one")

    crashes["weather_matched"] = crashes["weather_record_present"].fillna(False)
    crashes["weather_any_missing"] = crashes["weather_any_missing"].fillna(True)

    return crashes


def merge_traffic(crashes, traffic, cross_street_column):
    borough = "borough_final" if "borough_final" in crashes else "borough"
    require_columns(
        crashes,
        [borough, "crash_datetime", "on_street_name", cross_street_column],
        "Crash data",
    )
    require_columns(
        traffic,
        [
            "segmentid", "traffic_hour", "traffic_borough", "traffic_street",
            "traffic_from_street", "traffic_to_street",
            "traffic_nearby_segment_volume",
        ],
        "Traffic data",
    )

    crashes = crashes.copy()
    traffic = traffic.copy()
    traffic["traffic_hour"] = pd.to_datetime(traffic["traffic_hour"])
    crashes["traffic_hour"] = pd.to_datetime(crashes["crash_datetime"]).dt.floor("h")

    roads = []
    for endpoint in ["traffic_from_street", "traffic_to_street"]:
        road = traffic[["segmentid", "traffic_borough", "traffic_street", endpoint]].copy()
        road.columns = ["segmentid", "borough_key", "street_key", "cross_street_key"]
        roads.append(road)
    roads = pd.concat(roads).drop_duplicates()
    for column in ["borough_key", "street_key", "cross_street_key"]:
        roads[column] = normalize(roads[column])
    road_keys = ["borough_key", "street_key", "cross_street_key"]
    roads = roads.loc[roads[road_keys].ne("").all(axis=1) & roads["borough_key"].ne("UNKNOWN")]
    counts = roads.groupby(road_keys)["segmentid"].nunique().rename("traffic_candidate_segments").reset_index()
    unique_roads = roads.merge(counts, on=road_keys)
    unique_roads = unique_roads.loc[unique_roads["traffic_candidate_segments"] == 1]
    unique_roads = unique_roads[road_keys + ["segmentid"]].drop_duplicates(road_keys)
    lookup = counts.merge(unique_roads, on=road_keys, how="left")

    crashes["borough_key"] = normalize(crashes[borough])
    crashes["street_key"] = normalize(crashes["on_street_name"])
    crashes["cross_street_key"] = normalize(crashes[cross_street_column])
    crashes = crashes.merge(lookup, on=road_keys, how="left", validate="many_to_one")
    traffic["segmentid"] = traffic["segmentid"].astype("int64")
    crashes = crashes.merge(traffic, on=["segmentid", "traffic_hour"], how="left", validate="many_to_one")

    crashes["traffic_matched"] = crashes["traffic_nearby_segment_volume"].notna()
    crashes["traffic_match_status"] = "NO_UNIQUE_ROAD_MATCH"
    crashes.loc[crashes["traffic_candidate_segments"].gt(1), "traffic_match_status"] = "AMBIGUOUS_ROAD"
    crashes.loc[crashes["segmentid"].notna(), "traffic_match_status"] = "NO_COUNT_AT_CRASH_HOUR"
    crashes.loc[crashes["traffic_hour"].isna(), "traffic_match_status"] = "INVALID_CRASH_TIME"
    crashes.loc[crashes["traffic_matched"], "traffic_match_status"] = "MATCHED_NEARBY_SEGMENT"
    crashes = crashes.rename(columns={"segmentid": "traffic_segmentid"})
    return crashes.drop(columns=road_keys)


def merge_speed(crashes, speed):
    require_columns(
        crashes,
        ["latitude", "longitude", "on_street_name"],
        "Crash data",
    )
    require_columns(
        speed,
        [
            "speed_geometry_wkb", "speed_street_normalized",
            "speed_limit_snapshot_mph", "speed_source_row_id",
        ],
        "Speed-limit data",
    )

    crashes = crashes.reset_index(drop=True).copy()
    speed = speed.reset_index(drop=True).copy()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    lines = shapely.from_wkb(speed["speed_geometry_wkb"].to_numpy())
    lines = shapely.transform(lines, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])))
    tree = shapely.STRtree(lines)

    crashes["speed_limit_snapshot_mph"] = pd.Series(pd.NA, index=crashes.index, dtype="Int64")
    crashes["speed_match_distance_m"] = pd.Series(pd.NA, index=crashes.index, dtype="Float64")
    crashes["speed_source_row_id"] = pd.Series(pd.NA, index=crashes.index, dtype="string")
    crashes["speed_match_status"] = "NO_NEARBY_SAME_NAME_ROAD"
    crashes["speed_historical_verified"] = False
    crash_streets = normalize(crashes["on_street_name"])
    lon = pd.to_numeric(crashes["longitude"], errors="coerce")
    lat = pd.to_numeric(crashes["latitude"], errors="coerce")
    valid = lon.between(-74.3, -73.65) & lat.between(40.45, 40.95)
    if "valid_location" in crashes:
        valid &= crashes["valid_location"].fillna(False).astype(bool)
    crashes.loc[crash_streets.eq(""), "speed_match_status"] = "MISSING_ON_STREET"
    crashes.loc[~valid, "speed_match_status"] = "INVALID_CRASH_LOCATION"

    eligible = crashes.index[valid & crash_streets.ne("")].to_numpy()
    for start in range(0, len(eligible), 25000):
        rows = eligible[start:start + 25000]
        x, y = transformer.transform(lon.iloc[rows].to_numpy(), lat.iloc[rows].to_numpy())
        points = shapely.points(x, y)
        pairs = tree.query(points, predicate="dwithin", distance=30)
        if pairs.shape[1] == 0:
            continue
        crash_index, road_index = pairs
        same_street = (crash_streets.iloc[rows[crash_index]].to_numpy()
                       == speed["speed_street_normalized"].iloc[road_index].to_numpy())
        crash_index, road_index = crash_index[same_street], road_index[same_street]
        if len(crash_index) == 0:
            continue
        nearby = pd.DataFrame({
            "row": rows[crash_index], "road": road_index,
            "distance": shapely.distance(points[crash_index], lines[road_index]),
        })
        nearest = nearby.groupby("row")["distance"].min()
        nearby = nearby.loc[nearby["distance"] <= nearby["row"].map(nearest) + 5].copy()
        nearby["limit"] = speed["speed_limit_snapshot_mph"].iloc[nearby["road"]].to_numpy()
        different_limits = nearby["limit"].astype("string").fillna("UNKNOWN").groupby(nearby["row"]).nunique()
        ambiguous = different_limits.index[different_limits > 1]
        crashes.loc[ambiguous, "speed_match_status"] = "AMBIGUOUS_NEARBY_LIMITS"
        nearby = nearby.loc[~nearby["row"].isin(ambiguous)]
        chosen = nearby.sort_values(["row", "distance", "road"]).drop_duplicates("row")
        crashes.loc[chosen.loc[chosen["limit"].isna(), "row"], "speed_match_status"] = "NEARBY_LIMIT_UNKNOWN"
        chosen = chosen.loc[chosen["limit"].notna()]
        crashes.loc[chosen["row"], "speed_limit_snapshot_mph"] = chosen["limit"].astype(int).to_numpy()
        crashes.loc[chosen["row"], "speed_match_distance_m"] = chosen["distance"].to_numpy()
        crashes.loc[chosen["row"], "speed_source_row_id"] = speed["speed_source_row_id"].iloc[chosen["road"]].to_numpy()
        crashes.loc[chosen["row"], "speed_match_status"] = "MATCHED_CURRENT_SNAPSHOT"

    crashes["speed_matched"] = crashes["speed_limit_snapshot_mph"].notna()
    return crashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--crashes", type=Path)
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--traffic", type=Path)
    parser.add_argument("--speed-limits", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = args.root / "data/processed"
    crash_path = args.crashes or data / "crashes_geo_enriched.parquet"
    weather_path = args.weather or data / "weather_clean.parquet"
    traffic_path = args.traffic or data / "traffic_clean.parquet"
    speed_path = args.speed_limits or data / "speed_limits_clean.parquet"
    output = args.output or data / "crashes_weather_traffic_speed.parquet"

    inputs = {
        "crashes": crash_path,
        "weather": weather_path,
        "traffic": traffic_path,
        "speed limits": speed_path,
    }
    missing_inputs = [
        f"{label}: {path}"
        for label, path in inputs.items()
        if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing processed inputs. Run the cleaning scripts first:\n"
            + "\n".join(missing_inputs)
        )

    crashes = pd.read_parquet(crash_path)
    weather = pd.read_parquet(weather_path)
    traffic = pd.read_parquet(traffic_path)
    speed = pd.read_parquet(speed_path)

    original_count = len(crashes)
    original_ids = None
    if "collision_id" in crashes:
        original_ids = crashes["collision_id"].reset_index(drop=True).copy()

    crashes = merge_weather(crashes, weather)
    crashes = merge_traffic(crashes, traffic, "cross_street_name")
    crashes = merge_speed(crashes, speed)

    if len(crashes) != original_count:
        raise RuntimeError(
            "Integration changed the crash row count "
            f"from {original_count:,} to {len(crashes):,}"
        )
    if original_ids is not None:
        integrated_ids = crashes["collision_id"].reset_index(drop=True)
        if not integrated_ids.equals(original_ids):
            raise RuntimeError("Integration changed the collision IDs or their order")

    output.parent.mkdir(parents=True, exist_ok=True)
    crashes.to_parquet(output, index=False)
    print(f"Saved {len(crashes):,} crashes to {output}")
    print(f"Weather matches: {crashes['weather_matched'].sum():,}")
    print(f"Traffic matches: {crashes['traffic_matched'].sum():,}")
    print(f"Speed matches: {crashes['speed_matched'].sum():,}")


if __name__ == "__main__":
    main()
