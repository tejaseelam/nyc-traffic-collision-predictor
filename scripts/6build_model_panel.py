from pathlib import Path
import argparse
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely


ROOT = Path(__file__).resolve().parents[1]
METRIC_CRS = "EPSG:32618"
DISPLAY_CRS = "EPSG:4326"
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2025-01-01"


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def make_cell_ids(grid_x, grid_y):
    return np.array(
        [f"NYC_{x}_{y}" for x, y in zip(grid_x, grid_y)],
        dtype=object,
    )


def bin_crash_times(values, period_hours):
    return pd.to_datetime(values, errors="coerce").dt.floor(f"{period_hours}h")


def build_grid(boroughs, grid_size_m, minimum_land_fraction):
    require_columns(boroughs, ["boroname", "geometry"], "Borough boundaries")

    boroughs = boroughs[["boroname", "geometry"]].copy().to_crs(METRIC_CRS)
    boroughs["boroname"] = boroughs["boroname"].astype("string").str.upper()
    boroughs = boroughs.loc[boroughs.geometry.notna() & ~boroughs.geometry.is_empty]
    if boroughs.empty:
        raise ValueError("Borough boundary data has no usable geometry")

    land = shapely.union_all(boroughs.geometry.to_numpy())
    min_x, min_y, max_x, max_y = shapely.bounds(land)
    first_x = math.floor(min_x / grid_size_m)
    last_x = math.ceil(max_x / grid_size_m)
    first_y = math.floor(min_y / grid_size_m)
    last_y = math.ceil(max_y / grid_size_m)

    x_indexes = np.arange(first_x, last_x, dtype=np.int32)
    y_indexes = np.arange(first_y, last_y, dtype=np.int32)
    grid_x = np.tile(x_indexes, len(y_indexes))
    grid_y = np.repeat(y_indexes, len(x_indexes))
    min_cell_x = grid_x.astype(np.float64) * grid_size_m
    min_cell_y = grid_y.astype(np.float64) * grid_size_m
    boxes = shapely.box(
        min_cell_x,
        min_cell_y,
        min_cell_x + grid_size_m,
        min_cell_y + grid_size_m,
    )

    land_area = shapely.area(shapely.intersection(boxes, land))
    land_fraction = land_area / float(grid_size_m ** 2)
    keep = land_fraction >= minimum_land_fraction
    boxes = boxes[keep]
    grid_x = grid_x[keep]
    grid_y = grid_y[keep]
    land_area = land_area[keep]
    land_fraction = land_fraction[keep]

    if len(boxes) == 0:
        raise ValueError("No grid cells remain after applying the land threshold")

    borough_areas = np.column_stack([
        shapely.area(shapely.intersection(boxes, geometry))
        for geometry in boroughs.geometry.to_numpy()
    ])
    borough_index = borough_areas.argmax(axis=1)
    borough_names = boroughs["boroname"].to_numpy()[borough_index]

    grid = gpd.GeoDataFrame(
        {
            "cell_id": make_cell_ids(grid_x, grid_y),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "borough": borough_names,
            "land_area_m2": land_area.astype(np.float32),
            "land_fraction": land_fraction.astype(np.float32),
        },
        geometry=boxes,
        crs=METRIC_CRS,
    )
    return grid.sort_values(["grid_y", "grid_x"]).reset_index(drop=True)


def add_speed_features(grid, speed):
    require_columns(
        speed,
        [
            "speed_source_row_id",
            "speed_geometry_wkb",
            "speed_limit_snapshot_mph",
        ],
        "Speed-limit data",
    )

    lines = shapely.from_wkb(speed["speed_geometry_wkb"].to_numpy(), on_invalid="ignore")
    roads = gpd.GeoDataFrame(
        speed[["speed_source_row_id", "speed_limit_snapshot_mph"]].copy(),
        geometry=lines,
        crs=DISPLAY_CRS,
    ).to_crs(METRIC_CRS)
    roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].reset_index(drop=True)

    candidates = gpd.sjoin(
        roads,
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if candidates.empty:
        raise ValueError("No speed-limit roads intersect the grid")

    cell_geometry = grid.geometry.iloc[candidates["index_right"].to_numpy()].to_numpy()
    road_geometry = candidates.geometry.to_numpy()
    candidates["length_m"] = shapely.length(
        shapely.intersection(road_geometry, cell_geometry)
    )
    candidates = candidates.loc[candidates["length_m"] > 0].copy()

    all_features = candidates.groupby("cell_id", as_index=False).agg(
        speed_road_length_m=("length_m", "sum"),
        speed_road_segments=("speed_source_row_id", "nunique"),
    )

    known = candidates.loc[candidates["speed_limit_snapshot_mph"].notna()].copy()
    known["weighted_speed"] = (
        known["speed_limit_snapshot_mph"].astype(float) * known["length_m"]
    )
    known_features = known.groupby("cell_id", as_index=False).agg(
        speed_known_length_m=("length_m", "sum"),
        speed_weighted_sum=("weighted_speed", "sum"),
        speed_limit_min_mph=("speed_limit_snapshot_mph", "min"),
        speed_limit_max_mph=("speed_limit_snapshot_mph", "max"),
    )
    known_features["speed_limit_weighted_avg_mph"] = (
        known_features["speed_weighted_sum"]
        / known_features["speed_known_length_m"]
    )
    known_features = known_features.drop(columns="speed_weighted_sum")

    features = all_features.merge(known_features, on="cell_id", how="left")
    features["speed_limit_coverage"] = (
        features["speed_known_length_m"] / features["speed_road_length_m"]
    )
    return grid.merge(features, on="cell_id", how="left", validate="one_to_one")


def assign_crashes_to_cells(crashes, grid, grid_size_m, period_hours, start, end):
    require_columns(
        crashes,
        [
            "collision_id",
            "crash_datetime",
            "latitude",
            "longitude",
            "valid_datetime",
            "valid_location",
        ],
        "Crash data",
    )

    working = crashes.copy()
    working["crash_datetime"] = pd.to_datetime(
        working["crash_datetime"], errors="coerce"
    )
    working["latitude"] = pd.to_numeric(working["latitude"], errors="coerce")
    working["longitude"] = pd.to_numeric(working["longitude"], errors="coerce")

    valid_datetime = (
        working["valid_datetime"].fillna(False).astype(bool)
        & working["crash_datetime"].notna()
        & working["crash_datetime"].ge(start)
        & working["crash_datetime"].lt(end)
    )
    valid_location = (
        working["valid_location"].fillna(False).astype(bool)
        & working["latitude"].notna()
        & working["longitude"].notna()
    )
    eligible = working.loc[
        valid_datetime & valid_location,
        ["collision_id", "crash_datetime", "latitude", "longitude"],
    ].copy()

    points = gpd.GeoSeries(
        gpd.points_from_xy(eligible["longitude"], eligible["latitude"]),
        crs=DISPLAY_CRS,
    ).to_crs(METRIC_CRS)
    eligible["grid_x"] = np.floor(points.x.to_numpy() / grid_size_m).astype(np.int32)
    eligible["grid_y"] = np.floor(points.y.to_numpy() / grid_size_m).astype(np.int32)
    eligible["cell_id"] = make_cell_ids(
        eligible["grid_x"].to_numpy(),
        eligible["grid_y"].to_numpy(),
    )
    eligible["time_bin"] = bin_crash_times(
        eligible["crash_datetime"], period_hours
    )

    grid_ids = set(grid["cell_id"])
    inside_grid = eligible["cell_id"].isin(grid_ids)
    assigned = eligible.loc[inside_grid, ["collision_id", "cell_id", "time_bin"]]
    if assigned["collision_id"].duplicated().any():
        raise ValueError("A collision was assigned to more than one grid cell")

    counts = assigned.groupby(
        ["cell_id", "time_bin"], as_index=False
    ).size().rename(columns={"size": "crash_count"})
    counts["crash_count"] = counts["crash_count"].astype(np.int32)

    summary = {
        "input_crashes": int(len(working)),
        "invalid_or_out_of_range_datetime": int((~valid_datetime).sum()),
        "invalid_location": int((valid_datetime & ~valid_location).sum()),
        "eligible_crashes": int(len(eligible)),
        "outside_retained_grid": int((~inside_grid).sum()),
        "assigned_crashes": int(len(assigned)),
    }
    return counts, summary


def prepare_weather(weather):
    require_columns(
        weather,
        [
            "weather_date",
            "weather_temp_midpoint_c",
            "weather_precip_mm",
            "weather_snowfall_mm",
            "weather_snow_depth_mm",
            "weather_wind_mps",
            "weather_precip_day",
            "weather_snow_day",
            "weather_any_missing",
        ],
        "Weather data",
    )
    columns = [
        "weather_date",
        "weather_temp_midpoint_c",
        "weather_precip_mm",
        "weather_snowfall_mm",
        "weather_snow_depth_mm",
        "weather_wind_mps",
        "weather_precip_day",
        "weather_snow_day",
        "weather_any_missing",
    ]
    weather = weather[columns].copy()
    weather["weather_date"] = pd.to_datetime(
        weather["weather_date"], errors="coerce"
    ).dt.normalize()
    weather = weather.dropna(subset=["weather_date"])
    if weather["weather_date"].duplicated().any():
        raise ValueError("Weather data contains duplicate dates")
    return weather


def repeat_static(values, period_count):
    return np.tile(values, period_count)


def repeat_time(values, cell_count):
    return np.repeat(values, cell_count)


def build_panel(
    grid,
    periods,
    crash_counts,
    weather,
    output,
    chunk_periods,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)

    crash_lookup = crash_counts.set_index(["cell_id", "time_bin"])["crash_count"]
    weather = weather.set_index("weather_date")
    cell_count = len(grid)
    writer = None
    written_rows = 0
    written_crashes = 0

    static_columns = [
        "cell_id",
        "grid_x",
        "grid_y",
        "borough",
        "land_area_m2",
        "land_fraction",
        "speed_road_length_m",
        "speed_road_segments",
        "speed_known_length_m",
        "speed_limit_min_mph",
        "speed_limit_max_mph",
        "speed_limit_weighted_avg_mph",
        "speed_limit_coverage",
    ]
    for column in static_columns:
        if column not in grid:
            grid[column] = pd.NA

    try:
        for start_index in range(0, len(periods), chunk_periods):
            chunk_times = periods[start_index:start_index + chunk_periods]
            time_frame = pd.DataFrame({"time_bin": chunk_times})
            time_frame["weather_date"] = time_frame["time_bin"].dt.normalize()
            time_frame = time_frame.join(weather, on="weather_date")

            cell_ids = repeat_static(grid["cell_id"].to_numpy(), len(chunk_times))
            time_values = repeat_time(chunk_times.to_numpy(), cell_count)
            panel_index = pd.MultiIndex.from_arrays(
                [cell_ids, time_values], names=["cell_id", "time_bin"]
            )
            counts = crash_lookup.reindex(panel_index, fill_value=0).to_numpy(np.int32)

            chunk = pd.DataFrame({
                "cell_id": cell_ids,
                "time_bin": time_values,
                "crash_count": counts,
                "has_crash": counts > 0,
            })
            for column in static_columns[1:]:
                chunk[column] = repeat_static(grid[column].to_numpy(), len(chunk_times))

            chunk["year"] = repeat_time(time_frame["time_bin"].dt.year.to_numpy(np.int16), cell_count)
            chunk["month"] = repeat_time(time_frame["time_bin"].dt.month.to_numpy(np.int8), cell_count)
            chunk["day_of_week"] = repeat_time(
                time_frame["time_bin"].dt.day_of_week.to_numpy(np.int8), cell_count
            )
            chunk["hour"] = repeat_time(time_frame["time_bin"].dt.hour.to_numpy(np.int8), cell_count)
            chunk["is_weekend"] = chunk["day_of_week"].isin([5, 6])

            for column in weather.columns:
                chunk[column] = repeat_time(time_frame[column].to_numpy(), cell_count)

            for column in [
                "land_area_m2",
                "land_fraction",
                "speed_road_length_m",
                "speed_known_length_m",
                "speed_limit_weighted_avg_mph",
                "speed_limit_coverage",
                "weather_temp_midpoint_c",
                "weather_precip_mm",
                "weather_snowfall_mm",
                "weather_snow_depth_mm",
                "weather_wind_mps",
            ]:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Float32")
            chunk["speed_road_segments"] = pd.to_numeric(
                chunk["speed_road_segments"], errors="coerce"
            ).fillna(0).astype(np.int32)
            for column in ["speed_limit_min_mph", "speed_limit_max_mph"]:
                chunk[column] = pd.to_numeric(
                    chunk[column], errors="coerce"
                ).astype("Int16")
            for column in [
                "weather_precip_day",
                "weather_snow_day",
                "weather_any_missing",
            ]:
                chunk[column] = chunk[column].astype("boolean")

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["cell_id", "borough"],
                )
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            written_rows += len(chunk)
            written_crashes += int(counts.sum())
            print(
                f"Panel periods {start_index + 1:,}-{start_index + len(chunk_times):,} "
                f"of {len(periods):,}: {written_rows:,} rows"
            )
    finally:
        if writer is not None:
            writer.close()

    expected_rows = len(grid) * len(periods)
    expected_crashes = int(crash_counts["crash_count"].sum())
    if written_rows != expected_rows:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Panel has {written_rows:,} rows; expected {expected_rows:,}"
        )
    if written_crashes != expected_crashes:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Panel contains {written_crashes:,} crashes; expected {expected_crashes:,}"
        )

    temporary_output.replace(output)
    return {
        "panel_rows": int(written_rows),
        "panel_crashes": int(written_crashes),
        "rows_with_crashes": int(len(crash_counts)),
        "zero_crash_rows": int(written_rows - len(crash_counts)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--grid-size-m", type=int, default=500)
    parser.add_argument("--period-hours", type=int, default=6)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--minimum-land-fraction", type=float, default=0.01)
    parser.add_argument("--chunk-periods", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.grid_size_m <= 0:
        raise ValueError("Grid size must be positive")
    if args.period_hours <= 0 or 24 % args.period_hours != 0:
        raise ValueError("Period hours must be a positive divisor of 24")
    if not 0 <= args.minimum_land_fraction <= 1:
        raise ValueError("Minimum land fraction must be between 0 and 1")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end <= start:
        raise ValueError("End date must be after start date")

    raw = args.root / "data/raw"
    processed = args.root / "data/processed"
    paths = {
        "borough boundaries": raw / "nyc_borough_boundaries.geojson",
        "crashes": processed / "crashes_geo_enriched.parquet",
        "weather": processed / "weather_clean.parquet",
        "speed limits": processed / "speed_limits_clean.parquet",
    }
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing inputs. Run the earlier pipeline steps first:\n" + "\n".join(missing)
        )

    output = args.output or processed / "model_panel_v1.parquet"
    grid_parquet = processed / f"nyc_grid_{args.grid_size_m}m.parquet"
    grid_geojson = processed / f"nyc_grid_{args.grid_size_m}m.geojson"
    summary_path = processed / "model_panel_v1_summary.json"

    print("Building NYC grid...")
    boroughs = gpd.read_file(paths["borough boundaries"])
    grid = build_grid(
        boroughs,
        args.grid_size_m,
        args.minimum_land_fraction,
    )
    print(f"Retained grid cells: {len(grid):,}")

    print("Adding area-level speed-limit features...")
    speed = pd.read_parquet(paths["speed limits"])
    grid = add_speed_features(grid, speed)

    print("Assigning crashes to grid cells and six-hour periods...")
    crashes = pd.read_parquet(paths["crashes"])
    crash_counts, crash_summary = assign_crashes_to_cells(
        crashes,
        grid,
        args.grid_size_m,
        args.period_hours,
        start,
        end,
    )
    print(f"Assigned crashes: {crash_summary['assigned_crashes']:,}")

    periods = pd.date_range(
        start,
        end,
        freq=f"{args.period_hours}h",
        inclusive="left",
    )
    weather = prepare_weather(pd.read_parquet(paths["weather"]))

    print("Writing model panel in chunks...")
    panel_summary = build_panel(
        grid.drop(columns="geometry"),
        periods,
        crash_counts,
        weather,
        output,
        args.chunk_periods,
    )

    grid_display = grid.to_crs(DISPLAY_CRS)
    grid_display.to_parquet(grid_parquet, index=False)
    grid_display.to_file(grid_geojson, driver="GeoJSON")

    summary = {
        "grid_size_m": args.grid_size_m,
        "period_hours": args.period_hours,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "minimum_land_fraction": args.minimum_land_fraction,
        "grid_cells": int(len(grid)),
        "time_periods": int(len(periods)),
        **crash_summary,
        **panel_summary,
        "traffic_included": False,
        "traffic_note": "Traffic is excluded from version 1 until typical traffic profiles are implemented.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nMODEL PANEL SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Grid GeoParquet: {grid_parquet}")
    print(f"Grid GeoJSON: {grid_geojson}")
    print(f"Model panel: {output}")
    print(f"Validation summary: {summary_path}")


if __name__ == "__main__":
    main()
