"""Export compact, static dashboard files from the 2023 validation data.

The exporter intentionally reads only the 2021-2022 training panel and the 2023
validation panel. It never opens the 2024 test split. Run the two modeling
scripts first so their validation prediction parquet files exist under
``data/processed/predictions``.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCALE = 100_000
HOURS = [0, 6, 12, 18]
MODELS = {
    "baseline": "cell_hour_rate_rolling",
    "logistic": "logistic",
    "tree": "tree_core_plus_location",
}
MODEL_NAMES = {
    "baseline": "Rolling cell + time rate",
    "logistic": "Logistic regression",
    "tree": "LightGBM + location",
}


def compact_dump(value, path: Path) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")


def git_text(root: Path, revision_path: str) -> str:
    result = subprocess.run(
        ["git", "show", revision_path],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def rounded_geometry(geometry: dict) -> dict:
    def round_nested(value):
        if isinstance(value, list):
            return [round_nested(item) for item in value]
        return round(float(value), 6)

    return {"type": geometry["type"], "coordinates": round_nested(geometry["coordinates"])}


def export_grid(processed: Path, output: Path) -> tuple[list[str], dict]:
    source = json.loads((processed / "nyc_grid_500m.geojson").read_text())
    features = []
    for feature in source["features"]:
        props = feature["properties"]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "cell_id": props["cell_id"],
                    "borough": props.get("borough") or "UNKNOWN",
                    "land_fraction": round(float(props.get("land_fraction") or 0), 4),
                    "road_length_m": round(float(props.get("speed_road_length_m") or 0), 1),
                    "speed_limit_mph": (
                        round(float(props["speed_limit_weighted_avg_mph"]), 1)
                        if props.get("speed_limit_weighted_avg_mph") is not None
                        else None
                    ),
                },
                "geometry": rounded_geometry(feature["geometry"]),
            }
        )
    grid = {"type": "FeatureCollection", "features": features}
    compact_dump(grid, output / "grid.geojson")
    return [feature["properties"]["cell_id"] for feature in features], grid


def aligned(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label} predictions have {len(right):,} rows; expected {len(left):,}")
    sample = np.unique(np.linspace(0, len(left) - 1, 101, dtype=int))
    left_cell = left["cell_id"].astype("string").iloc[sample].reset_index(drop=True)
    right_cell = right["cell_id"].astype("string").iloc[sample].reset_index(drop=True)
    left_time = pd.to_datetime(left["time_bin"].iloc[sample]).reset_index(drop=True)
    right_time = pd.to_datetime(right["time_bin"].iloc[sample]).reset_index(drop=True)
    if not left_cell.equals(right_cell) or not left_time.equals(right_time):
        raise ValueError(f"{label} predictions are not aligned with the validation panel")


def load_validation(processed: Path) -> pd.DataFrame:
    columns = [
        "cell_id", "time_bin", "crash_count", "has_crash", "hour", "month",
        "borough", "weather_precip_day",
    ]
    validation = pd.read_parquet(processed / "model_panel_validation_2023.parquet", columns=columns)
    validation["time_bin"] = pd.to_datetime(validation["time_bin"])
    validation["cell_id"] = validation["cell_id"].astype("category")
    validation["borough"] = validation["borough"].astype("category")
    validation["wet"] = validation["weather_precip_day"].fillna(False).astype(bool)

    baseline = pd.read_parquet(
        processed / "predictions" / "validation_2023_predictions.parquet",
        columns=["cell_id", "time_bin", "cell_hour_rate_rolling", "logistic"],
    )
    tree = pd.read_parquet(
        processed / "predictions" / "validation_2023_tree_predictions.parquet",
        columns=["cell_id", "time_bin", "tree_core_plus_location"],
    )
    aligned(validation, baseline, "Baseline/logistic")
    aligned(validation, tree, "Tree")
    validation["baseline"] = baseline["cell_hour_rate_rolling"].to_numpy(dtype=np.float32)
    validation["logistic"] = baseline["logistic"].to_numpy(dtype=np.float32)
    validation["tree"] = tree["tree_core_plus_location"].to_numpy(dtype=np.float32)
    return validation


def aggregate_slices(validation: pd.DataFrame, cell_ids: list[str]) -> dict:
    frame = validation[["cell_id", "month", "hour", "wet", "has_crash", "crash_count", *MODELS]].copy()
    frame["cell_id"] = frame["cell_id"].astype("string")
    aggregation = {
        "has_crash": "mean", "crash_count": "sum",
        "baseline": "mean", "logistic": "mean", "tree": "mean",
    }
    grouped_month = frame.groupby(["wet", "month", "hour", "cell_id"], observed=True).agg(aggregation)
    grouped_all = frame.groupby(["wet", "hour", "cell_id"], observed=True).agg(aggregation)
    grouped_month_any = frame.groupby(["month", "hour", "cell_id"], observed=True).agg(aggregation)
    grouped_all_any = frame.groupby(["hour", "cell_id"], observed=True).agg(aggregation)

    slices: dict[str, dict] = {"all": {}, "wet": {}, "dry": {}}

    def encode(table: pd.DataFrame, period_count: int) -> dict:
        table = table.reindex(cell_ids)
        return {
            "o": np.rint(table["has_crash"].fillna(0).to_numpy() * SCALE).astype(int).tolist(),
            "b": np.rint(table["baseline"].fillna(0).to_numpy() * SCALE).astype(int).tolist(),
            "l": np.rint(table["logistic"].fillna(0).to_numpy() * SCALE).astype(int).tolist(),
            "t": np.rint(table["tree"].fillna(0).to_numpy() * SCALE).astype(int).tolist(),
            "c": table["crash_count"].fillna(0).astype(int).tolist(),
            "n": int(period_count),
        }

    for hour in HOURS:
        any_rows = frame[frame["hour"] == hour]
        slices["all"][f"0-{hour}"] = encode(grouped_all_any.loc[hour], any_rows["month"].count() // len(cell_ids))
        for month in range(1, 13):
            rows = frame[(frame["month"] == month) & (frame["hour"] == hour)]
            slices["all"][f"{month}-{hour}"] = encode(grouped_month_any.loc[(month, hour)], len(rows) // len(cell_ids))

        for wet_value, weather_name in [(True, "wet"), (False, "dry")]:
            weather_rows = any_rows[any_rows["wet"] == wet_value]
            slices[weather_name][f"0-{hour}"] = encode(
                grouped_all.loc[(wet_value, hour)], len(weather_rows) // len(cell_ids)
            )
            for month in range(1, 13):
                rows = frame[(frame["wet"] == wet_value) & (frame["month"] == month) & (frame["hour"] == hour)]
                key = (wet_value, month, hour)
                table = grouped_month.loc[key] if key in grouped_month.index.droplevel("cell_id") else pd.DataFrame(index=cell_ids, columns=aggregation)
                slices[weather_name][f"{month}-{hour}"] = encode(table, len(rows) // len(cell_ids))
    return slices


def quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if len(finite) else 0.0


def build_summary(validation: pd.DataFrame, cell_ids: list[str], slices: dict) -> dict:
    probability = validation[["baseline", "logistic", "tree"]].to_numpy(dtype=np.float32)
    observed_by_cell = validation.groupby("cell_id", observed=True)["has_crash"].mean().to_numpy()
    # The difference map compares aggregated rates, not each binary row. Build
    # its fixed legend from the same monthly/time slices displayed in the UI.
    slice_differences = []
    for item in slices["all"].values():
        observed = np.asarray(item["o"], dtype=float) / SCALE
        predicted = np.asarray(item["t"], dtype=float) / SCALE
        slice_differences.append(np.abs(predicted - observed))
    diff = np.concatenate(slice_differences)
    return {
        "meta": {
            "year": 2023,
            "split": "validation",
            "scale": SCALE,
            "models": MODEL_NAMES,
            "legend": {
                "observed_max": round(quantile(observed_by_cell, .99), 5),
                "predicted_max": round(quantile(probability, .995), 5),
                "difference_abs_max": round(quantile(np.abs(diff), .99), 5),
            },
        },
        "cell_ids": cell_ids,
        "slices": slices,
    }


def percentile_tenths(values: pd.Series, cell_ids: list[str]) -> list[int]:
    ranked = values.reindex(cell_ids).rank(method="average", pct=True).fillna(0)
    return np.rint(ranked.to_numpy() * 1000).astype(int).tolist()


def clipped_ratio(numerator: pd.Series, denominator: float, cell_ids: list[str]) -> list[float]:
    values = numerator.reindex(cell_ids).fillna(0).to_numpy(dtype=float) / max(denominator, 1e-9)
    return np.round(np.clip(values, .05, 2.0), 2).tolist()


def build_detail(root: Path, processed: Path, validation: pd.DataFrame, cell_ids: list[str], grid: dict) -> dict:
    train = pd.read_parquet(
        processed / "model_panel_train_2021_2022.parquet",
        columns=["cell_id", "time_bin", "has_crash", "hour"],
    )
    train["time_bin"] = pd.to_datetime(train["time_bin"])
    history_source = pd.concat(
        [
            train[["cell_id", "time_bin", "has_crash"]],
            validation[["cell_id", "time_bin", "has_crash"]],
        ],
        ignore_index=True,
    )
    history_source["period"] = history_source["time_bin"].dt.to_period("M")
    periods = pd.period_range("2021-01", "2023-12", freq="M")
    monthly = history_source.groupby(["cell_id", "period"], observed=True)["has_crash"].sum().unstack(fill_value=0)
    monthly = monthly.reindex(index=cell_ids, columns=periods, fill_value=0)

    train_cell = train.groupby("cell_id", observed=True)["has_crash"].agg(["sum", "count"])
    global_rate = float(train["has_crash"].mean())
    smoothed = (train_cell["sum"] + 20 * global_rate) / (train_cell["count"] + 20)
    slot_rates = train.groupby(["cell_id", "hour"], observed=True)["has_crash"].mean().unstack()
    city_slot = train.groupby("hour", observed=True)["has_crash"].mean()
    time_ratio = (slot_rates.div(city_slot, axis="columns").max(axis=1)).fillna(1)

    grid_frame = pd.DataFrame([feature["properties"] for feature in grid["features"]]).set_index("cell_id")
    road_density = grid_frame["road_length_m"] / grid_frame["land_fraction"].clip(lower=.05)
    density_ratio = road_density / max(float(road_density[road_density > 0].median()), 1)
    speed_ratio = grid_frame["speed_limit_mph"].fillna(grid_frame["speed_limit_mph"].median()) / max(float(grid_frame["speed_limit_mph"].median()), 1)
    roads = (density_ratio.clip(0, 2) + speed_ratio.clip(0, 2)) / 2

    cell_predictions = validation.groupby("cell_id", observed=True)[list(MODELS)].mean()
    return {
        "cell_ids": cell_ids,
        "months": [str(period) for period in periods],
        "monthly": monthly.astype(int).values.tolist(),
        "percentile": {
            model: percentile_tenths(cell_predictions[model], cell_ids) for model in MODELS
        },
        "context": {
            "history": clipped_ratio(smoothed, global_rate, cell_ids),
            "time": np.round(np.clip(time_ratio.reindex(cell_ids).fillna(1).to_numpy(), .05, 2), 2).tolist(),
            "roads": np.round(np.clip(roads.reindex(cell_ids).fillna(1).to_numpy(), .05, 2), 2).tolist(),
        },
    }


def top_capture(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(score) * fraction)))
    selected = np.argpartition(score, -count)[-count:]
    positives = y.sum()
    return float(y[selected].sum() / positives) if positives else 0.0


def metric_row(model: str, y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "id": model,
        "name": MODEL_NAMES[model],
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "top5_capture": top_capture(y, score, .05),
        "brier_score": float(brier_score_loss(y, score)),
    }


def capture_curve(y: np.ndarray, score: np.ndarray) -> list[list[float]]:
    order = np.argsort(-score)
    cumulative = np.cumsum(y[order])
    total = max(1, int(cumulative[-1]))
    points = [[0.0, 0.0]]
    for fraction in np.linspace(.01, .20, 20):
        index = min(len(y) - 1, max(0, int(np.ceil(len(y) * fraction)) - 1))
        points.append([round(float(fraction), 3), round(float(cumulative[index] / total), 5)])
    return points


def precision_recall_points(y: np.ndarray, score: np.ndarray) -> list[list[float]]:
    order = np.argsort(-score)
    ordered = y[order]
    cumulative = np.cumsum(ordered)
    total = max(1, int(cumulative[-1]))
    indices = np.unique(np.geomspace(1, len(y), 100, dtype=int) - 1)
    points = [[0.0, 1.0]]
    points.extend(
        [round(float(cumulative[index] / total), 5), round(float(cumulative[index] / (index + 1)), 5)]
        for index in indices
    )
    return sorted(points)


def calibration_points(y: np.ndarray, score: np.ndarray) -> list[list[float]]:
    frame = pd.DataFrame({"y": y, "score": score})
    frame["bin"] = pd.qcut(frame["score"], q=10, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(predicted=("score", "mean"), observed=("y", "mean"))
    return [[round(float(row.predicted), 5), round(float(row.observed), 5)] for row in grouped.itertuples()]


def build_metrics(root: Path, validation: pd.DataFrame) -> dict:
    y = validation["has_crash"].to_numpy(dtype=bool)
    scores = {model: validation[model].to_numpy(dtype=np.float32) for model in MODELS}
    importance_path = root / "reports" / "tree_model" / "feature_importance.csv"
    if importance_path.is_file():
        importance = pd.read_csv(importance_path)
    else:
        importance_csv = git_text(
            root,
            "remotes/origin/modeling/tree-comparison:reports/tree_model/feature_importance.csv",
        )
        importance = pd.read_csv(io.StringIO(importance_csv))
    importance = importance[(importance["model"] == "tree_core_plus_location") & (importance["level"] == "group")]
    return {
        "validation": {
            "year": 2023,
            "split": "validation",
            "rows": int(len(validation)),
            "positive_rows": int(y.sum()),
            "positive_rate": float(y.mean()),
        },
        "models": [metric_row(model, y, score) for model, score in scores.items()],
        "curves": {
            "capture": {model: capture_curve(y, score) for model, score in scores.items()},
            "precision_recall": {model: precision_recall_points(y, score) for model, score in scores.items()},
            "calibration": {model: calibration_points(y, score) for model, score in scores.items()},
        },
        "importance": [
            {"name": row.name, "drop": float(row.average_precision_drop)}
            for row in importance.sort_values("average_precision_drop", ascending=False).itertuples()
        ],
    }


def subgroup_rows(frame: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for group, group_frame in frame.groupby(column, observed=True):
        y = group_frame["has_crash"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for model in MODELS:
            score = group_frame[model].to_numpy(dtype=np.float32)
            rows.append(
                {
                    "group": str(group), "model": model, "rows": int(len(group_frame)),
                    "positive_rate": float(y.mean()),
                    "average_precision": float(average_precision_score(y, score)),
                    "top5_capture": top_capture(y, score, .05),
                }
            )
    return rows


def build_subgroups(validation: pd.DataFrame) -> dict:
    frame = validation[["has_crash", "borough", "hour", "month", "wet", *MODELS]].copy()
    frame["time"] = frame["hour"].map({0: "12 AM–6 AM", 6: "6 AM–12 PM", 12: "12 PM–6 PM", 18: "6 PM–12 AM"})
    frame["season"] = frame["month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
    frame["weather"] = np.where(frame["wet"], "Wet days", "Dry days")
    return {
        "borough": subgroup_rows(frame, "borough"),
        "time": subgroup_rows(frame, "time"),
        "season": subgroup_rows(frame, "season"),
        "weather": subgroup_rows(frame, "weather"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    processed = root / "data" / "processed"
    output = (args.output or root / "docs" / "data").resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Exporting compact grid...")
    cell_ids, grid = export_grid(processed, output)
    print("Loading 2023 validation predictions...")
    validation = load_validation(processed)
    if set(validation["cell_id"].astype("string").unique()) != set(cell_ids):
        raise ValueError("Validation cell IDs do not match the grid")
    print("Aggregating map slices...")
    slices = aggregate_slices(validation, cell_ids)
    compact_dump(build_summary(validation, cell_ids, slices), output / "cell_period_summary.json")
    print("Building cell history and context...")
    compact_dump(build_detail(root, processed, validation, cell_ids, grid), output / "cell_detail.json")
    print("Computing validation metrics and curves...")
    compact_dump(build_metrics(root, validation), output / "metrics.json")
    print("Computing subgroup checks...")
    compact_dump(build_subgroups(validation), output / "subgroup_metrics.json")

    print("\nDashboard files:")
    total = 0
    for path in sorted(output.glob("*")):
        size = path.stat().st_size
        total += size
        print(f"  {path.name:32} {size / 1_000_000:6.2f} MB")
    print(f"  {'TOTAL':32} {total / 1_000_000:6.2f} MB")
    if total > 15_000_000:
        raise RuntimeError("Dashboard payload exceeds the 15 MB target")


if __name__ == "__main__":
    main()
