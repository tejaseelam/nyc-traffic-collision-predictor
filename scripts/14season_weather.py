"""Evaluate season and precipitation/trace-day subgroups on 2023 validation.

The top 5% is selected separately inside each subgroup. The weather indicator
comes from the cleaned panel: it is true for measurable precipitation or an
NOAA trace amount. This script reads no 2024 data.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_POSITIVES = 86_527
MODELS = {
    "baseline": "cell_hour_rate_rolling",
    "logistic": "logistic",
    "lightgbm": "tree_core_plus_location",
}


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    seasons = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    }
    data = data.copy()
    data["season"] = data["month"].map(seasons)
    data["weather"] = np.where(
        data["weather_precip_day"].fillna(False),
        "precip_or_trace_days",
        "no_precip_days",
    )
    groups = {
        "season": ["winter", "spring", "summer", "fall"],
        "weather": ["precip_or_trace_days", "no_precip_days"],
    }
    rows = []
    for experiment, names in groups.items():
        for name in names:
            group = data.loc[data[experiment] == name]
            y_true = group["has_crash"].to_numpy(dtype=bool)
            if not y_true.any():
                raise ValueError(f"Subgroup {name} has no positive rows")
            for model, column in MODELS.items():
                score = group[column].to_numpy(dtype=np.float64)
                top_count = max(1, round(len(group) * 0.05))
                top = np.argsort(-score, kind="stable")[:top_count]
                note = (
                    "Meteorological seasons; top 5% ranked within season"
                    if experiment == "season"
                    else "Panel weather_precip_day includes measurable precipitation and NOAA trace amounts; top 5% ranked within weather group"
                )
                rows.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "group": name,
                        "average_precision": average_precision_score(y_true, score),
                        "roc_auc": roc_auc_score(y_true, score),
                        "top5_capture": y_true[top].sum() / y_true.sum(),
                        "top5_scope": "within_group",
                        "n_rows": len(group),
                        "n_positive": int(y_true.sum()),
                        "split": "2023_validation",
                        "status": "complete",
                        "notes": note,
                    }
                )
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, output: Path) -> None:
    specifications = [
        ("season", "Seasonal model performance", ["winter", "spring", "summer", "fall"]),
        ("weather", "Precipitation-day model performance", ["precip_or_trace_days", "no_precip_days"]),
    ]
    for experiment, title, groups in specifications:
        subset = results.loc[results["experiment"] == experiment]
        figure, axes = plt.subplots(1, 3, figsize=(11, 4))
        for axis, metric, label in zip(
            axes,
            ["average_precision", "roc_auc", "top5_capture"],
            ["Average precision", "ROC AUC", "Top-5% capture"],
        ):
            table = subset.pivot(index="group", columns="model", values=metric).loc[
                groups, ["baseline", "logistic", "lightgbm"]
            ]
            table.index = [group.replace("_", " ").title() for group in table.index]
            table.plot.bar(
                ax=axis, color=["#8f97a3", "#4384b6", "#24936e"], width=0.75, legend=False
            )
            axis.set_title(label)
            axis.set_xlabel("")
            axis.set_ylim(0, table.to_numpy().max() * 1.15)
            axis.tick_params(axis="x", rotation=0)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        handles, _ = axes[0].get_legend_handles_labels()
        figure.suptitle(title + ": 2023 validation")
        figure.legend(handles, ["Baseline", "Logistic", "LightGBM"], loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=[0, 0.09, 1, 0.9])
        figure.savefig(output / f"{experiment}_performance.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def load_data(root: Path) -> pd.DataFrame:
    processed = root / "data" / "processed"
    predictions = processed / "predictions"
    keys = ["cell_id", "time_bin"]
    data = pd.read_parquet(
        processed / "model_panel_validation_2023.parquet",
        columns=keys + ["has_crash", "month", "weather_precip_day"],
    )
    baseline = pd.read_parquet(
        predictions / "validation_2023_predictions.parquet",
        columns=keys + ["has_crash", "cell_hour_rate_rolling", "logistic"],
    )
    tree = pd.read_parquet(
        predictions / "validation_2023_tree_predictions.parquet",
        columns=keys + ["has_crash", "tree_core_plus_location"],
    )
    for frame in (baseline, tree):
        for column in keys + ["has_crash"]:
            if not np.array_equal(data[column].to_numpy(), frame[column].to_numpy()):
                raise ValueError("Prediction rows do not match the validation panel")
    if not pd.to_datetime(data["time_bin"]).dt.year.eq(2023).all():
        raise ValueError("Expected only 2023 validation data")
    if int(data["has_crash"].sum()) != EXPECTED_POSITIVES:
        raise ValueError("This is not the canonical 86,527-positive validation panel")
    data["cell_hour_rate_rolling"] = baseline["cell_hour_rate_rolling"]
    data["logistic"] = baseline["logistic"]
    data["tree_core_plus_location"] = tree["tree_core_plus_location"]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory or args.root / "reports" / "season_weather"
    output.mkdir(parents=True, exist_ok=True)
    results = evaluate(load_data(args.root))
    results.to_csv(output / "season_weather.csv", index=False, float_format="%.6f")
    plot_results(results, output)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
