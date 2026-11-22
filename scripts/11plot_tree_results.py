"""Plot the 2023 validation results from 10tree_model.py for the report.

Figures written to reports/tree_model/figures/:

* feature_importance.png:        average-precision lost when each group of
                                  features is shuffled, for every model
* feature_importance_detail.png: the same for single features of the tree with
                                  grid location
* capture_curve.png:             share of crashes caught when flagging the
                                  riskiest cell-periods, tree vs logistic
* precision_recall.png:          precision-recall curves, tree vs logistic
* calibration.png:               predicted vs observed crash rate

The logistic regression and baseline appear when 8baseline_logreg.py has been
run and its predictions are saved.
"""

from pathlib import Path
import argparse
import importlib.util

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_logreg_results", Path(__file__).resolve().parent / "9plot_logreg_results.py"
)
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)

TARGET = PLOTS.TARGET
TREE_PREDICTIONS_FILE = "validation_2023_tree_predictions.parquet"
LOGREG_PREDICTIONS_FILE = PLOTS.PREDICTIONS_FILE

# Colors follow the model: the logistic and baseline keep the colors used in
# the logistic figures, and the trees take the next categorical slots.
MODELS = [
    ("tree_core", "Boosted trees, same features", "#e87ba4"),
    ("tree_core_plus_location", "Boosted trees + grid location", "#008300"),
    ("logistic", "Logistic regression", "#2a78d6"),
    ("cell_hour_rate_rolling", "Cell + time-of-day rate, updated", "#eb6834"),
]
CURVE_MODELS = ["tree_core", "logistic", "cell_hour_rate_rolling"]
FEATURE_NAMES = {
    **PLOTS.FEATURE_NAMES,
    "hour": "Time of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "borough": "Borough",
    "grid_x": "Grid position, west-east",
    "grid_y": "Grid position, south-north",
}


def model_style(name):
    return next(model for model in MODELS if model[0] == name)


def plot_group_importance(importance, path):
    table = importance[importance["level"] == "group"]
    models = [model for model in MODELS if model[0] in set(table["model"])]
    # Groups sorted by the first tree's importance, largest at the top.
    order = (
        table[table["model"] == models[0][0]]
        .sort_values("relative_drop")["name"].tolist()
    )
    height = 0.8 / len(models)

    figure, axes = PLOTS.new_figure(7.0, 0.55 * len(order) * len(models) / 2 + 2.2)
    positions = np.arange(len(order))
    for index, (column, label, color) in enumerate(models):
        values = (
            table[table["model"] == column].set_index("name")["relative_drop"]
            .reindex(order)
        )
        offset = (len(models) - 1 - index - (len(models) - 1) / 2) * height
        axes.barh(
            positions + offset, values * 100, height=height, color=color,
            edgecolor=PLOTS.SURFACE, linewidth=2, label=label,
        )
        for position, value in zip(positions + offset, values):
            if pd.notna(value):
                axes.text(
                    max(value, 0) * 100 + 0.6, position, f"{max(value, 0):.0%}",
                    va="center", fontsize=8, color=PLOTS.INK_SECONDARY,
                )
    axes.set_yticks(positions, order)
    axes.axvline(0, color=PLOTS.BASELINE, linewidth=1)
    axes.set_xlim(min(0, table["relative_drop"].min() * 100), table["relative_drop"].max() * 115)
    axes.set_xlabel("Average precision lost when the group is shuffled (%)")
    PLOTS.style_axes(
        axes,
        "Which kinds of information each model relies on",
        "2023 validation; each group's columns are shuffled together",
    )
    axes.grid(axis="y", visible=False)
    # Legend entries follow the bars, top to bottom.
    axes.legend(frameon=False, fontsize=9, labelcolor=PLOTS.INK_SECONDARY, loc="lower right")
    return PLOTS.save(figure, path)


def plot_feature_importance(importance, model, path, top=15):
    table = importance[(importance["level"] == "feature") & (importance["model"] == model)]
    if table.empty:
        return None
    table = table.sort_values("relative_drop", ascending=False).head(top).iloc[::-1]
    _, label, color = model_style(model)

    figure, axes = PLOTS.new_figure(7.0, 0.32 * len(table) + 1.6)
    axes.barh(
        [FEATURE_NAMES.get(name, name) for name in table["name"]],
        table["relative_drop"] * 100, color=color, height=0.7,
        edgecolor=PLOTS.SURFACE, linewidth=1,
    )
    axes.axvline(0, color=PLOTS.BASELINE, linewidth=1)
    axes.set_xlabel("Average precision lost when the feature is shuffled (%)")
    PLOTS.style_axes(
        axes,
        f"Top {len(table)} single features",
        label,
    )
    axes.grid(axis="y", visible=False)
    return PLOTS.save(figure, path)


def plot_calibration(predictions, path, bins=20):
    models = [model for model in MODELS if model[0] in predictions and model[0] != CURVE_MODELS[-1]]
    y_true = predictions[TARGET].to_numpy(dtype=bool)
    tables = [
        (label, color, PLOTS.calibration_bins(y_true, predictions[column].to_numpy(), bins))
        for column, label, color in models
    ]
    values = pd.concat([table for _, _, table in tables])
    low = max(values.min().min(), 1e-5) * 0.8
    high = values.max().max() * 1.2

    figure, axes = PLOTS.new_figure(5.5, 5.0)
    axes.plot(
        [low, high], [low, high], color=PLOTS.INK_MUTED, linewidth=1,
        linestyle="--", label="Perfect calibration",
    )
    for label, color, table in tables:
        axes.plot(
            table["predicted"], table["observed"].clip(lower=low),
            color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=PLOTS.SURFACE, markeredgewidth=1.5, label=label,
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed crash rate")
    PLOTS.style_axes(
        axes, "Calibration on 2023 validation", f"{bins} equal-size groups of cell-periods"
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=PLOTS.INK_SECONDARY, loc="upper left")
    return PLOTS.save(figure, path)


def plot_all(predictions, importance, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    # The shared curve plots draw whichever of PLOTS.MODELS are present.
    PLOTS.MODELS = [model_style(name) for name in CURVE_MODELS]
    paths = [
        plot_group_importance(importance, output_directory / "feature_importance.png"),
        plot_feature_importance(
            importance, "tree_core_plus_location",
            output_directory / "feature_importance_detail.png",
        ),
        PLOTS.plot_capture(predictions, output_directory / "capture_curve.png"),
        PLOTS.plot_precision_recall(predictions, output_directory / "precision_recall.png"),
        plot_calibration(predictions, output_directory / "calibration.png"),
    ]
    return [path for path in paths if path is not None]


def load_predictions(processed):
    tree_path = processed / "predictions" / TREE_PREDICTIONS_FILE
    if not tree_path.is_file():
        raise FileNotFoundError(
            f"Predictions not found: {tree_path}\nRun scripts/10tree_model.py first."
        )
    predictions = pd.read_parquet(tree_path)
    logreg_path = processed / "predictions" / LOGREG_PREDICTIONS_FILE
    if logreg_path.is_file():
        keys = ["cell_id", "time_bin"]
        logreg = pd.read_parquet(
            logreg_path, columns=keys + ["logistic", "cell_hour_rate_rolling"]
        )
        for frame in [predictions, logreg]:
            frame["cell_id"] = frame["cell_id"].astype(str)
        predictions = predictions.merge(logreg, on=keys, how="left", validate="one_to_one")
    else:
        print(f"Plotting the tree only: {logreg_path} not found")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    report = args.root / "reports" / "tree_model"
    output_directory = args.output_directory or report / "figures"
    importance_path = report / "feature_importance.csv"
    if not importance_path.is_file():
        raise FileNotFoundError(
            f"Feature importance not found: {importance_path}\n"
            "Run scripts/10tree_model.py first."
        )

    predictions = load_predictions(processed)
    importance = pd.read_csv(importance_path)
    for path in plot_all(predictions, importance, output_directory):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
