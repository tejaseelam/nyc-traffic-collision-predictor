"""Plot the 2023 validation results from 8baseline_logreg.py for the report.

Figures written to reports/baseline_logreg/figures/:

* capture_curve.png:    share of crashes caught when flagging the riskiest
                         share of cell-periods (hotspot view)
* precision_recall.png: precision-recall curves for every model
* calibration.png:      predicted vs observed crash rate for the logistic model
* coefficients.png:     the largest standardized logistic coefficients
* risk_map.png:         predicted vs observed 2023 crash rate per grid cell
                         (only when the grid file is available)
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = "validation_2023_predictions.parquet"
GRID_FILE = "nyc_grid_500m.parquet"
TARGET = "has_crash"
METRIC_CRS = "EPSG:32618"

# Fixed slot order: the logistic model always takes the first color.
MODELS = [
    ("logistic", "Logistic regression", "#2a78d6"),
    ("cell_hour_rate_rolling", "Cell + time-of-day rate, updated", "#eb6834"),
    ("cell_hour_rate", "Cell + time-of-day rate, 2021-22", "#1baf7a"),
    ("cell_rate", "Cell rate, 2021-22", "#eda100"),
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
RAISES_RISK = "#e34948"
LOWERS_RISK = "#2a78d6"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
CURVE_POINTS = 500


def style_axes(axes, title, subtitle=None):
    axes.set_facecolor(SURFACE)
    axes.grid(color=GRIDLINE, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    axes.xaxis.label.set_color(INK_SECONDARY)
    axes.yaxis.label.set_color(INK_SECONDARY)
    axes.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    if subtitle:
        axes.text(
            0, 1.02, subtitle, transform=axes.transAxes,
            color=INK_SECONDARY, fontsize=9, va="bottom",
        )


def new_figure(width=7.0, height=4.5):
    figure, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    return figure, axes


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def available_models(predictions):
    return [model for model in MODELS if model[0] in predictions]


def capture_curve(y_true, probability, max_fraction):
    """Share of crashes in the top `fraction` of rows, ranked by probability."""
    order = np.argsort(-probability, kind="stable")
    captured = np.cumsum(y_true[order]) / max(int(y_true.sum()), 1)
    flagged = np.arange(1, len(order) + 1) / len(order)
    stop = max(1, int(np.ceil(max_fraction * len(order))))
    keep = np.unique(np.linspace(0, stop - 1, CURVE_POINTS).astype(int))
    return np.concatenate([[0], flagged[keep]]), np.concatenate([[0], captured[keep]])


def plot_capture(predictions, path, max_fraction=0.2):
    y_true = predictions[TARGET].to_numpy(dtype=bool)
    figure, axes = new_figure()
    for column, label, color in available_models(predictions):
        flagged, captured = capture_curve(y_true, predictions[column].to_numpy(), max_fraction)
        at_five = np.interp(0.05, flagged, captured)
        axes.plot(
            flagged * 100, captured * 100, color=color, linewidth=2,
            label=f"{label}: {at_five:.1%} at top 5%",
        )
        axes.plot(
            [5], [at_five * 100], "o", color=color, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
        )
    axes.plot(
        [0, max_fraction * 100], [0, max_fraction * 100],
        color=INK_MUTED, linewidth=1, linestyle="--", label="Random",
    )
    axes.axvline(5, color=BASELINE, linewidth=0.8)
    axes.set_xlim(0, max_fraction * 100)
    axes.set_ylim(0, None)
    axes.set_xlabel("Riskiest cell-periods flagged (%)")
    axes.set_ylabel("Crashes captured (%)")
    style_axes(
        axes,
        "Share of 2023 crashes caught by flagging the riskiest areas",
        "Each row is one 500 m cell in one six-hour period; dots mark the top 5%",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    return save(figure, path)


def plot_precision_recall(predictions, path):
    y_true = predictions[TARGET].to_numpy(dtype=bool)
    figure, axes = new_figure()
    for column, label, color in available_models(predictions):
        precision, recall, _ = precision_recall_curve(y_true, predictions[column].to_numpy())
        # Sample evenly along recall. The first half percent of recall is only a
        # few hundred crashes, so its precision is noise and is left out.
        recall, precision = recall[::-1], precision[::-1]
        targets = np.linspace(0.005, 1, CURVE_POINTS)
        keep = np.unique(np.searchsorted(recall, targets).clip(0, len(recall) - 1))
        axes.plot(recall[keep], precision[keep], color=color, linewidth=2, label=label)
    axes.axhline(
        y_true.mean(), color=INK_MUTED, linewidth=1, linestyle="--",
        label=f"Crash rate ({y_true.mean():.1%})",
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, None)
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    style_axes(axes, "Precision-recall on 2023 validation", "Higher and further right is better")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
    return save(figure, path)


def calibration_bins(y_true, probability, bins):
    frame = pd.DataFrame({"predicted": probability, "observed": y_true.astype(float)})
    frame["bin"] = pd.qcut(frame["predicted"], bins, labels=False, duplicates="drop")
    return frame.groupby("bin").agg(predicted=("predicted", "mean"), observed=("observed", "mean"))


def plot_calibration(predictions, path, bins=20):
    if "logistic" not in predictions:
        return None
    y_true = predictions[TARGET].to_numpy(dtype=bool)
    table = calibration_bins(y_true, predictions["logistic"].to_numpy(), bins)
    low = max(min(table["predicted"].min(), table["observed"].min()), 1e-5) * 0.8
    high = max(table["predicted"].max(), table["observed"].max()) * 1.2

    figure, axes = new_figure(5.5, 5.0)
    axes.plot([low, high], [low, high], color=INK_MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    axes.plot(
        table["predicted"], table["observed"].clip(lower=low),
        color=MODELS[0][2], linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label="Logistic regression",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed crash rate")
    style_axes(axes, "Calibration on 2023 validation", f"{len(table)} equal-size groups of cell-periods")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    return save(figure, path)


DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURE_NAMES = {
    "history_cell_logit": "Cell crash history",
    "history_cell_hour_logit": "Cell crash history, same time of day",
    "history_recent_logit": "Cell crash history, last 28 days",
    "log_speed_road_length_m": "Road length (log)",
    "log_speed_road_segments": "Road segments (log)",
    "speed_limit_weighted_avg_mph": "Average speed limit",
    "speed_limit_coverage": "Speed-limit coverage",
    "land_fraction": "Land share of cell",
    "weather_temp_midpoint_c": "Temperature",
    "weather_precip_mm": "Precipitation",
    "weather_snowfall_mm": "Snowfall",
    "weather_snow_depth_mm": "Snow depth",
    "weather_wind_mps": "Wind speed",
    "weather_precip_day": "Rainy day",
    "weather_snow_day": "Snowy day",
    "weather_any_missing": "Weather missing",
}


def readable_feature(name):
    name = name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        return f"{readable_feature(name[len('missingindicator_'):])} missing"
    prefix, _, value = name.rpartition("_")
    if prefix == "hour":
        return f"{int(value):02d}:00-{int(value) + 6:02d}:00"
    if prefix == "day_of_week":
        return DAYS[int(value)]
    if prefix == "month":
        return MONTHS[int(value) - 1]
    if prefix == "borough":
        return value.title()
    return FEATURE_NAMES.get(name, name.replace("_", " "))


def plot_coefficients(coefficients, path, top=15):
    table = coefficients.sort_values("absolute_coefficient", ascending=False).head(top)
    table = table.iloc[::-1]
    colors = [RAISES_RISK if value > 0 else LOWERS_RISK for value in table["coefficient"]]

    figure, axes = new_figure(7.0, 0.32 * len(table) + 1.6)
    axes.barh(
        [readable_feature(name) for name in table["feature"]],
        table["coefficient"], color=colors, height=0.7,
        edgecolor=SURFACE, linewidth=1,
    )
    axes.axvline(0, color=BASELINE, linewidth=1)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("Change in log-odds per standard deviation (or category)")
    style_axes(
        axes,
        f"Top {len(table)} logistic regression coefficients",
        "Red raises risk, blue lowers it. Reference: 00:00-06:00, Mon, Jan, Bronx",
    )
    axes.grid(axis="y", visible=False)
    return save(figure, path)


def plot_risk_map(predictions, grid, path):
    if "logistic" not in predictions:
        return None
    by_cell = (
        predictions.groupby("cell_id", observed=True)
        .agg(predicted=("logistic", "mean"), observed=(TARGET, "mean"))
        .reset_index()
    )
    by_cell["cell_id"] = by_cell["cell_id"].astype(str)
    cells = grid[["cell_id", "geometry"]].copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.merge(by_cell, on="cell_id", how="left").to_crs(METRIC_CRS)
    high = float(np.nanpercentile(cells[["predicted", "observed"]].to_numpy(), 99))

    min_x, min_y, max_x, max_y = cells.total_bounds
    panel_height = 4.6 * (max_y - min_y) / (max_x - min_x)
    figure, panels = plt.subplots(
        1, 2, figsize=(10.5, panel_height + 1.0), facecolor=SURFACE, layout="constrained"
    )
    for axes, column, title in [
        (panels[0], "predicted", "Predicted (logistic regression)"),
        (panels[1], "observed", "Observed"),
    ]:
        cells.plot(
            column=column, ax=axes, cmap=SEQUENTIAL, vmin=0, vmax=high,
            linewidth=0, missing_kwds={"color": GRIDLINE},
        )
        axes.set_axis_off()
        axes.set_title(title, loc="left", color=INK_SECONDARY, fontsize=10)
    colorbar = figure.colorbar(
        panels[1].collections[0], ax=panels, shrink=0.8,
        format=lambda value, _: f"{value:.0%}",
    )
    colorbar.set_label("Share of six-hour periods with a crash", color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    colorbar.outline.set_visible(False)
    figure.suptitle(
        "2023 crash risk by 500 m cell", x=0.01, ha="left",
        color=INK, fontsize=12, fontweight="bold",
    )
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_all(predictions, coefficients, grid, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_capture(predictions, output_directory / "capture_curve.png"),
        plot_precision_recall(predictions, output_directory / "precision_recall.png"),
        plot_calibration(predictions, output_directory / "calibration.png"),
    ]
    if coefficients is not None:
        paths.append(plot_coefficients(coefficients, output_directory / "coefficients.png"))
    if grid is not None:
        paths.append(plot_risk_map(predictions, grid, output_directory / "risk_map.png"))
    return [path for path in paths if path is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    report = args.root / "reports" / "baseline_logreg"
    output_directory = args.output_directory or report / "figures"
    predictions_path = processed / "predictions" / PREDICTIONS_FILE
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\n"
            "Run scripts/8baseline_logreg.py first."
        )

    predictions = pd.read_parquet(predictions_path)
    coefficients_path = report / "coefficients.csv"
    coefficients = pd.read_csv(coefficients_path) if coefficients_path.is_file() else None
    grid = None
    grid_path = processed / GRID_FILE
    if grid_path.is_file():
        import geopandas as gpd

        grid = gpd.read_parquet(grid_path)
    else:
        print(f"Skipping the risk map: {grid_path} not found")

    for path in plot_all(predictions, coefficients, grid, output_directory):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
