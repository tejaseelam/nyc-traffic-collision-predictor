"""Compare the rolling baseline, logistic regression, and LightGBM on 2023.

Reads the saved 2023 validation predictions of scripts/8baseline_logreg.py and
scripts/10tree_model.py, checks that every model was scored on the same rows,
and measures all of them with one set of definitions:

* average precision (main metric), ROC AUC, and top-5% capture, the share of
  cell-periods with a crash that fall in the riskiest 5% of all cell-periods;
* calibration: Brier score, log loss, mean predicted vs observed rate, and the
  calibration intercept and slope from refitting the outcome on logit(p).

To tell whether the differences between models are larger than chance, 2023 is
resampled by whole days (a block bootstrap). Crashes on the same day share the
weather and citywide conditions, so resampling single rows would overstate how
certain the differences are.

The 2024 test file is never opened.
"""

from pathlib import Path
import argparse
import importlib.util

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("baseline_logreg", ROOT / "scripts" / "8baseline_logreg.py")
LOGREG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGREG)
PLOTS_SPEC = importlib.util.spec_from_file_location(
    "plot_logreg_results", ROOT / "scripts" / "9plot_logreg_results.py"
)
PLOTS = importlib.util.module_from_spec(PLOTS_SPEC)
PLOTS_SPEC.loader.exec_module(PLOTS)

TARGET = LOGREG.TARGET
LOGREG_PREDICTIONS = "validation_2023_predictions.parquet"
TREE_PREDICTIONS = "validation_2023_tree_predictions.parquet"
KEYS = ["cell_id", "time_bin"]
VALIDATION_YEAR = 2023
TOP_FRACTION = 0.05

# (name in the shared CSV, prediction column, group label, plot label, color)
MODELS = [
    ("baseline", "cell_hour_rate_rolling", "all_2023", "Rolling cell + time-of-day rate", "#eb6834"),
    ("logistic", "logistic", "all_2023", "Logistic regression", "#2a78d6"),
    ("lightgbm", "tree_core_plus_location", "all_2023", "LightGBM + location", "#1baf7a"),
    ("lightgbm", "tree_core", "all_2023_core_features", "LightGBM, logistic features only", "#eda100"),
]
BENCHMARK = "cell_hour_rate_rolling"


def load_predictions(processed):
    """Join both prediction files and check they cover the same 2023 rows."""
    logreg = pd.read_parquet(
        processed / "predictions" / LOGREG_PREDICTIONS, columns=KEYS + [TARGET, "cell_hour_rate_rolling", "logistic"]
    )
    tree = pd.read_parquet(
        processed / "predictions" / TREE_PREDICTIONS, columns=KEYS + [TARGET, "tree_core", "tree_core_plus_location"]
    )
    return align(logreg, tree)


def align(logreg, tree):
    for frame in (logreg, tree):
        frame["cell_id"] = frame["cell_id"].astype(str)
    if len(logreg) != len(tree):
        raise ValueError(f"Prediction files differ in length: {len(logreg):,} vs {len(tree):,}")
    if logreg.duplicated(KEYS).any() or tree.duplicated(KEYS).any():
        raise ValueError("A prediction file has duplicate cell-periods")
    merged = logreg.merge(tree, on=KEYS, how="inner", suffixes=("", "_tree"), validate="one_to_one")
    if len(merged) != len(logreg):
        raise ValueError("The prediction files do not cover the same cell-periods")
    if not (merged[TARGET].astype(bool) == merged[f"{TARGET}_tree"].astype(bool)).all():
        raise ValueError("The prediction files disagree on has_crash")
    years = pd.to_datetime(merged["time_bin"]).dt.year.unique()
    if list(years) != [VALIDATION_YEAR]:
        raise ValueError(f"Expected only {VALIDATION_YEAR} validation rows, found years {sorted(years)}")
    columns = [column for _, column, *_ in MODELS]
    if merged[columns].isna().any().any():
        raise ValueError("Some predictions are missing")
    if ((merged[columns] <= 0) | (merged[columns] >= 1)).any().any():
        raise ValueError("Predictions must be probabilities strictly between 0 and 1")
    merged = merged.drop(columns=f"{TARGET}_tree").sort_values(KEYS, ignore_index=True)
    merged[TARGET] = merged[TARGET].astype(bool)
    return merged


def calibration_line(y_true, probability):
    """Intercept and slope of a logistic fit of the outcome on logit(p).

    A perfectly calibrated model has intercept 0 and slope 1. A slope above 1
    means predictions are too timid, below 1 too extreme.
    """
    fit = LogisticRegression(penalty=None, max_iter=1000)
    fit.fit(logit(probability).reshape(-1, 1), y_true)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def model_metrics(y_true, probability):
    shared = LOGREG.evaluate(y_true, probability, top_fraction=TOP_FRACTION)
    intercept, slope = calibration_line(y_true, probability)
    return {
        "n_rows": shared["rows"],
        "n_positive": int(y_true.sum()),
        "average_precision": shared["average_precision"],
        "roc_auc": shared["roc_auc"],
        "top5_capture": shared["recall_at_top_fraction"],
        "top5_precision": shared["precision_at_top_fraction"],
        "brier_score": shared["brier_score"],
        "log_loss": shared["log_loss"],
        "mean_predicted": shared["mean_predicted_probability"],
        "observed_rate": shared["positive_rate"],
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def ranked(probability, y_true):
    """Rows sorted from riskiest to safest, with the end of each run of tied scores."""
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    group_end = np.append(sorted_probability[1:] != sorted_probability[:-1], True)
    return order, group_end


def weighted_scores(y_sorted, w_sorted, group_end, top_fraction=TOP_FRACTION):
    """Average precision and top-fraction capture when each row counts `w` times.

    With all weights 1 these equal sklearn's average_precision_score and the
    top-fraction capture used everywhere else, so a bootstrap sample can be
    scored from one sort of the full year.
    """
    hits = np.cumsum(w_sorted * y_sorted)
    flagged = np.cumsum(w_sorted)
    hits_at, flagged_at = hits[group_end], flagged[group_end]
    positives = hits[-1]
    recall = hits_at / positives
    # Rows from days left out of a bootstrap sample have weight 0, so the
    # riskiest rows can have nothing flagged yet; recall does not move there.
    precision = np.divide(hits_at, flagged_at, out=np.zeros_like(hits_at), where=flagged_at > 0)
    average_precision = float(np.sum(np.diff(recall, prepend=0) * precision))
    top = np.searchsorted(flagged, top_fraction * flagged[-1], side="left")
    capture = float(hits[min(top, len(hits) - 1)] / positives)
    return average_precision, capture


def bootstrap_differences(frame, repeats, seed):
    """Day-block bootstrap of each model's AP and top-5% capture minus the benchmark's."""
    y_true = frame[TARGET].to_numpy(dtype=np.float64)
    day = pd.to_datetime(frame["time_bin"]).dt.normalize()
    day_codes, days = pd.factorize(day)
    rankings = {
        column: ranked(frame[column].to_numpy(dtype=np.float64), y_true)
        for _, column, *_ in MODELS
    }
    random = np.random.default_rng(seed)
    draws = []
    for repeat in range(repeats):
        day_weights = np.bincount(random.integers(0, len(days), len(days)), minlength=len(days))
        row_weights = day_weights[day_codes].astype(np.float64)
        scores = {}
        for column, (order, group_end) in rankings.items():
            scores[column] = weighted_scores(y_true[order], row_weights[order], group_end)
        for _, column, *_ in MODELS:
            draws.append({
                "repeat": repeat,
                "column": column,
                "average_precision": scores[column][0],
                "top5_capture": scores[column][1],
                "average_precision_vs_baseline": scores[column][0] - scores[BENCHMARK][0],
                "top5_capture_vs_baseline": scores[column][1] - scores[BENCHMARK][1],
                "average_precision_vs_logistic": scores[column][0] - scores["logistic"][0],
                "top5_capture_vs_logistic": scores[column][1] - scores["logistic"][1],
            })
    return pd.DataFrame(draws)


def summarize_bootstrap(draws):
    rows = []
    for _, column, group, label, _ in MODELS:
        model_draws = draws[draws["column"] == column]
        for measure in [
            "average_precision", "top5_capture",
            "average_precision_vs_baseline", "top5_capture_vs_baseline",
            "average_precision_vs_logistic", "top5_capture_vs_logistic",
        ]:
            values = model_draws[measure]
            rows.append({
                "prediction_column": column,
                "label": label,
                "measure": measure,
                "low_95": values.quantile(0.025),
                "high_95": values.quantile(0.975),
                "share_above_zero": float((values > 0).mean()),
            })
    return pd.DataFrame(rows)


def shared_csv_rows(metrics, intervals, repeats):
    """Rows in the format of reports/evaluation/evaluation_results_template.csv."""
    rows = []
    for name, column, group, label, _ in MODELS:
        result = metrics[column]

        def interval(measure):
            row = intervals[(intervals["prediction_column"] == column) & (intervals["measure"] == measure)].iloc[0]
            return row["low_95"], row["high_95"]

        ap_low, ap_high = interval("average_precision")
        notes = [
            f"{label} ({column})",
            f"AP 95% CI {ap_low:.4f}-{ap_high:.4f} ({repeats} day-block bootstrap samples)",
            f"Brier {result['brier_score']:.5f}",
            f"log loss {result['log_loss']:.4f}",
            f"mean predicted {result['mean_predicted']:.4f} vs observed {result['observed_rate']:.4f}",
            f"calibration slope {result['calibration_slope']:.2f}",
        ]
        if column != BENCHMARK:
            low, high = interval("average_precision_vs_baseline")
            notes.append(f"AP minus rolling baseline 95% CI {low:+.4f} to {high:+.4f}")
        rows.append({
            "experiment": "model_comparison",
            "model": name,
            "group": group,
            "average_precision": round(result["average_precision"], 6),
            "roc_auc": round(result["roc_auc"], 6),
            "top5_capture": round(result["top5_capture"], 6),
            "top5_scope": "global",
            "n_rows": result["n_rows"],
            "n_positive": result["n_positive"],
            "split": "2023_validation",
            "status": "complete",
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows)


def plot_differences(intervals, metrics, path):
    """Each model's AP and top-5% capture minus the rolling baseline, with 95% intervals."""
    others = [model for model in MODELS if model[1] != BENCHMARK]
    figure, axes_pair = plt.subplots(1, 2, figsize=(9.0, 3.6), facecolor=PLOTS.SURFACE)
    panels = [
        ("average_precision", "Average precision", 1.0, "{:+.4f}"),
        ("top5_capture", "Top-5% capture (percentage points)", 100.0, "{:+.1f}"),
    ]
    for axes, (measure, title, scale, label_format) in zip(axes_pair, panels):
        for position, (_, column, _, label, color) in enumerate(reversed(others)):
            row = intervals[
                (intervals["prediction_column"] == column)
                & (intervals["measure"] == f"{measure}_vs_baseline")
            ].iloc[0]
            point = (metrics[column][measure] - metrics[BENCHMARK][measure]) * scale
            axes.plot([row["low_95"] * scale, row["high_95"] * scale], [position, position],
                      color=color, linewidth=3, solid_capstyle="round", alpha=0.45)
            axes.plot([point], [position], "o", color=color, markersize=8,
                      markeredgecolor=PLOTS.SURFACE, markeredgewidth=1.5)
            axes.annotate(label_format.format(point), (point, position), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=9, color=PLOTS.INK_SECONDARY)
        axes.axvline(0, color=PLOTS.INK_MUTED, linewidth=1, linestyle="--")
        axes.set_yticks(range(len(others)))
        axes.set_yticklabels([label for *_, label, _ in reversed(others)])
        axes.set_ylim(-0.6, len(others) - 0.4)
        PLOTS.style_axes(axes, title)
        axes.set_title(title, loc="left", color=PLOTS.INK, fontsize=10.5, fontweight="bold", pad=8)
        axes.grid(axis="y", visible=False)
    axes_pair[1].set_yticklabels([])
    figure.suptitle(
        "Gain over the rolling cell + time-of-day baseline, 2023 validation",
        x=0.01, ha="left", fontsize=12, fontweight="bold", color=PLOTS.INK,
    )
    figure.text(
        0.01, 0.875, "Dot: difference on all of 2023. Bar: 95% interval from resampling whole days. "
        "Dashed line: no gain.", fontsize=9, color=PLOTS.INK_SECONDARY,
    )
    figure.subplots_adjust(left=0.25, right=0.98, top=0.76, bottom=0.12, wspace=0.08)
    figure.savefig(path, dpi=200, facecolor=PLOTS.SURFACE)
    plt.close(figure)
    return path


def plot_capture(frame, path, max_fraction=0.2):
    y_true = frame[TARGET].to_numpy(dtype=bool)
    figure, axes = PLOTS.new_figure()
    for _, column, _, label, color in MODELS[:3]:
        flagged, captured = PLOTS.capture_curve(y_true, frame[column].to_numpy(), max_fraction)
        at_five = np.interp(TOP_FRACTION, flagged, captured)
        axes.plot(flagged * 100, captured * 100, color=color, linewidth=2,
                  label=f"{label}: {at_five:.1%} at top 5%")
        axes.plot([5], [at_five * 100], "o", color=color, markersize=6,
                  markeredgecolor=PLOTS.SURFACE, markeredgewidth=1.5)
    axes.plot([0, max_fraction * 100], [0, max_fraction * 100], color=PLOTS.INK_MUTED,
              linewidth=1, linestyle="--", label="Random")
    axes.axvline(5, color=PLOTS.BASELINE, linewidth=0.8)
    axes.set_xlim(0, max_fraction * 100)
    axes.set_ylim(0, None)
    axes.set_xlabel("Riskiest cell-periods flagged (%)")
    axes.set_ylabel("Cell-periods with a crash captured (%)")
    PLOTS.style_axes(axes, "Share of 2023 crash cell-periods caught by each model",
                     "Each row is one 500 m cell in one six-hour period; dots mark the top 5%")
    axes.legend(frameon=False, fontsize=9, labelcolor=PLOTS.INK_SECONDARY, loc="upper left")
    return PLOTS.save(figure, path)


def plot_calibration(frame, path, bins=20):
    y_true = frame[TARGET].to_numpy(dtype=bool)
    tables = {column: PLOTS.calibration_bins(y_true, frame[column].to_numpy(), bins)
              for _, column, *_ in MODELS[:3]}
    values = pd.concat([pd.concat([table["predicted"], table["observed"]]) for table in tables.values()])
    low = max(values.min(), 1e-5) * 0.8
    high = values.max() * 1.2
    figure, axes = PLOTS.new_figure(5.5, 5.0)
    axes.plot([low, high], [low, high], color=PLOTS.INK_MUTED, linewidth=1, linestyle="--",
              label="Perfect calibration")
    for _, column, _, label, color in MODELS[:3]:
        table = tables[column]
        axes.plot(table["predicted"], table["observed"].clip(lower=low), color=color, linewidth=2,
                  marker="o", markersize=5, markeredgecolor=PLOTS.SURFACE, markeredgewidth=1.2,
                  label=label)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed crash rate")
    PLOTS.style_axes(axes, "Calibration on 2023 validation", f"{bins} equal-size groups of cell-periods per model")
    axes.legend(frameon=False, fontsize=9, labelcolor=PLOTS.INK_SECONDARY, loc="upper left")
    return PLOTS.save(figure, path)


def compare(frame, repeats, seed):
    y_true = frame[TARGET].to_numpy(dtype=bool)
    metrics = {column: model_metrics(y_true, frame[column].to_numpy(dtype=np.float64))
               for _, column, *_ in MODELS}
    draws = bootstrap_differences(frame, repeats, seed)
    intervals = summarize_bootstrap(draws)
    details = pd.DataFrame([
        {"model": name, "prediction_column": column, "group": group, "label": label, **metrics[column]}
        for name, column, group, label, _ in MODELS
    ])
    return metrics, intervals, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    output = args.output_directory or args.root / "reports" / "model_comparison"
    (output / "figures").mkdir(parents=True, exist_ok=True)

    print("Loading 2023 validation predictions...")
    frame = load_predictions(processed)
    print(f"{len(frame):,} cell-periods, {int(frame[TARGET].sum()):,} with a crash")

    print(f"Scoring models and drawing {args.bootstrap_repeats} day-block bootstrap samples...")
    metrics, intervals, details = compare(frame, args.bootstrap_repeats, args.seed)

    shared_csv_rows(metrics, intervals, args.bootstrap_repeats).to_csv(
        output / "maria_model_comparison.csv", index=False
    )
    details.to_csv(output / "model_comparison_details.csv", index=False)
    intervals.to_csv(output / "bootstrap_intervals.csv", index=False)
    plot_differences(intervals, metrics, output / "figures" / "gain_over_baseline.png")
    plot_capture(frame, output / "figures" / "capture_curve.png")
    plot_calibration(frame, output / "figures" / "calibration.png")

    print("\nMODEL COMPARISON (2023 validation)")
    for row in details.itertuples():
        print(
            f"{row.label:>36}: AP={row.average_precision:.4f} ROC-AUC={row.roc_auc:.4f} "
            f"top5={row.top5_capture:.3f} Brier={row.brier_score:.5f} "
            f"slope={row.calibration_slope:.2f}"
        )
    gains = intervals[intervals["measure"].str.endswith("_vs_baseline")
                      & (intervals["prediction_column"] != BENCHMARK)]
    print("\nDIFFERENCE FROM ROLLING BASELINE (95% day-block interval)")
    for row in gains.itertuples():
        print(f"{row.label:>36} {row.measure:>32}: {row.low_95:+.4f} to {row.high_95:+.4f}")
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
