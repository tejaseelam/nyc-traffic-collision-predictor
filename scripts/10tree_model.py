"""Fit a gradient-boosted tree model for has_crash and analyze its features.

Only the training (2021-2022) and validation (2023) files are read. The 2024
test file is intentionally never opened here.

The setup matches scripts/8baseline_logreg.py so the two models can be compared
directly: the same past-only crash-history features, burn-in period, negative
subsampling with probability correction, evaluation measurements, and model
selection by validation average precision. The shared pieces are imported from
that script rather than copied.

Feature sets:

* core:                exactly the logistic regression's features
* core_plus_location:  the core features plus the grid cell's x and y position,
                       which a tree can split into neighborhoods but a linear
                       model cannot use

Feature importance is measured by permutation on a sample of validation rows:
the drop in average precision when one feature, or one group of related
features, is shuffled. When the saved logistic regression is available, the
same measurement is made for it so the two models can be compared.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baseline_logreg", Path(__file__).resolve().parent / "8baseline_logreg.py"
)
LOGREG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGREG)

TARGET = LOGREG.TARGET
PREDICTIONS_FILE = "validation_2023_tree_predictions.parquet"
LOGREG_MODEL_FILE = "logreg_has_crash.joblib"
LOGREG_METRICS_FILE = Path("reports") / "baseline_logreg" / "validation_metrics.json"

LOCATION_COLUMNS = ["grid_x", "grid_y"]
CATEGORICAL_FEATURES = LOGREG.CATEGORICAL_FEATURES
FEATURE_SETS = {
    "core": LOGREG.FEATURES,
    "core_plus_location": LOGREG.FEATURES + LOCATION_COLUMNS,
}
FEATURE_GROUPS = {
    "Crash history": [
        "history_cell_logit",
        "history_cell_hour_logit",
        "history_recent_logit",
    ],
    "Time": ["hour", "day_of_week", "month"],
    "Location": ["borough", "land_fraction", *LOCATION_COLUMNS],
    "Roads and speed limits": [
        "log_speed_road_length_m",
        "log_speed_road_segments",
        "speed_limit_weighted_avg_mph",
        "speed_limit_coverage",
    ],
    "Weather": [
        "weather_temp_midpoint_c",
        "weather_precip_mm",
        "weather_snowfall_mm",
        "weather_snow_depth_mm",
        "weather_wind_mps",
        "weather_precip_day",
        "weather_snow_day",
        "weather_any_missing",
    ],
}
# Settings shared by every candidate. Early stopping holds out a random tenth
# of the sampled training rows, so the 2023 file is never used while fitting.
COMMON_SETTINGS = {
    "max_iter": 1000,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 30,
}
CANDIDATES = {
    "shallow": {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 200},
    "medium": {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 500},
    "deep": {"learning_rate": 0.05, "max_leaf_nodes": 127, "min_samples_leaf": 1000},
}


def load_split(path, label):
    """Load a split like the logistic script does, plus the cell's grid position."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    frame = pd.read_parquet(
        path,
        columns=LOGREG.PANEL_COLUMNS + LOCATION_COLUMNS,
        read_dictionary=["cell_id", "borough"],
    )
    LOGREG.require_columns(frame, LOCATION_COLUMNS, label)
    panel = LOGREG.prepare_panel(frame, label)
    for column in LOCATION_COLUMNS:
        panel[column] = frame[column].to_numpy(dtype=np.float32)
    return panel


def category_levels(frame):
    """Fix the category levels from the training rows so every split agrees."""
    return {
        column: sorted(frame[column].astype("string").dropna().unique())
        if column == "borough"
        else sorted(int(value) for value in frame[column].unique())
        for column in CATEGORICAL_FEATURES
    }


def model_inputs(frame, features, levels):
    """Feature table with calendar fields and borough as tree categories."""
    inputs = frame[features].copy()
    for column in CATEGORICAL_FEATURES:
        if column in inputs:
            inputs[column] = pd.Categorical(
                inputs[column].astype("string") if column == "borough" else inputs[column],
                categories=levels[column],
            )
    return inputs


def sampled_training_rows(train, negative_rate, burn_in_days, seed):
    """Training rows after the burn-in, and those rows with zero-crash rows subsampled."""
    burn_in_end = train["time_bin"].min() + pd.Timedelta(days=burn_in_days)
    eligible = train.loc[train["time_bin"] >= burn_in_end]
    if eligible.empty:
        raise ValueError("No training rows remain after the burn-in period")
    rows = LOGREG.sample_training_rows(eligible[TARGET].to_numpy(), negative_rate, seed)
    return eligible, eligible.iloc[rows]


def build_model(settings, seed):
    return HistGradientBoostingClassifier(
        **COMMON_SETTINGS,
        **settings,
        categorical_features="from_dtype",
        random_state=seed,
    )


def tree_scorer(model, features, levels, negative_rate):
    def score(frame):
        return LOGREG.corrected_probability(
            model, model_inputs(frame, features, levels), negative_rate
        )
    return score


def logistic_scorer(saved):
    def score(frame):
        return LOGREG.corrected_probability(
            saved["model"], frame[saved["features"]], saved["negative_sample_rate"]
        )
    return score


def permutation_importance(score, frame, groups, repeats, seed):
    """Average-precision drop when each group of columns is shuffled together.

    Shuffling related columns as one block (for example all three crash-history
    features) measures what the model loses without that kind of information,
    instead of letting a correlated partner column cover for the shuffled one.
    """
    y_true = frame[TARGET].to_numpy(dtype=bool)
    reference = average_precision_score(y_true, score(frame))
    random = np.random.default_rng(seed)
    rows = []
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            order = random.permutation(len(frame))
            shuffled = frame.copy(deep=False)
            for column in columns:
                shuffled[column] = frame[column].iloc[order].set_axis(frame.index)
            drops.append(reference - average_precision_score(y_true, score(shuffled)))
        rows.append({
            "name": name,
            "columns": ",".join(columns),
            "average_precision_drop": float(np.mean(drops)),
            "average_precision_drop_std": float(np.std(drops)),
            "relative_drop": float(np.mean(drops) / reference),
        })
    table = pd.DataFrame(rows).sort_values("average_precision_drop", ascending=False)
    return table.reset_index(drop=True), reference


def importance_tables(score, features, frame, repeats, seed, model_name):
    """Grouped and single-feature permutation importance for one model."""
    groups = {
        name: [column for column in columns if column in features]
        for name, columns in FEATURE_GROUPS.items()
    }
    groups = {name: columns for name, columns in groups.items() if columns}
    tables = []
    for level, level_groups in [
        ("group", groups),
        ("feature", {column: [column] for column in features}),
    ]:
        print(f"Permutation importance for {model_name} ({level})...")
        table, _ = permutation_importance(score, frame, level_groups, repeats, seed)
        table.insert(0, "level", level)
        table.insert(0, "model", model_name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def fit_and_evaluate(
    train,
    validation,
    candidates=None,
    feature_sets=None,
    negative_rate=0.1,
    burn_in_days=28,
    prior_strength=20.0,
    recent_window_days=28,
    seed=42,
    top_fraction=0.05,
):
    """Fit every candidate on every feature set and score them on validation.

    Returns the metrics, the best model for each feature set, the category
    levels, and the validation rows with each selected model's predictions.
    """
    candidates = candidates or CANDIDATES
    feature_sets = feature_sets or FEATURE_SETS
    train, validation = LOGREG.add_history_features(
        [train, validation], prior_strength, recent_window_days
    )
    y_validation = validation[TARGET].to_numpy()
    levels = category_levels(train)

    eligible, fit_frame = sampled_training_rows(train, negative_rate, burn_in_days, seed)
    y_fit = fit_frame[TARGET].to_numpy()
    print(
        f"Fitting on {len(fit_frame):,} sampled rows "
        f"({int(y_fit.sum()):,} with crashes)"
    )

    results = {"tree": {}, "selected": {}}
    models = {}
    predictions = validation[["cell_id", "time_bin", TARGET]].copy()
    for set_name, features in feature_sets.items():
        fit_inputs = model_inputs(fit_frame, features, levels)
        validation_inputs = model_inputs(validation, features, levels)
        results["tree"][set_name] = {}
        best_name = None
        for name, settings in candidates.items():
            print(f"Fitting tree {set_name}/{name}...")
            model = build_model(settings, seed)
            start = time.perf_counter()
            model.fit(fit_inputs, y_fit)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            probability = LOGREG.corrected_probability(model, validation_inputs, negative_rate)
            predict_seconds = time.perf_counter() - start

            metrics = LOGREG.evaluate(y_validation, probability, top_fraction)
            metrics.update({
                "settings": {**COMMON_SETTINGS, **settings},
                "iterations": int(model.n_iter_),
                "fit_seconds": round(fit_seconds, 1),
                "predict_seconds": round(predict_seconds, 1),
            })
            results["tree"][set_name][name] = metrics
            print(
                f"  AP={metrics['average_precision']:.4f} "
                f"iterations={model.n_iter_} fit={fit_seconds:.0f}s"
            )
            best = results["tree"][set_name].get(best_name)
            if best is None or metrics["average_precision"] > best["average_precision"]:
                best_name = name
                models[set_name] = model
                predictions[f"tree_{set_name}"] = probability.astype(np.float32)
        results["selected"][set_name] = best_name

    results["selection_metric"] = "validation average_precision"
    results["settings"] = {
        "model": "sklearn HistGradientBoostingClassifier",
        "candidates": {name: {**COMMON_SETTINGS, **settings} for name, settings in candidates.items()},
        "feature_sets": feature_sets,
        "negative_sample_rate": negative_rate,
        "burn_in_days": burn_in_days,
        "prior_strength": prior_strength,
        "recent_window_days": recent_window_days,
        "seed": seed,
        "training_rows_after_burn_in": int(len(eligible)),
        "fitted_rows": int(len(fit_frame)),
    }
    return results, models, levels, validation, predictions


def comparison_table(results, logreg_metrics):
    """One row per model, using the logistic script's saved metrics if present."""
    rows = []
    if logreg_metrics:
        for name, metrics in logreg_metrics["baselines"].items():
            rows.append({"model": f"baseline {name}", **metrics})
        selected = logreg_metrics["selected_logistic"]
        rows.append({"model": f"logistic {selected}", **logreg_metrics["logistic"][selected]})
    for set_name, selected in results["selected"].items():
        metrics = results["tree"][set_name][selected]
        rows.append({"model": f"tree {set_name} ({selected})", **metrics})
    columns = [
        "model",
        "average_precision",
        "roc_auc",
        "log_loss",
        "brier_score",
        "precision_at_top_fraction",
        "recall_at_top_fraction",
        "best_f1",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--negative-sample-rate", type=float, default=0.1)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--recent-window-days", type=int, default=28)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--importance-rows", type=int, default=1_000_000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    output_directory = args.output_directory or args.root / "reports" / "tree_model"
    model_path = args.model_path or processed / "models" / "tree_has_crash.joblib"
    names = [name.strip() for name in args.candidates.split(",") if name.strip()]
    unknown = sorted(set(names) - set(CANDIDATES))
    if not names or unknown:
        raise ValueError(f"Choose candidates from: {', '.join(CANDIDATES)}")
    candidates = {name: CANDIDATES[name] for name in names}

    started = time.perf_counter()
    print("Loading training and validation files...")
    train = load_split(processed / LOGREG.TRAIN_FILE, "Training file")
    validation = load_split(processed / LOGREG.VALIDATION_FILE, "Validation file")

    results, models, levels, validation, predictions = fit_and_evaluate(
        train,
        validation,
        candidates=candidates,
        negative_rate=args.negative_sample_rate,
        burn_in_days=args.burn_in_days,
        prior_strength=args.prior_strength,
        recent_window_days=args.recent_window_days,
        seed=args.seed,
        top_fraction=args.top_fraction,
    )
    del train

    sample_size = min(args.importance_rows, len(validation))
    sample = validation.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    importance = [
        importance_tables(
            tree_scorer(models[set_name], features, levels, args.negative_sample_rate),
            features, sample, args.importance_repeats, args.seed, f"tree_{set_name}",
        )
        for set_name, features in FEATURE_SETS.items()
    ]
    logreg_path = processed / "models" / LOGREG_MODEL_FILE
    if logreg_path.is_file():
        saved = joblib.load(logreg_path)
        importance.append(importance_tables(
            logistic_scorer(saved), saved["features"], sample,
            args.importance_repeats, args.seed, "logistic",
        ))
    else:
        print(f"Skipping logistic importance: {logreg_path} not found")
    importance = pd.concat(importance, ignore_index=True)

    logreg_metrics_path = args.root / LOGREG_METRICS_FILE
    logreg_metrics = (
        json.loads(logreg_metrics_path.read_text()) if logreg_metrics_path.is_file() else None
    )
    comparison = comparison_table(results, logreg_metrics)

    results["settings"].update({
        "importance_rows": sample_size,
        "importance_repeats": args.importance_repeats,
    })
    results["runtime"] = {
        "total_seconds": round(time.perf_counter() - started, 1),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "validation_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    importance.to_csv(output_directory / "feature_importance.csv", index=False)
    comparison.to_csv(output_directory / "model_comparison.csv", index=False)
    predictions_path = processed / "predictions" / PREDICTIONS_FILE
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "selected": results["selected"],
            "feature_sets": FEATURE_SETS,
            "category_levels": levels,
            "negative_sample_rate": args.negative_sample_rate,
        },
        model_path,
    )

    print("\nVALIDATION RESULTS (2023)")
    for row in comparison.itertuples():
        print(
            f"{row.model:>36}: AP={row.average_precision:.4f} "
            f"ROC-AUC={row.roc_auc:.4f} log-loss={row.log_loss:.4f} "
            f"recall@top{args.top_fraction:.0%}={row.recall_at_top_fraction:.3f}"
        )
    print(f"Total runtime: {results['runtime']['total_seconds']:.0f}s")
    print(f"Metrics: {output_directory / 'validation_metrics.json'}")
    print(f"Comparison: {output_directory / 'model_comparison.csv'}")
    print(f"Feature importance: {output_directory / 'feature_importance.csv'}")
    print(f"Validation predictions: {predictions_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
