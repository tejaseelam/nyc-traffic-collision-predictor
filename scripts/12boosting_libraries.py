"""Fit LightGBM and XGBoost for has_crash and compare them with the other models.

Only the training (2021-2022) and validation (2023) files are read. The 2024
test file is intentionally never opened here.

Both libraries build gradient-boosted trees, like scikit-learn's
HistGradientBoostingClassifier in scripts/10tree_model.py, and use the same
setup so the three can be compared directly: the same feature sets, past-only
crash-history features, burn-in period, negative subsampling with probability
correction, evaluation measurements, and selection by validation average
precision. The shared pieces are imported from 10tree_model.py.

The tree sizes from 10tree_model.py are translated to each library:

* LightGBM grows each tree one leaf at a time, like scikit-learn, so the leaf
  count and minimum rows per leaf carry over directly.
* XGBoost grows each tree one level at a time by default. Each size uses the
  depth that allows about the same number of leaves (31 leaves -> depth 5,
  63 -> 6, 127 -> 7), and the minimum leaf size is converted from rows to the
  weight XGBoost uses instead (see min_child_weight below).

Early stopping holds out the same random tenth of the sampled training rows for
both libraries, so the 2023 file is never used while fitting.

On macOS both libraries need OpenMP: brew install libomp
"""

from pathlib import Path
import argparse
import importlib.util
import json
import os
import platform
import time

import joblib
import lightgbm
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
import xgboost


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tree_model", Path(__file__).resolve().parent / "10tree_model.py"
)
TREE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TREE)
LOGREG = TREE.LOGREG

TARGET = TREE.TARGET
PREDICTIONS_FILE = "validation_2023_boosting_predictions.parquet"
TREE_METRICS_FILE = Path("reports") / "tree_model" / "validation_metrics.json"

FEATURE_SETS = TREE.FEATURE_SETS
CANDIDATES = TREE.CANDIDATES
LIBRARIES = ["lightgbm", "xgboost"]
MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 30
EARLY_STOPPING_FRACTION = 0.1
L2_REGULARIZATION = 1.0


def depth_for_leaves(leaves):
    """Smallest depth whose level-by-level tree can hold this many leaves."""
    return int(np.ceil(np.log2(leaves)))


def build_model(library, settings, seed, positive_share):
    if library == "lightgbm":
        return lightgbm.LGBMClassifier(
            n_estimators=MAX_ROUNDS,
            learning_rate=settings["learning_rate"],
            num_leaves=settings["max_leaf_nodes"],
            min_child_samples=settings["min_samples_leaf"],
            reg_lambda=L2_REGULARIZATION,
            random_state=seed,
            verbose=-1,
        )
    # XGBoost limits leaves by total weight, where each row weighs p * (1 - p)
    # for its predicted probability p. Using the crash share of the sampled
    # rows for p turns the row minimum into a comparable weight minimum.
    return xgboost.XGBClassifier(
        n_estimators=MAX_ROUNDS,
        learning_rate=settings["learning_rate"],
        max_depth=depth_for_leaves(settings["max_leaf_nodes"]),
        min_child_weight=settings["min_samples_leaf"] * positive_share * (1 - positive_share),
        reg_lambda=L2_REGULARIZATION,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="logloss",
        random_state=seed,
    )


def fit_model(model, fit_inputs, y_fit, stop_inputs, y_stop):
    """Fit with early stopping and return the number of trees kept."""
    if isinstance(model, lightgbm.LGBMClassifier):
        model.fit(
            fit_inputs,
            y_fit,
            eval_set=[(stop_inputs, y_stop)],
            eval_metric="binary_logloss",
            callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return int(model.best_iteration_)
    model.fit(fit_inputs, y_fit, eval_set=[(stop_inputs, y_stop)], verbose=False)
    return int(model.best_iteration) + 1


def corrected_probability(model, inputs, negative_rate):
    """Undo the log-odds shift caused by subsampling negatives.

    Same correction as LOGREG.corrected_probability, which needs a
    decision_function that these libraries do not have.
    """
    if isinstance(model, lightgbm.LGBMClassifier):
        log_odds = model.predict(inputs, raw_score=True)
    else:
        log_odds = model.predict(inputs, output_margin=True)
    return expit(log_odds + np.log(negative_rate))


def fit_and_evaluate(
    train,
    validation,
    libraries=None,
    candidates=None,
    feature_sets=None,
    negative_rate=0.1,
    burn_in_days=28,
    prior_strength=20.0,
    recent_window_days=28,
    seed=42,
    top_fraction=0.05,
):
    """Fit every library and candidate on every feature set and score them.

    Returns the metrics, the best model for each library and feature set, and
    the validation rows with each selected model's predictions.
    """
    libraries = libraries or LIBRARIES
    candidates = candidates or CANDIDATES
    feature_sets = feature_sets or FEATURE_SETS
    train, validation = LOGREG.add_history_features(
        [train, validation], prior_strength, recent_window_days
    )
    y_validation = validation[TARGET].to_numpy()
    levels = TREE.category_levels(train)

    eligible, sampled = TREE.sampled_training_rows(train, negative_rate, burn_in_days, seed)
    fit_rows, stop_rows = train_test_split(
        np.arange(len(sampled)),
        test_size=EARLY_STOPPING_FRACTION,
        stratify=sampled[TARGET].to_numpy(),
        random_state=seed,
    )
    fit_frame = sampled.iloc[fit_rows]
    stop_frame = sampled.iloc[stop_rows]
    y_fit = fit_frame[TARGET].to_numpy()
    y_stop = stop_frame[TARGET].to_numpy()
    positive_share = float(sampled[TARGET].mean())
    print(
        f"Fitting on {len(fit_frame):,} sampled rows ({int(y_fit.sum()):,} with crashes), "
        f"early stopping on {len(stop_frame):,}"
    )

    results = {library: {} for library in libraries}
    results["selected"] = {library: {} for library in libraries}
    models = {library: {} for library in libraries}
    predictions = validation[["cell_id", "time_bin", TARGET]].copy()
    for set_name, features in feature_sets.items():
        fit_inputs = TREE.model_inputs(fit_frame, features, levels)
        stop_inputs = TREE.model_inputs(stop_frame, features, levels)
        validation_inputs = TREE.model_inputs(validation, features, levels)
        for library in libraries:
            results[library][set_name] = {}
            best_name = None
            for name, settings in candidates.items():
                print(f"Fitting {library} {set_name}/{name}...")
                model = build_model(library, settings, seed, positive_share)
                start = time.perf_counter()
                trees = fit_model(model, fit_inputs, y_fit, stop_inputs, y_stop)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                probability = corrected_probability(model, validation_inputs, negative_rate)
                predict_seconds = time.perf_counter() - start

                metrics = LOGREG.evaluate(y_validation, probability, top_fraction)
                metrics.update({
                    "settings": model.get_params(),
                    "iterations": trees,
                    "fit_seconds": round(fit_seconds, 1),
                    "predict_seconds": round(predict_seconds, 1),
                })
                results[library][set_name][name] = metrics
                print(
                    f"  AP={metrics['average_precision']:.4f} "
                    f"trees={trees} fit={fit_seconds:.0f}s"
                )
                best = results[library][set_name].get(best_name)
                if best is None or metrics["average_precision"] > best["average_precision"]:
                    best_name = name
                    models[library][set_name] = model
                    predictions[f"{library}_{set_name}"] = probability.astype(np.float32)
            results["selected"][library][set_name] = best_name

    results["selection_metric"] = "validation average_precision"
    results["settings"] = {
        "libraries": libraries,
        "candidates": candidates,
        "feature_sets": feature_sets,
        "max_rounds": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_fraction": EARLY_STOPPING_FRACTION,
        "l2_regularization": L2_REGULARIZATION,
        "negative_sample_rate": negative_rate,
        "burn_in_days": burn_in_days,
        "prior_strength": prior_strength,
        "recent_window_days": recent_window_days,
        "seed": seed,
        "training_rows_after_burn_in": int(len(eligible)),
        "fitted_rows": int(len(fit_frame)),
        "early_stopping_rows": int(len(stop_frame)),
    }
    return results, models, levels, predictions


def comparison_table(results, tree_metrics, logreg_metrics):
    """One row per model: the saved baseline, logistic, and scikit-learn tree
    results when present, then the selected LightGBM and XGBoost models."""
    earlier = TREE.comparison_table(tree_metrics or {"selected": {}}, logreg_metrics)
    earlier["model"] = earlier["model"].replace(r"^tree ", "sklearn ", regex=True)
    rows = []
    for library in results["settings"]["libraries"]:
        for set_name, selected in results["selected"][library].items():
            metrics = results[library][set_name][selected]
            rows.append({"model": f"{library} {set_name} ({selected})", **metrics})
    return pd.concat([earlier, pd.DataFrame(rows)[earlier.columns]], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--libraries", default=",".join(LIBRARIES))
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--negative-sample-rate", type=float, default=0.1)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--recent-window-days", type=int, default=28)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    output_directory = args.output_directory or args.root / "reports" / "boosting_libraries"
    model_path = args.model_path or processed / "models" / "boosting_has_crash.joblib"
    choices = {"libraries": LIBRARIES, "candidates": CANDIDATES}
    chosen = {}
    for option, allowed in choices.items():
        names = [name.strip() for name in getattr(args, option).split(",") if name.strip()]
        if not names or set(names) - set(allowed):
            raise ValueError(f"Choose {option} from: {', '.join(allowed)}")
        chosen[option] = names
    candidates = {name: CANDIDATES[name] for name in chosen["candidates"]}

    started = time.perf_counter()
    print("Loading training and validation files...")
    train = TREE.load_split(processed / LOGREG.TRAIN_FILE, "Training file")
    validation = TREE.load_split(processed / LOGREG.VALIDATION_FILE, "Validation file")

    results, models, levels, predictions = fit_and_evaluate(
        train,
        validation,
        libraries=chosen["libraries"],
        candidates=candidates,
        negative_rate=args.negative_sample_rate,
        burn_in_days=args.burn_in_days,
        prior_strength=args.prior_strength,
        recent_window_days=args.recent_window_days,
        seed=args.seed,
        top_fraction=args.top_fraction,
    )
    del train, validation

    saved = {}
    for name, path in [("tree", TREE_METRICS_FILE), ("logreg", TREE.LOGREG_METRICS_FILE)]:
        path = args.root / path
        saved[name] = json.loads(path.read_text()) if path.is_file() else None
    comparison = comparison_table(results, saved["tree"], saved["logreg"])

    results["runtime"] = {
        "total_seconds": round(time.perf_counter() - started, 1),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "validation_metrics.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
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
            f"{row.model:>40}: AP={row.average_precision:.4f} "
            f"ROC-AUC={row.roc_auc:.4f} log-loss={row.log_loss:.4f} "
            f"recall@top{args.top_fraction:.0%}={row.recall_at_top_fraction:.3f}"
        )
    print(f"Total runtime: {results['runtime']['total_seconds']:.0f}s")
    print(f"Metrics: {output_directory / 'validation_metrics.json'}")
    print(f"Comparison: {output_directory / 'model_comparison.csv'}")
    print(f"Validation predictions: {predictions_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
