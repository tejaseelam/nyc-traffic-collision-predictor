"""Run the locked model once on the untouched 2024 test split.

The script requires the pre-test model freeze, records access before opening
the test file, and refuses to run if a final-test access record already exists.
This prevents accidental tuning or repeated evaluation on 2024.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREE = load_module("tree_model", ROOT / "scripts" / "10tree_model.py")
EXPECTED_TEST_POSITIVES = 81_413
EXPECTED_TEST_ROWS = 5_328_960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_freeze(freeze: dict) -> None:
    if freeze.get("status") != "locked_before_2024_test":
        raise RuntimeError("The model freeze is not locked")
    if freeze.get("test_data_opened") is not False:
        raise RuntimeError("The freeze must state that test data were not opened")
    if not freeze.get("features") or freeze.get("selected_model") not in {"full", "no_weather"}:
        raise RuntimeError("The freeze does not contain a valid selected model and feature list")
    if freeze.get("evaluation_rule") != "Rank all NYC cell-periods together and report the share of positive periods in the global highest-risk 5%.":
        raise RuntimeError("The required global top-5% rule is not frozen")


def shared_row(model: str, metrics: dict, notes: str) -> dict:
    return {
        "experiment": "final_test",
        "model": model,
        "group": "all_2024",
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "top5_capture": metrics["recall_at_top_fraction"],
        "top5_scope": "global",
        "n_rows": metrics["rows"],
        "n_positive": EXPECTED_TEST_POSITIVES,
        "split": "2024_test",
        "status": "complete",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    freeze_path = args.root / "reports" / "final_model" / "model_freeze.json"
    access_path = args.root / "reports" / "final_evaluation" / "FINAL_TEST_LOCK.json"
    if access_path.exists():
        raise RuntimeError(
            f"Final-test access is already recorded at {access_path}; refusing to open 2024 again"
        )
    if not freeze_path.is_file():
        raise FileNotFoundError("Run scripts/15select_final_model.py before the final test")
    freeze = json.loads(freeze_path.read_text())
    validate_freeze(freeze)

    output = access_path.parent
    output.mkdir(parents=True, exist_ok=True)
    access = {
        "status": "2024_test_access_started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha256(freeze_path),
        "selected_model": freeze["selected_model"],
        "warning": "This record is written before reading 2024. Do not rerun or tune after this point.",
    }
    access_path.write_text(json.dumps(access, indent=2) + "\n")

    processed = args.root / "data" / "processed"
    train_path = processed / TREE.LOGREG.TRAIN_FILE
    validation_path = processed / TREE.LOGREG.VALIDATION_FILE
    test_path = processed / "model_panel_test_2024.parquet"
    print("Opening the locked 2024 test split for the one final evaluation...")
    train = TREE.load_split(train_path, "Training file")
    validation = TREE.load_split(validation_path, "Validation file")
    test = TREE.load_split(test_path, "Test file")
    if len(test) != EXPECTED_TEST_ROWS or int(test[TREE.TARGET].sum()) != EXPECTED_TEST_POSITIVES:
        raise RuntimeError("The 2024 test file does not match the frozen split summary")
    if sorted(test["time_bin"].dt.year.unique()) != [2024]:
        raise RuntimeError("The test file must contain only 2024")

    train, validation, test = TREE.LOGREG.add_history_features(
        [train, validation, test],
        freeze["model_settings"]["prior_strength"],
        freeze["model_settings"]["recent_window_days"],
    )
    development = pd.concat([train, validation], ignore_index=True)
    del train, validation
    y_test = test[TREE.TARGET].to_numpy(dtype=bool)
    predictions = test[["cell_id", "time_bin", TREE.TARGET]].copy()

    baseline_probability = expit(test["history_cell_hour_logit"].to_numpy(dtype=np.float64))
    predictions["baseline"] = baseline_probability.astype(np.float32)

    negative_rate = freeze["model_settings"]["negative_sample_rate"]
    burn_in_days = freeze["model_settings"]["burn_in_days"]
    seed = freeze["model_settings"]["seed"]
    _, sampled = TREE.sampled_training_rows(development, negative_rate, burn_in_days, seed)

    print(f"Fitting locked logistic comparison on {len(sampled):,} sampled development rows...")
    logistic = TREE.LOGREG.build_model(0.1)
    logistic.fit(sampled[TREE.LOGREG.FEATURES], sampled[TREE.TARGET].to_numpy())
    logistic_probability = TREE.LOGREG.corrected_probability(
        logistic, test[TREE.LOGREG.FEATURES], negative_rate
    )
    predictions["logistic"] = logistic_probability.astype(np.float32)

    features = freeze["features"]
    levels = TREE.category_levels(development)
    settings = {
        key: freeze["model_settings"][key]
        for key in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "verbose")
    }
    settings["n_estimators"] = freeze["model_settings"]["validation_best_iteration"]
    print(f"Fitting locked {freeze['selected_model']} LightGBM on the same sampled rows...")
    tree = lightgbm.LGBMClassifier(**settings, random_state=seed)
    tree.fit(
        TREE.model_inputs(sampled, features, levels),
        sampled[TREE.TARGET].to_numpy(),
    )
    tree_probability = TREE.corrected_probability(
        tree, TREE.model_inputs(test, features, levels), negative_rate
    )
    predictions["lightgbm"] = tree_probability.astype(np.float32)

    probabilities = {
        "baseline": baseline_probability,
        "logistic": logistic_probability,
        "lightgbm": tree_probability,
    }
    metrics = {
        name: TREE.LOGREG.evaluate(y_test, probability, top_fraction=0.05)
        for name, probability in probabilities.items()
    }
    rows = [
        shared_row("baseline", metrics["baseline"], "Rolling past-only cell and six-hour-slot rate"),
        shared_row("logistic", metrics["logistic"], "Locked logistic comparison, C=0.1"),
        shared_row("lightgbm", metrics["lightgbm"], f"Locked {freeze['selected_model'].replace('_', ' ')} LightGBM"),
    ]
    pd.DataFrame(rows).to_csv(output / "final_test_metrics.csv", index=False)
    (output / "final_test_metrics.json").write_text(json.dumps({"models": metrics}, indent=2) + "\n")
    predictions.to_parquet(
        processed / "predictions" / "final_test_2024_predictions.parquet", index=False
    )
    model_path = processed / "models" / "final_lightgbm.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": tree,
            "features": features,
            "category_levels": levels,
            "negative_sample_rate": negative_rate,
            "freeze": freeze,
        },
        model_path,
    )

    access.update(
        {
            "status": "complete_do_not_rerun",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "test_sha256": sha256(test_path),
            "test_rows": len(test),
            "test_positive_periods": int(y_test.sum()),
            "top5_rule": "global highest-risk 5% of all 2024 cell-periods",
            "metrics_file": "final_test_metrics.json",
        }
    )
    access_path.write_text(json.dumps(access, indent=2) + "\n")
    print("\nFINAL 2024 TEST RESULTS — do not rerun or tune")
    for name, result in metrics.items():
        print(
            f"{name:>10}: AP={result['average_precision']:.4f} "
            f"ROC-AUC={result['roc_auc']:.4f} top5={result['recall_at_top_fraction']:.1%}"
        )


if __name__ == "__main__":
    main()
