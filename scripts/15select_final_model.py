"""Choose between full and no-weather LightGBM using only 2023 validation.

This is the last model-selection step. It deliberately refuses paths that
mention 2024. After fitting both candidates with identical settings, it uses a
whole-day block bootstrap to compare average precision and top-5% capture.
The simpler no-weather model is selected when its point average precision is
not lower and the 95% interval does not show a meaningful disadvantage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREE = load_module("tree_model", ROOT / "scripts" / "10tree_model.py")
EXPECTED_VALIDATION_POSITIVES = 86_527


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_features() -> dict[str, list[str]]:
    full = list(TREE.FEATURE_SETS["core_plus_location"])
    weather = set(TREE.FEATURE_GROUPS["Weather"])
    return {
        "full": full,
        "no_weather": [column for column in full if column not in weather],
    }


def bootstrap_difference(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    """Bootstrap daily score differences, resampling complete calendar days.

    Computing each day's scores once makes 1,000 resamples practical while
    retaining the important dependence unit: every NYC cell-period on a given
    calendar day stays together.
    """
    work = frame.assign(day=pd.to_datetime(frame["time_bin"]).dt.normalize())
    daily = []
    for day, group in work.groupby("day", sort=True, observed=True):
        y_true = group[TREE.TARGET].to_numpy(dtype=bool)
        full = TREE.LOGREG.evaluate(y_true, group["full"].to_numpy())
        no_weather = TREE.LOGREG.evaluate(y_true, group["no_weather"].to_numpy())
        daily.append(
            {
                "day": day,
                "average_precision_difference": no_weather["average_precision"] - full["average_precision"],
                "top5_capture_difference": no_weather["recall_at_top_fraction"] - full["recall_at_top_fraction"],
            }
        )
    daily = pd.DataFrame(daily)
    random = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        sample = daily.iloc[random.integers(0, len(daily), len(daily))]
        rows.append(
            {
                "repeat": repeat,
                "average_precision_difference": sample["average_precision_difference"].mean(),
                "top5_capture_difference": sample["top5_capture_difference"].mean(),
            }
        )
    draws = pd.DataFrame(rows)
    result = {}
    for measure in ("average_precision_difference", "top5_capture_difference"):
        values = draws[measure]
        result[measure] = {
            "low_95": float(values.quantile(0.025)),
            "high_95": float(values.quantile(0.975)),
            "share_above_zero": float((values > 0).mean()),
        }
    return result, draws


def choose_model(full_ap: float, no_weather_ap: float, interval: dict) -> tuple[str, str]:
    """Apply the predeclared simplicity rule without looking at test data."""
    if no_weather_ap >= full_ap and interval["low_95"] > -0.0005:
        return (
            "no_weather",
            "No-weather was not worse on validation and its 95% interval did not show a meaningful disadvantage; selected for simplicity.",
        )
    return (
        "full",
        "The no-weather candidate did not satisfy the predeclared validation and uncertainty rule.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    train_path = processed / TREE.LOGREG.TRAIN_FILE
    validation_path = processed / TREE.LOGREG.VALIDATION_FILE
    if any("2024" in str(path) for path in (train_path, validation_path)):
        raise RuntimeError("Model selection must not open 2024 data")

    train = TREE.load_split(train_path, "Training file")
    validation = TREE.load_split(validation_path, "Validation file")
    positives = int(validation[TREE.TARGET].sum())
    if positives != EXPECTED_VALIDATION_POSITIVES:
        raise RuntimeError(
            f"Expected canonical validation panel with {EXPECTED_VALIDATION_POSITIVES:,} positives; found {positives:,}"
        )

    features = candidate_features()
    results, models, levels, _, predictions = TREE.fit_and_evaluate(
        train,
        validation,
        candidates=TREE.CANDIDATES,
        feature_sets=features,
        seed=args.seed,
    )
    predictions = predictions.rename(
        columns={"tree_full": "full", "tree_no_weather": "no_weather"}
    )
    bootstrap, draws = bootstrap_difference(predictions, args.bootstrap_repeats, args.seed)
    metrics = {
        name: results["tree"][name][results["selected"][name]]
        for name in features
    }
    selected, reason = choose_model(
        metrics["full"]["average_precision"],
        metrics["no_weather"]["average_precision"],
        bootstrap["average_precision_difference"],
    )

    output = args.root / "reports" / "final_model"
    output.mkdir(parents=True, exist_ok=True)
    draws.to_csv(output / "full_vs_no_weather_bootstrap.csv", index=False)
    predictions.to_parquet(
        processed / "predictions" / "validation_2023_final_candidates.parquet",
        index=False,
    )

    freeze = {
        "status": "locked_before_2024_test",
        "canonical_panel": {
            "validation_positive_periods": positives,
            "train_sha256": sha256(train_path),
            "validation_sha256": sha256(validation_path),
        },
        "selection_split": "2023_validation",
        "selected_model": selected,
        "selection_reason": reason,
        "features": features[selected],
        "excluded_features": sorted(set(features["full"]) - set(features[selected])),
        "model_settings": {
            **TREE.COMMON_SETTINGS,
            **TREE.CANDIDATES[results["selected"][selected]],
            "validation_best_iteration": metrics[selected]["iterations"],
            "negative_sample_rate": results["settings"]["negative_sample_rate"],
            "burn_in_days": results["settings"]["burn_in_days"],
            "prior_strength": results["settings"]["prior_strength"],
            "recent_window_days": results["settings"]["recent_window_days"],
            "seed": args.seed,
        },
        "decision_rule": "Choose no-weather when its validation AP is not lower and the day-block 95% CI lower bound is above -0.0005; otherwise choose full.",
        "evaluation_rule": "Rank all NYC cell-periods together and report the share of positive periods in the global highest-risk 5%.",
        "validation_metrics": metrics,
        "no_weather_minus_full_bootstrap": bootstrap,
        "bootstrap_repeats": args.bootstrap_repeats,
        "test_data_opened": False,
    }
    (output / "model_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    (args.root / "MODEL_FREEZE.md").write_text(
        "# Final model freeze\n\n"
        f"The canonical validation panel contains **{positives:,} positive cell-periods** and has SHA-256 `{freeze['canonical_panel']['validation_sha256']}`.\n\n"
        f"The selected model is **LightGBM ({selected.replace('_', ' ')})**. {reason}\n\n"
        "Features, settings, and the citywide top-5% evaluation rule are locked in `reports/final_model/model_freeze.json`. The 2024 test was not opened during selection.\n"
    )
    print(f"Selected {selected}: {reason}")
    print(f"Freeze: {output / 'model_freeze.json'}")


if __name__ == "__main__":
    main()
