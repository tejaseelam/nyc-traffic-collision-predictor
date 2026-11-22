"""Validate teammate CSVs and build the one official 2023 evaluation table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "experiment", "model", "group", "average_precision", "roc_auc",
    "top5_capture", "top5_scope", "n_rows", "n_positive", "split",
    "status", "notes",
]
CANONICAL_ROWS = 5_314_400
CANONICAL_POSITIVES = 86_527


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def model_comparison(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path).set_index("model")
    choices = {
        "baseline cell_hour_rate_rolling": ("baseline", "Rolling past-only cell and six-hour-slot rate"),
        "logistic C=0.1": ("logistic", "Logistic regression with C=0.1"),
        "tree core_plus_location (shallow)": ("lightgbm", "Full LightGBM with location"),
    }
    rows = []
    for source_name, (model, note) in choices.items():
        source = table.loc[source_name]
        rows.append(
            {
                "experiment": "model_comparison",
                "model": model,
                "group": "all_2023",
                "average_precision": source["average_precision"],
                "roc_auc": source["roc_auc"],
                "top5_capture": source["recall_at_top_fraction"],
                "top5_scope": "global",
                "n_rows": CANONICAL_ROWS,
                "n_positive": CANONICAL_POSITIVES,
                "split": "2023_validation",
                "status": "complete",
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def validate(table: pd.DataFrame) -> pd.DataFrame:
    if list(table.columns) != COLUMNS:
        raise ValueError(f"Expected columns {COLUMNS}; found {list(table.columns)}")
    if table[COLUMNS[:-1]].isna().any().any():
        raise ValueError("Official result rows may not have missing required values")
    if not table["split"].eq("2023_validation").all():
        raise ValueError("Only 2023 validation results belong in this file")
    if not table["status"].eq("complete").all():
        raise ValueError("Only complete result rows belong in this file")
    for column in ("average_precision", "roc_auc", "top5_capture"):
        if not table[column].between(0, 1, inclusive="both").all():
            raise ValueError(f"{column} must be between zero and one")
    if not table["top5_scope"].isin(["global", "within_group"]).all():
        raise ValueError("Unknown top5_scope")
    keys = ["experiment", "model", "group", "top5_scope", "split"]
    if table.duplicated(keys).any():
        raise ValueError("Duplicate official result rows")
    overall = table[table["top5_scope"].eq("global")]
    if not overall["n_rows"].eq(CANONICAL_ROWS).all() or not overall["n_positive"].eq(CANONICAL_POSITIVES).all():
        raise ValueError("Global results do not use the canonical 86,527-positive panel")
    for experiment in ("time_period", "historical_risk", "season", "weather"):
        part = table[table["experiment"].eq(experiment)]
        for model, group in part.groupby("model"):
            if group["n_positive"].sum() != CANONICAL_POSITIVES or group["n_rows"].sum() != CANONICAL_ROWS:
                raise ValueError(f"{experiment}/{model} does not partition the canonical panel")
    return table.sort_values(["experiment", "model", "group"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    sources = [
        args.root / "output" / "evaluation" / "feature_removal" / "blakely_feature_removal.csv",
        args.root / "yashwant_reporting_delay (1).csv",
        args.root / "ashik_time_risk.csv",
        args.root / "reports" / "season_weather" / "season_weather.csv",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checked teammate result files: " + ", ".join(missing))
    frames = [model_comparison(args.root / "reports" / "tree_model" / "model_comparison.csv")]
    for path in sources:
        frame = pd.read_csv(path)
        if list(frame.columns) != COLUMNS:
            raise ValueError(f"{path} does not use the shared results format")
        frames.append(frame)
    official = validate(pd.concat(frames, ignore_index=True))
    output = args.root / "reports" / "evaluation" / "evaluation_results.csv"
    official.to_csv(output, index=False, float_format="%.9f")
    manifest = {
        "canonical_validation_rows": CANONICAL_ROWS,
        "canonical_validation_positive_periods": CANONICAL_POSITIVES,
        "row_count": len(official),
        "sources": {str(path.relative_to(args.root)): sha256(path) for path in sources},
        "model_comparison_source": "reports/tree_model/model_comparison.csv",
        "validation": "All rows complete; metrics in [0,1]; global counts canonical; subgroup partitions sum to canonical counts; no duplicate keys; 2023 only.",
    }
    (output.parent / "evaluation_results_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(official)} checked rows to {output}")


if __name__ == "__main__":
    main()
