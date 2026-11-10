"""Run LightGBM feature-group removal experiments on 2023 validation data.

This script reads only the 2021-2022 training split and 2023 validation split.
It never opens the 2024 test file. Every ablation uses the same sampled training
rows, LightGBM settings, early-stopping procedure, and global top-5% definition
as scripts/10tree_model.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tree_model", Path(__file__).resolve().parent / "10tree_model.py"
)
TREE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TREE)

SHARED_COLUMNS = [
    "experiment",
    "model",
    "group",
    "average_precision",
    "roc_auc",
    "top5_capture",
    "top5_scope",
    "n_rows",
    "n_positive",
    "split",
    "status",
    "notes",
]

REMOVALS = {
    "full_features": [],
    "no_crash_history": TREE.FEATURE_GROUPS["Crash history"],
    "no_time": TREE.FEATURE_GROUPS["Time"],
    "no_roads_speed": TREE.FEATURE_GROUPS["Roads and speed limits"],
    "no_location": TREE.FEATURE_GROUPS["Location"],
    "no_weather": TREE.FEATURE_GROUPS["Weather"],
}

DISPLAY_NAMES = {
    "full_features": "Full model",
    "no_crash_history": "Without crash history",
    "no_time": "Without time",
    "no_roads_speed": "Without roads / speed",
    "no_location": "Without location",
    "no_weather": "Without weather",
}


def feature_sets() -> dict[str, list[str]]:
    full = TREE.FEATURE_SETS["core_plus_location"]
    result = {}
    for name, removed in REMOVALS.items():
        unknown = sorted(set(removed) - set(full))
        if unknown:
            raise ValueError(f"Removal group {name} contains unknown columns: {unknown}")
        result[name] = [column for column in full if column not in removed]
    return result


def selected_metrics(results: dict, name: str) -> dict:
    selected = results["selected"][name]
    return results["tree"][name][selected]


def reference_metrics(root: Path) -> dict[str, dict]:
    path = root / "reports" / "tree_model" / "model_comparison.csv"
    table = pd.read_csv(path)
    references = {}
    choices = {
        "baseline": "baseline cell_hour_rate_rolling",
        "logistic": "logistic C=0.1",
    }
    for short_name, report_name in choices.items():
        match = table.loc[table["model"] == report_name]
        if len(match) != 1:
            raise ValueError(f"Expected one {report_name!r} row in {path}")
        references[short_name] = match.iloc[0].to_dict()
    return references


def shared_rows(results: dict, n_positive: int) -> list[dict]:
    rows = []
    for name in REMOVALS:
        metrics = selected_metrics(results, name)
        rows.append(
            {
                "experiment": "feature_removal",
                "model": "lightgbm",
                "group": name,
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "top5_capture": metrics["recall_at_top_fraction"],
                "top5_scope": "global",
                "n_rows": metrics["rows"],
                "n_positive": n_positive,
                "split": "2023_validation",
                "status": "complete",
                "notes": (
                    "Current full LightGBM feature set"
                    if name == "full_features"
                    else f"Retrained after removing: {', '.join(REMOVALS[name])}"
                ),
            }
        )
    return rows


def detail_rows(results: dict, references: dict[str, dict]) -> list[dict]:
    full = selected_metrics(results, "full_features")
    rows = []
    for name in REMOVALS:
        metrics = selected_metrics(results, name)
        rows.append(
            {
                "variant": name,
                "display_name": DISPLAY_NAMES[name],
                "removed_columns": ",".join(REMOVALS[name]),
                "n_features": len(results["settings"]["feature_sets"][name]),
                "average_precision": metrics["average_precision"],
                "average_precision_drop": full["average_precision"] - metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "roc_auc_drop": full["roc_auc"] - metrics["roc_auc"],
                "top5_capture": metrics["recall_at_top_fraction"],
                "top5_capture_drop": full["recall_at_top_fraction"] - metrics["recall_at_top_fraction"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "iterations": metrics["iterations"],
                "fit_seconds": metrics["fit_seconds"],
                "predict_seconds": metrics["predict_seconds"],
            }
        )
    for model, metrics in references.items():
        rows.append(
            {
                "variant": model,
                "display_name": "Rolling baseline" if model == "baseline" else "Logistic regression",
                "removed_columns": "",
                "n_features": "",
                "average_precision": metrics["average_precision"],
                "average_precision_drop": full["average_precision"] - metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "roc_auc_drop": full["roc_auc"] - metrics["roc_auc"],
                "top5_capture": metrics["recall_at_top_fraction"],
                "top5_capture_drop": full["recall_at_top_fraction"] - metrics["recall_at_top_fraction"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "iterations": "",
                "fit_seconds": "",
                "predict_seconds": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def svg_chart(rows: list[dict], path: Path) -> None:
    variants = [row for row in rows if row["variant"] not in {"baseline", "logistic"}]
    variants = sorted(variants, key=lambda row: row["average_precision_drop"], reverse=True)
    references = {row["variant"]: row for row in rows if row["variant"] in {"baseline", "logistic"}}
    width, height = 1400, 940
    left, top, chart_width = 320, 175, 980
    row_height = 70
    max_drop = max(max(row["average_precision_drop"], 0) for row in variants) or 1
    max_capture = max(row["top5_capture"] for row in rows) * 1.05

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17303a}.title{font-family:Georgia,serif;font-size:28px;font-weight:bold}.sub{font-size:14px;fill:#617078}.label{font-size:15px}.value{font-size:13px;font-weight:bold}.axis{font-size:12px;fill:#617078}.grid{stroke:#dfe5e1;stroke-width:1}</style>',
        '<text x="55" y="52" class="title">What happens when each feature group is removed?</text>',
        '<text x="55" y="82" class="sub">LightGBM retrained on 2021–2022 and evaluated on the untouched 2023 validation split</text>',
        f'<text x="{left}" y="125" class="axis">Average-precision loss from the full model</text>',
    ]
    for tick in range(5):
        value = max_drop * tick / 4
        x = left + chart_width * tick / 4
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + row_height * len(variants)}" class="grid"/>')
        pieces.append(f'<text x="{x:.1f}" y="{top - 18}" text-anchor="middle" class="axis">{value:.4f}</text>')
    for index, row in enumerate(variants):
        y = top + index * row_height + 14
        bar_width = max(0, row["average_precision_drop"]) / max_drop * chart_width
        color = "#173f5f" if row["variant"] == "full_features" else "#14807e"
        pieces.append(f'<text x="{left - 16}" y="{y + 17}" text-anchor="end" class="label">{row["display_name"]}</text>')
        pieces.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="26" rx="5" fill="{color}"/>')
        pieces.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 18}" class="value">{row["average_precision_drop"]:+.4f}</text>')

    capture_y = top + row_height * len(variants) + 85
    pieces.extend(
        [
            f'<line x1="55" y1="{capture_y - 38}" x2="1345" y2="{capture_y - 38}" stroke="#cfd7d3"/>',
            f'<text x="55" y="{capture_y}" class="title" style="font-size:22px">Top-5% capture reference</text>',
            f'<text x="55" y="{capture_y + 24}" class="sub">Share of positive cell-periods found in the citywide highest-risk 5%</text>',
        ]
    )
    capture_left, capture_width = 560, 720
    for tick in range(5):
        value = max_capture * tick / 4
        x = capture_left + capture_width * tick / 4
        pieces.append(f'<line x1="{x:.1f}" y1="{capture_y + 50}" x2="{x:.1f}" y2="{capture_y + 164}" class="grid"/>')
        pieces.append(f'<text x="{x:.1f}" y="{capture_y + 186}" text-anchor="middle" class="axis">{value:.0%}</text>')
    comparison = [
        ("Rolling baseline", references["baseline"]["top5_capture"], "#d66a2c"),
        ("Logistic regression", references["logistic"]["top5_capture"], "#6d5aa8"),
        ("Full LightGBM", next(row for row in variants if row["variant"] == "full_features")["top5_capture"], "#14807e"),
    ]
    for index, (label, value, color) in enumerate(comparison):
        y = capture_y + 56 + index * 34
        pieces.append(f'<text x="{capture_left - 14}" y="{y + 16}" text-anchor="end" class="label">{label}</text>')
        pieces.append(f'<rect x="{capture_left}" y="{y}" width="{value / max_capture * capture_width:.1f}" height="22" rx="4" fill="{color}"/>')
        pieces.append(f'<text x="{capture_left + value / max_capture * capture_width + 8:.1f}" y="{y + 16}" class="value">{value:.1%}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n")


def findings_markdown(rows: list[dict], path: Path) -> None:
    variants = {row["variant"]: row for row in rows}
    full = variants["full_features"]
    removals = [row for row in rows if row["variant"].startswith("no_")]
    ranked = sorted(removals, key=lambda row: row["average_precision_drop"], reverse=True)
    largest = ranked[0]
    roads = variants["no_roads_speed"]
    weather = variants["no_weather"]
    baseline = variants["baseline"]
    logistic = variants["logistic"]
    lines = [
        "# Feature-removal findings",
        "",
        "All results use 2021–2022 for training and 2023 for validation. The 2024 test file was not opened.",
        "",
        "## Main findings",
        "",
        f"1. **{largest['display_name']} caused the largest decline.** Average precision fell by {largest['average_precision_drop']:.4f}, from {full['average_precision']:.4f} to {largest['average_precision']:.4f}.",
        f"2. **Roads/speed and weather changed the score very little.** Removing roads/speed reduced average precision by only {roads['average_precision_drop']:.4f}; removing weather increased it by {-weather['average_precision_drop']:.4f}. This small change should be treated as negligible rather than evidence that weather prevents crashes.",
        f"3. **The full LightGBM model still improves only modestly over simpler approaches.** Its top-5% capture was {full['top5_capture']:.1%}, compared with {logistic['top5_capture']:.1%} for logistic regression and {baseline['top5_capture']:.1%} for the rolling baseline.",
        "",
        "These results describe predictive value, not cause and effect. A feature group can be useful for ranking without causing collisions.",
        "",
        "## Results",
        "",
        "| Variant | Average precision | Change from full | ROC AUC | Top-5% capture |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in [full, *ranked, baseline, logistic]:
        lines.append(
            f"| {row['display_name']} | {row['average_precision']:.4f} | {row['average_precision_drop']:+.4f} | {row['roc_auc']:.4f} | {row['top5_capture']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `blakely_feature_removal.csv`: shared team-results format",
            "- `feature_removal_detailed.csv`: additional metrics and runtime",
            "- `feature_removal_metrics.json`: complete reproducible settings and results",
            "- `feature_removal.svg`: presentation-ready chart",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--negative-sample-rate", type=float, default=0.1)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--recent-window-days", type=int, default=28)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.data_root / "data" / "processed"
    train_path = processed / TREE.LOGREG.TRAIN_FILE
    validation_path = processed / TREE.LOGREG.VALIDATION_FILE
    if "2024" in str(train_path) or "2024" in str(validation_path):
        raise RuntimeError("Feature-removal experiments must not open 2024 data")

    print("Loading 2021-2022 training and 2023 validation files...")
    train = TREE.load_split(train_path, "Training file")
    validation = TREE.load_split(validation_path, "Validation file")
    n_positive = int(validation[TREE.TARGET].sum())

    sets = feature_sets()
    results, _, _, _, _ = TREE.fit_and_evaluate(
        train,
        validation,
        candidates=TREE.CANDIDATES,
        feature_sets=sets,
        negative_rate=args.negative_sample_rate,
        burn_in_days=args.burn_in_days,
        prior_strength=args.prior_strength,
        recent_window_days=args.recent_window_days,
        seed=args.seed,
        top_fraction=args.top_fraction,
    )
    results["experiment"] = "feature_removal"
    results["split"] = "2023_validation"
    results["test_data_opened"] = False

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_removal_metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    shared = shared_rows(results, n_positive)
    write_csv(output / "blakely_feature_removal.csv", shared, SHARED_COLUMNS)

    references = reference_metrics(ROOT)
    detailed = detail_rows(results, references)
    write_csv(output / "feature_removal_detailed.csv", detailed, list(detailed[0]))
    svg_chart(detailed, output / "feature_removal.svg")
    findings_markdown(detailed, output / "FEATURE_REMOVAL_FINDINGS.md")

    print("\nFEATURE-REMOVAL RESULTS")
    for row in detailed:
        print(
            f"{row['display_name']:>26}: AP={row['average_precision']:.4f} "
            f"drop={row['average_precision_drop']:+.4f} "
            f"top5={row['top5_capture']:.1%}"
        )
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
