"""Create chronological train, validation, and test files for modeling.

The split keeps every NYC grid cell from a given six-hour period together:

* train:      2021-01-01 through 2022-12-31
* validation: 2023-01-01 through 2023-12-31
* test:       2024-01-01 through 2024-12-31

The 2024 test data should remain untouched until the team has selected its
features, model type, and model settings with the training and validation data.
"""

from pathlib import Path
from datetime import datetime
import argparse
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "train": {
        "start": datetime(2021, 1, 1),
        "end": datetime(2023, 1, 1),
        "filename": "model_panel_train_2021_2022.parquet",
        "purpose": "Fit model parameters.",
    },
    "validation": {
        "start": datetime(2023, 1, 1),
        "end": datetime(2024, 1, 1),
        "filename": "model_panel_validation_2023.parquet",
        "purpose": "Compare features, models, and settings.",
    },
    "test": {
        "start": datetime(2024, 1, 1),
        "end": datetime(2025, 1, 1),
        "filename": "model_panel_test_2024.parquet",
        "purpose": "Run once for the final reported evaluation.",
    },
}


def require_columns(schema, columns):
    missing = sorted(set(columns) - set(schema.names))
    if missing:
        raise ValueError(
            "Model panel is missing required columns: " + ", ".join(missing)
        )


def split_mask(times, start, end):
    """Return a mask for the half-open time range [start, end)."""
    start_value = pa.scalar(start, type=times.type)
    end_value = pa.scalar(end, type=times.type)
    return pc.and_(
        pc.greater_equal(times, start_value),
        pc.less(times, end_value),
    )


def empty_statistics(specification):
    return {
        "start": specification["start"].isoformat(),
        "end_exclusive": specification["end"].isoformat(),
        "purpose": specification["purpose"],
        "rows": 0,
        "crashes": 0,
        "rows_with_crashes": 0,
        "time_periods": set(),
        "cell_ids": set(),
        "minimum_time": None,
        "maximum_time": None,
    }


def update_statistics(statistics, batch):
    if batch.num_rows == 0:
        return

    crash_counts = batch.column("crash_count")
    times = batch.column("time_bin")
    statistics["rows"] += batch.num_rows
    statistics["crashes"] += int(pc.sum(crash_counts).as_py() or 0)
    positive = pc.cast(pc.greater(crash_counts, 0), pa.int64())
    statistics["rows_with_crashes"] += int(pc.sum(positive).as_py() or 0)
    statistics["time_periods"].update(pc.unique(times).to_pylist())
    statistics["cell_ids"].update(
        pc.unique(batch.column("cell_id")).to_pylist()
    )

    minimum = pc.min(times).as_py()
    maximum = pc.max(times).as_py()
    if statistics["minimum_time"] is None or minimum < statistics["minimum_time"]:
        statistics["minimum_time"] = minimum
    if statistics["maximum_time"] is None or maximum > statistics["maximum_time"]:
        statistics["maximum_time"] = maximum


def validate_statistics(statistics, all_cells):
    for name, values in statistics.items():
        if values["rows"] == 0:
            raise RuntimeError(f"The {name} split contains no rows")
        if values["cell_ids"] != all_cells:
            missing = len(all_cells - values["cell_ids"])
            extra = len(values["cell_ids"] - all_cells)
            raise RuntimeError(
                f"The {name} split has incomplete grid coverage "
                f"({missing} missing cells and {extra} unexpected cells)"
            )

        expected_rows = len(values["cell_ids"]) * len(values["time_periods"])
        if values["rows"] != expected_rows:
            raise RuntimeError(
                f"The {name} split has {values['rows']:,} rows; "
                f"expected {expected_rows:,} from its cells and time periods"
            )

        expected_start = SPLITS[name]["start"]
        expected_last = SPLITS[name]["end"]
        if values["minimum_time"] < expected_start:
            raise RuntimeError(f"The {name} split starts before its approved range")
        if values["maximum_time"] >= expected_last:
            raise RuntimeError(f"The {name} split ends after its approved range")


def serializable_statistics(statistics, output_paths):
    result = {}
    for name, values in statistics.items():
        result[name] = {
            "file": str(output_paths[name]),
            "start": values["start"],
            "end_exclusive": values["end_exclusive"],
            "purpose": values["purpose"],
            "rows": values["rows"],
            "crashes": values["crashes"],
            "rows_with_crashes": values["rows_with_crashes"],
            "zero_crash_rows": values["rows"] - values["rows_with_crashes"],
            "positive_row_rate": values["rows_with_crashes"] / values["rows"],
            "mean_crashes_per_row": values["crashes"] / values["rows"],
            "grid_cells": len(values["cell_ids"]),
            "time_periods": len(values["time_periods"]),
            "minimum_time": values["minimum_time"].isoformat(),
            "maximum_time": values["maximum_time"].isoformat(),
        }
    return result


def split_panel(input_path, output_directory, batch_size=500_000):
    """Split a model panel and return a JSON-ready validation summary."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"Model panel not found: {input_path}")

    output_directory.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(input_path)
    require_columns(parquet.schema_arrow, ["cell_id", "time_bin", "crash_count"])

    output_paths = {
        name: output_directory / specification["filename"]
        for name, specification in SPLITS.items()
    }
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp")
        for name, path in output_paths.items()
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    writers = {}
    statistics = {
        name: empty_statistics(specification)
        for name, specification in SPLITS.items()
    }
    all_cells = set()
    input_rows = 0
    input_crashes = 0
    assigned_rows = 0

    try:
        for batch_number, batch in enumerate(
            parquet.iter_batches(batch_size=batch_size), start=1
        ):
            input_rows += batch.num_rows
            input_crashes += int(pc.sum(batch.column("crash_count")).as_py() or 0)
            all_cells.update(pc.unique(batch.column("cell_id")).to_pylist())

            batch_assigned_rows = 0
            times = batch.column("time_bin")
            for name, specification in SPLITS.items():
                mask = split_mask(
                    times,
                    specification["start"],
                    specification["end"],
                )
                split_batch = batch.filter(mask)
                if split_batch.num_rows == 0:
                    continue

                if name not in writers:
                    writers[name] = pq.ParquetWriter(
                        temporary_paths[name],
                        split_batch.schema,
                        compression="zstd",
                        use_dictionary=["cell_id", "borough"],
                    )
                writers[name].write_batch(split_batch)
                update_statistics(statistics[name], split_batch)
                batch_assigned_rows += split_batch.num_rows

            if batch_assigned_rows != batch.num_rows:
                raise RuntimeError(
                    f"Batch {batch_number} assigned {batch_assigned_rows:,} of "
                    f"{batch.num_rows:,} rows. Check the split date ranges."
                )
            assigned_rows += batch_assigned_rows
            print(
                f"Processed batch {batch_number}: "
                f"{input_rows:,} of {parquet.metadata.num_rows:,} rows"
            )
    except Exception:
        for writer in writers.values():
            writer.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for writer in writers.values():
            writer.close()

    try:
        validate_statistics(statistics, all_cells)
        output_rows = sum(values["rows"] for values in statistics.values())
        output_crashes = sum(values["crashes"] for values in statistics.values())
        if input_rows != parquet.metadata.num_rows or assigned_rows != input_rows:
            raise RuntimeError("The split did not preserve the model panel row count")
        if output_rows != input_rows:
            raise RuntimeError("The split outputs do not add up to the input row count")
        if output_crashes != input_crashes:
            raise RuntimeError("The split outputs do not preserve the crash total")

        for name, temporary_path in temporary_paths.items():
            temporary_path.replace(output_paths[name])
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    split_summary = serializable_statistics(statistics, output_paths)
    return {
        "input_file": str(input_path),
        "split_rule": "Whole calendar years; all cells from each time period stay together.",
        "final_test_warning": (
            "Do not use the 2024 test file to choose features, models, thresholds, "
            "or settings. Use it once after development is complete."
        ),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "input_crashes": input_crashes,
        "output_crashes": output_crashes,
        "grid_cells": len(all_cells),
        "splits": split_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    input_path = args.input or processed / "model_panel_v1.parquet"
    output_directory = args.output_directory or processed
    summary_path = output_directory / "model_panel_split_summary.json"

    summary = split_panel(input_path, output_directory, args.batch_size)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nMODEL PANEL SPLIT SUMMARY")
    for name, values in summary["splits"].items():
        print(
            f"{name}: {values['rows']:,} rows, "
            f"{values['crashes']:,} crashes, "
            f"{values['time_periods']:,} periods"
        )
        print(f"  {values['file']}")
    print(f"Validation summary: {summary_path}")
    print(summary["final_test_warning"])


if __name__ == "__main__":
    main()
