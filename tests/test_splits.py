import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "split_model_panel",
    ROOT / "scripts" / "7split_model_panel.py",
)
SPLITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLITTER)


class ModelPanelSplitTests(unittest.TestCase):
    def test_split_mask_uses_half_open_date_range(self):
        times = pa.array(
            [
                pd.Timestamp("2022-12-31 18:00"),
                pd.Timestamp("2023-01-01 00:00"),
                pd.Timestamp("2023-12-31 18:00"),
                pd.Timestamp("2024-01-01 00:00"),
            ],
            type=pa.timestamp("ns"),
        )
        mask = SPLITTER.split_mask(
            times,
            SPLITTER.SPLITS["validation"]["start"],
            SPLITTER.SPLITS["validation"]["end"],
        )
        self.assertEqual(mask.to_pylist(), [False, True, True, False])

    def test_split_preserves_rows_crashes_and_complete_cells(self):
        cells = ["NYC_1_1", "NYC_2_1"]
        times = [
            pd.Timestamp("2021-01-01 00:00"),
            pd.Timestamp("2022-12-31 18:00"),
            pd.Timestamp("2023-01-01 00:00"),
            pd.Timestamp("2023-12-31 18:00"),
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-12-31 18:00"),
        ]
        rows = []
        for time_index, time in enumerate(times):
            for cell_index, cell in enumerate(cells):
                rows.append({
                    "cell_id": cell,
                    "time_bin": time,
                    "borough": "TEST",
                    "crash_count": int(time_index == cell_index),
                    "has_crash": time_index == cell_index,
                })
        panel = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "panel.parquet"
            pq.write_table(pa.Table.from_pandas(panel, preserve_index=False), input_path)
            summary = SPLITTER.split_panel(
                input_path,
                directory / "outputs",
                batch_size=3,
            )

            outputs = {
                name: pd.read_parquet(values["file"])
                for name, values in summary["splits"].items()
            }

        self.assertEqual(summary["input_rows"], len(panel))
        self.assertEqual(summary["output_rows"], len(panel))
        self.assertEqual(summary["input_crashes"], int(panel["crash_count"].sum()))
        self.assertEqual(summary["input_crashes"], summary["output_crashes"])
        self.assertEqual(len(outputs["train"]), 4)
        self.assertEqual(len(outputs["validation"]), 4)
        self.assertEqual(len(outputs["test"]), 4)
        self.assertTrue((outputs["train"]["time_bin"].dt.year <= 2022).all())
        self.assertTrue((outputs["validation"]["time_bin"].dt.year == 2023).all())
        self.assertTrue((outputs["test"]["time_bin"].dt.year == 2024).all())
        for output in outputs.values():
            self.assertEqual(set(output["cell_id"]), set(cells))


if __name__ == "__main__":
    unittest.main()
