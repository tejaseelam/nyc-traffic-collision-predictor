from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = json.loads((DATA / "grid.geojson").read_text())
        cls.summary = json.loads((DATA / "cell_period_summary.json").read_text())
        cls.detail = json.loads((DATA / "cell_detail.json").read_text())
        cls.metrics = json.loads((DATA / "metrics.json").read_text())

    def test_public_data_contains_the_complete_grid(self):
        self.assertEqual(len(self.grid["features"]), 3_640)
        self.assertEqual(len(self.summary["cell_ids"]), 3_640)
        self.assertEqual(len(self.detail["monthly"]), 3_640)

    def test_public_results_are_validation_only(self):
        self.assertEqual(self.summary["meta"]["year"], 2023)
        self.assertEqual(self.summary["meta"]["split"], "validation")
        self.assertEqual(self.metrics["validation"]["year"], 2023)
        self.assertEqual(self.detail["months"][-1], "2023-12")

    def test_every_filter_slice_aligns_with_the_grid(self):
        for weather in ("all", "wet", "dry"):
            self.assertEqual(len(self.summary["slices"][weather]), 52)
            for values in self.summary["slices"][weather].values():
                for field in ("o", "b", "l", "t", "c"):
                    self.assertEqual(len(values[field]), 3_640)

    def test_tree_metrics_match_the_validation_report(self):
        models = {row["id"]: row for row in self.metrics["models"]}
        self.assertAlmostEqual(models["tree"]["average_precision"], 0.06079, places=4)
        self.assertAlmostEqual(models["tree"]["top5_capture"], 0.23273, places=4)
        self.assertGreater(
            models["tree"]["average_precision"],
            models["logistic"]["average_precision"],
        )

    def test_public_payload_stays_under_fifteen_megabytes(self):
        total = sum(path.stat().st_size for path in DATA.iterdir() if path.is_file())
        self.assertLess(total, 15_000_000)


if __name__ == "__main__":
    unittest.main()
