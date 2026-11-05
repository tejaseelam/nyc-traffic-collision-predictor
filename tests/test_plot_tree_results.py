import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_tree_results", ROOT / "scripts" / "11plot_tree_results.py"
)
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


def make_predictions(cell_count=16, period_count=200, seed=0):
    random = np.random.default_rng(seed)
    risk = np.tile(np.linspace(0.01, 0.3, cell_count), period_count)
    return pd.DataFrame({
        "cell_id": np.tile([f"NYC_{i}_0" for i in range(cell_count)], period_count),
        "time_bin": np.repeat(pd.date_range("2023-01-01", periods=period_count, freq="6h"), cell_count),
        "has_crash": random.random(len(risk)) < risk,
        "tree_core": risk,
        "tree_core_plus_location": risk,
    })


def make_importance(models):
    rows = []
    for model in models:
        for group in ["Crash history", "Time", "Location", "Roads and speed limits", "Weather"]:
            rows.append({"model": model, "level": "group", "name": group, "relative_drop": 0.1})
        for feature in ["history_cell_logit", "grid_x", "hour"]:
            rows.append({"model": model, "level": "feature", "name": feature, "relative_drop": 0.05})
    return pd.DataFrame(rows)


class PlotTests(unittest.TestCase):
    def test_writes_every_figure_with_the_logistic_model(self):
        predictions = make_predictions()
        predictions["logistic"] = predictions["tree_core"]
        predictions["cell_hour_rate_rolling"] = predictions["tree_core"]
        importance = make_importance(["tree_core", "tree_core_plus_location", "logistic"])
        with tempfile.TemporaryDirectory() as directory:
            paths = PLOTS.plot_all(predictions, importance, Path(directory))
            self.assertEqual(
                sorted(path.name for path in paths),
                sorted([
                    "feature_importance.png",
                    "feature_importance_detail.png",
                    "capture_curve.png",
                    "precision_recall.png",
                    "calibration.png",
                ]),
            )
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)

    def test_plots_the_tree_alone(self):
        importance = make_importance(["tree_core", "tree_core_plus_location"])
        with tempfile.TemporaryDirectory() as directory:
            paths = PLOTS.plot_all(make_predictions(), importance, Path(directory))
            self.assertEqual(len(paths), 5)

    def test_merges_logistic_predictions_by_cell_and_period(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory)
            (processed / "predictions").mkdir()
            tree = make_predictions()
            tree.to_parquet(processed / "predictions" / PLOTS.TREE_PREDICTIONS_FILE)
            logreg = tree[["cell_id", "time_bin"]].iloc[::-1].copy()
            logreg["logistic"] = np.arange(len(logreg), dtype=float)
            logreg["cell_hour_rate_rolling"] = 0.1
            logreg.to_parquet(processed / "predictions" / PLOTS.LOGREG_PREDICTIONS_FILE)

            merged = PLOTS.load_predictions(processed)
            self.assertEqual(len(merged), len(tree))
            # Rows were reversed in the logistic file, so the merge must use keys.
            self.assertEqual(merged["logistic"].iloc[0], len(tree) - 1)


if __name__ == "__main__":
    unittest.main()
