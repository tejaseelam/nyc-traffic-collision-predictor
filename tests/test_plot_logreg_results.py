import importlib.util
from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_logreg_results", ROOT / "scripts" / "9plot_logreg_results.py"
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
        "cell_rate": risk + random.normal(0, 0.02, len(risk)).clip(-0.009),
        "cell_hour_rate": risk,
        "cell_hour_rate_rolling": risk,
        "logistic": risk,
    })


class PlotTests(unittest.TestCase):
    def test_capture_curve_reaches_all_crashes_when_every_row_is_flagged(self):
        y_true = np.array([True, False, True, False])
        flagged, captured = PLOTS.capture_curve(y_true, np.array([0.9, 0.8, 0.1, 0.2]), 1.0)
        self.assertEqual(flagged[-1], 1.0)
        self.assertEqual(captured[-1], 1.0)
        # The top-ranked row is a crash, so flagging 25% catches half of them.
        self.assertEqual(captured[1], 0.5)

    def test_readable_feature_names(self):
        self.assertEqual(PLOTS.readable_feature("categorical__hour_18"), "18:00-24:00")
        self.assertEqual(PLOTS.readable_feature("categorical__day_of_week_0"), "Mon")
        self.assertEqual(PLOTS.readable_feature("categorical__borough_STATEN ISLAND"), "Staten Island")
        self.assertEqual(
            PLOTS.readable_feature("numeric__missingindicator_speed_limit_weighted_avg_mph"),
            "Average speed limit missing",
        )

    def test_plot_all_writes_every_figure(self):
        predictions = make_predictions()
        coefficients = pd.DataFrame({
            "feature": ["numeric__history_cell_logit", "categorical__hour_18"],
            "coefficient": [0.9, -0.2],
        })
        coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
        grid = gpd.GeoDataFrame(
            {"cell_id": [f"NYC_{i}_0" for i in range(16)]},
            geometry=shapely.box(
                580000 + np.arange(16) % 4 * 500, 4500000 + np.arange(16) // 4 * 500,
                580500 + np.arange(16) % 4 * 500, 4500500 + np.arange(16) // 4 * 500,
            ),
            crs="EPSG:32618",
        ).to_crs("EPSG:4326")

        with tempfile.TemporaryDirectory() as directory:
            paths = PLOTS.plot_all(predictions, coefficients, grid, Path(directory))
            names = sorted(path.name for path in paths)
            self.assertEqual(names, [
                "calibration.png", "capture_curve.png", "coefficients.png",
                "precision_recall.png", "risk_map.png",
            ])
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_risk_map_is_skipped_without_a_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PLOTS.plot_all(make_predictions(), None, None, Path(directory))
            self.assertNotIn("risk_map.png", [path.name for path in paths])


if __name__ == "__main__":
    unittest.main()
