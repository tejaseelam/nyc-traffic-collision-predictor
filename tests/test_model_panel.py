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
    "model_panel",
    ROOT / "scripts" / "6build_model_panel.py",
)
MODEL_PANEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL_PANEL)


class ModelPanelTests(unittest.TestCase):
    def test_cell_ids_are_stable(self):
        result = MODEL_PANEL.make_cell_ids(
            np.array([1000, 1001]),
            np.array([2000, 2000]),
        )
        self.assertEqual(result.tolist(), ["NYC_1000_2000", "NYC_1001_2000"])

    def test_crash_times_use_six_hour_periods(self):
        values = pd.Series([
            "2024-01-01 00:00",
            "2024-01-01 05:59",
            "2024-01-01 06:00",
            "2024-01-01 23:59",
        ])
        result = MODEL_PANEL.bin_crash_times(values, 6)
        self.assertEqual(
            result.astype(str).tolist(),
            [
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",
                "2024-01-01 06:00:00",
                "2024-01-01 18:00:00",
            ],
        )

    def test_one_square_kilometer_produces_four_cells(self):
        boroughs = gpd.GeoDataFrame(
            {"boroname": ["TEST"]},
            geometry=[shapely.box(500000, 4500000, 501000, 4501000)],
            crs=MODEL_PANEL.METRIC_CRS,
        )
        grid = MODEL_PANEL.build_grid(
            boroughs,
            grid_size_m=500,
            minimum_land_fraction=0.01,
        )
        self.assertEqual(len(grid), 4)
        self.assertTrue((grid["borough"] == "TEST").all())
        self.assertTrue(np.allclose(grid["land_fraction"], 1.0))

    def test_panel_includes_zero_crash_rows_and_preserves_counts(self):
        grid = pd.DataFrame({
            "cell_id": ["NYC_1000_2000", "NYC_1001_2000"],
            "grid_x": [1000, 1001],
            "grid_y": [2000, 2000],
            "borough": ["TEST", "TEST"],
            "land_area_m2": [250000.0, 250000.0],
            "land_fraction": [1.0, 1.0],
            "speed_road_length_m": [100.0, 200.0],
            "speed_road_segments": [1, 2],
            "speed_known_length_m": [100.0, 200.0],
            "speed_limit_min_mph": [25, 25],
            "speed_limit_max_mph": [25, 30],
            "speed_limit_weighted_avg_mph": [25.0, 27.5],
            "speed_limit_coverage": [1.0, 1.0],
        })
        periods = pd.date_range("2024-01-01", periods=2, freq="6h")
        crash_counts = pd.DataFrame({
            "cell_id": ["NYC_1000_2000"],
            "time_bin": [periods[0]],
            "crash_count": [2],
        })
        weather = pd.DataFrame({
            "weather_date": [pd.Timestamp("2024-01-01")],
            "weather_temp_midpoint_c": [5.0],
            "weather_precip_mm": [0.0],
            "weather_snowfall_mm": [0.0],
            "weather_snow_depth_mm": [0.0],
            "weather_wind_mps": [2.0],
            "weather_precip_day": [False],
            "weather_snow_day": [False],
            "weather_any_missing": [False],
        })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "panel.parquet"
            summary = MODEL_PANEL.build_panel(
                grid,
                periods,
                crash_counts,
                weather,
                output,
                chunk_periods=1,
            )
            panel = pd.read_parquet(output)

        self.assertEqual(len(panel), 4)
        self.assertEqual(int(panel["crash_count"].sum()), 2)
        self.assertEqual(int(panel["has_crash"].sum()), 1)
        self.assertEqual(summary["zero_crash_rows"], 3)


if __name__ == "__main__":
    unittest.main()
