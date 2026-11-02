import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "8baseline_logreg.py"
SPEC = importlib.util.spec_from_file_location("baseline_logreg", SCRIPT)
LOGREG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGREG)


def make_panel(start, end, cell_count=12, seed=0):
    """Small complete panel where busier cells and evening hours crash more."""
    random = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="6h", inclusive="left")
    cells = [f"NYC_{index}_0" for index in range(cell_count)]
    cell_risk = np.linspace(0.02, 0.4, cell_count)
    hour_risk = {0: 0.5, 6: 0.8, 12: 1.0, 18: 1.4}

    time_values = np.repeat(times, cell_count)
    probability = np.tile(cell_risk, len(times)) * np.repeat(
        [hour_risk[time.hour] for time in times], cell_count
    )
    crashes = random.random(len(probability)) < probability
    panel = pd.DataFrame({
        "cell_id": np.tile(cells, len(times)),
        "time_bin": time_values,
        "has_crash": crashes,
        "crash_count": crashes.astype(np.int32),
        "borough": np.tile(["MANHATTAN", "BROOKLYN"], cell_count // 2 * len(times)),
        "hour": time_values.hour.astype(np.int8),
        "day_of_week": time_values.day_of_week.astype(np.int8),
        "month": time_values.month.astype(np.int8),
        "land_fraction": pd.array(np.tile(np.linspace(0.5, 1, cell_count), len(times)), dtype="Float32"),
        "speed_road_length_m": pd.array(np.tile(np.linspace(0, 900, cell_count), len(times)), dtype="Float32"),
        "speed_road_segments": np.tile(np.arange(cell_count), len(times)).astype(np.int32),
        "speed_limit_weighted_avg_mph": pd.array([25.0, None] * (cell_count // 2) * len(times), dtype="Float32"),
        "speed_limit_coverage": pd.array([1.0, None] * (cell_count // 2) * len(times), dtype="Float32"),
        "weather_temp_midpoint_c": pd.array(random.normal(12, 8, len(probability)), dtype="Float32"),
        "weather_precip_mm": pd.array(random.exponential(2, len(probability)), dtype="Float32"),
        "weather_snowfall_mm": pd.array(np.zeros(len(probability)), dtype="Float32"),
        "weather_snow_depth_mm": pd.array(np.zeros(len(probability)), dtype="Float32"),
        "weather_wind_mps": pd.array(random.uniform(0, 8, len(probability)), dtype="Float32"),
        "weather_precip_day": pd.array(random.random(len(probability)) < 0.3, dtype="boolean"),
        "weather_snow_day": pd.array(np.zeros(len(probability), dtype=bool), dtype="boolean"),
        "weather_any_missing": pd.array([False] * len(probability), dtype="boolean"),
    })
    return panel


class HistoryFeatureTests(unittest.TestCase):
    def test_expanding_rate_uses_only_earlier_periods(self):
        outcomes = np.zeros((6, 3), dtype=np.float32)
        changed = outcomes.copy()
        changed[3, 1] = 1

        original = LOGREG.expanding_rate(outcomes, prior_strength=2)
        updated = LOGREG.expanding_rate(changed, prior_strength=2)

        np.testing.assert_array_equal(original[:4], updated[:4])
        self.assertTrue((updated[4:, 1] > original[4:, 1]).all())

    def test_recent_rate_only_counts_the_window(self):
        outcomes = np.array([[1], [0], [0], [0]], dtype=np.float32)
        rate = LOGREG.recent_rate(outcomes, window=2, prior_strength=0)
        # Period 1 sees period 0; period 3 sees only periods 1 and 2.
        self.assertEqual(rate[1, 0], 1.0)
        self.assertEqual(rate[3, 0], 0.0)

    def test_history_features_span_splits_without_using_current_label(self):
        train = LOGREG.prepare_panel(make_panel("2021-01-01", "2021-01-08"), "train")
        validation = LOGREG.prepare_panel(make_panel("2021-01-08", "2021-01-10", seed=1), "validation")
        flipped = validation.copy()
        flipped["has_crash"] = ~flipped["has_crash"]

        _, original = LOGREG.add_history_features([train, validation], 5, 2)
        _, changed = LOGREG.add_history_features([train, flipped], 5, 2)

        first_period = validation["time_bin"] == validation["time_bin"].min()
        for column in ["history_cell_logit", "history_cell_hour_logit", "history_recent_logit"]:
            np.testing.assert_array_equal(
                original.loc[first_period, column], changed.loc[first_period, column]
            )
            self.assertFalse(np.allclose(
                original.loc[~first_period, column], changed.loc[~first_period, column]
            ))

    def test_history_features_reject_missing_cell_periods(self):
        panel = LOGREG.prepare_panel(make_panel("2021-01-01", "2021-01-03"), "train")
        with self.assertRaisesRegex(ValueError, "gaps"):
            LOGREG.add_history_features([panel.iloc[1:]], 5, 1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = LOGREG.prepare_panel(make_panel("2021-01-01", "2021-07-01"), "train")
        cls.validation = LOGREG.prepare_panel(
            make_panel("2021-07-01", "2021-10-01", seed=1), "validation"
        )
        cls.results, cls.model = LOGREG.fit_and_evaluate(
            cls.train,
            cls.validation,
            c_values=(0.1, 1.0),
            negative_rate=0.5,
            burn_in_days=14,
            prior_strength=5,
            recent_window_days=14,
        )

    def test_selects_a_logistic_model_by_validation_average_precision(self):
        logistic = self.results["logistic"]
        selected = self.results["selected_logistic"]
        self.assertIn(selected, logistic)
        self.assertEqual(
            logistic[selected]["average_precision"],
            max(metrics["average_precision"] for metrics in logistic.values()),
        )

    def test_models_beat_random_ranking(self):
        for metrics in [
            *self.results["baselines"].values(),
            *self.results["logistic"].values(),
        ]:
            self.assertGreater(metrics["roc_auc"], 0.7)
            self.assertGreater(metrics["average_precision"], metrics["positive_rate"])

    def test_negative_sampling_correction_keeps_probabilities_calibrated(self):
        selected = self.results["logistic"][self.results["selected_logistic"]]
        self.assertAlmostEqual(
            selected["mean_predicted_probability"], selected["positive_rate"], delta=0.02
        )

    def test_coefficient_table_names_every_feature(self):
        table = LOGREG.coefficient_table(self.model)
        self.assertTrue(table["feature"].str.contains("history_cell_hour_logit").any())
        self.assertTrue(table["feature"].str.contains("borough_MANHATTAN").any())


class CommandLineTests(unittest.TestCase):
    def test_script_runs_without_a_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            make_panel("2021-01-01", "2021-03-01").to_parquet(processed / LOGREG.TRAIN_FILE)
            make_panel("2021-03-01", "2021-04-01", seed=1).to_parquet(
                processed / LOGREG.VALIDATION_FILE
            )

            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--root", str(root),
                    "--c-values", "1",
                    "--negative-sample-rate", "0.5",
                    "--burn-in-days", "7",
                ],
                check=True,
                capture_output=True,
            )

            metrics = json.loads(
                (root / "reports" / "baseline_logreg" / "validation_metrics.json").read_text()
            )
            self.assertEqual(metrics["selected_logistic"], "C=1")
            self.assertTrue((root / "reports" / "baseline_logreg" / "coefficients.csv").is_file())
            self.assertTrue((processed / "models" / "logreg_has_crash.joblib").is_file())


if __name__ == "__main__":
    unittest.main()
