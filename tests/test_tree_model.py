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
SCRIPT = ROOT / "scripts" / "10tree_model.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREE = load_module("tree_model", SCRIPT)
LOGREG_TESTS = load_module("test_baseline_logreg", ROOT / "tests" / "test_baseline_logreg.py")

FAST_CANDIDATES = {
    "small": {"learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 20},
    "smaller": {"learning_rate": 0.1, "max_leaf_nodes": 3, "min_samples_leaf": 50},
}


def make_panel(start, end, cell_count=12, seed=0):
    """The logistic tests' panel plus each cell's grid position."""
    panel = LOGREG_TESTS.make_panel(start, end, cell_count, seed)
    cell_number = panel["cell_id"].str.split("_").str[1].astype(np.int32)
    panel["grid_x"] = cell_number
    panel["grid_y"] = np.int32(0)
    return panel


def prepare(panel, label):
    panel = panel.copy()
    prepared = TREE.LOGREG.prepare_panel(panel, label)
    for column in TREE.LOCATION_COLUMNS:
        prepared[column] = panel[column].to_numpy(dtype=np.float32)
    return prepared


class FeatureTests(unittest.TestCase):
    def test_core_features_match_the_logistic_regression(self):
        self.assertEqual(TREE.FEATURE_SETS["core"], TREE.LOGREG.FEATURES)

    def test_every_feature_belongs_to_exactly_one_group(self):
        grouped = [column for columns in TREE.FEATURE_GROUPS.values() for column in columns]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(TREE.FEATURE_SETS["core_plus_location"]), set(grouped))

    def test_model_inputs_use_training_category_levels(self):
        panel = prepare(make_panel("2021-01-01", "2021-01-03"), "train")
        levels = TREE.category_levels(panel)
        later = panel.copy()
        later["hour"] = later["hour"].where(later["hour"] != 18, 3)
        inputs = TREE.model_inputs(later, TREE.CATEGORICAL_FEATURES, levels)
        self.assertEqual(list(inputs["hour"].cat.categories), [0, 6, 12, 18])
        # A level never seen in training becomes missing instead of a new category.
        self.assertTrue(inputs.loc[later["hour"] == 3, "hour"].isna().all())


class PermutationImportanceTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(0)
        signal = random.random(5000)
        self.frame = pd.DataFrame({
            TREE.TARGET: random.random(5000) < signal * 0.4,
            "signal": signal,
            "signal_copy": signal,
            "noise": random.random(5000),
        })

    def test_only_informative_columns_lose_average_precision(self):
        table, reference = TREE.permutation_importance(
            lambda frame: frame["signal"].to_numpy(),
            self.frame,
            {"signal": ["signal"], "noise": ["noise"]},
            repeats=2,
            seed=0,
        )
        drops = table.set_index("name")["average_precision_drop"]
        self.assertGreater(drops["signal"], 0.1)
        self.assertEqual(drops["noise"], 0)
        self.assertGreater(reference, self.frame[TREE.TARGET].mean())

    def test_grouped_columns_are_shuffled_together(self):
        def score(frame):
            np.testing.assert_array_equal(frame["signal"], frame["signal_copy"])
            return frame["signal"].to_numpy()

        TREE.permutation_importance(
            score, self.frame, {"pair": ["signal", "signal_copy"]}, repeats=2, seed=0
        )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = prepare(make_panel("2021-01-01", "2021-07-01"), "train")
        validation = prepare(make_panel("2021-07-01", "2021-10-01", seed=1), "validation")
        (
            cls.results, cls.models, cls.levels, cls.validation, cls.predictions
        ) = TREE.fit_and_evaluate(
            train,
            validation,
            candidates=FAST_CANDIDATES,
            negative_rate=0.5,
            burn_in_days=14,
            prior_strength=5,
            recent_window_days=14,
        )

    def test_selects_each_feature_set_by_validation_average_precision(self):
        for set_name in TREE.FEATURE_SETS:
            candidates = self.results["tree"][set_name]
            selected = self.results["selected"][set_name]
            self.assertEqual(
                candidates[selected]["average_precision"],
                max(metrics["average_precision"] for metrics in candidates.values()),
            )
            self.assertIn(f"tree_{set_name}", self.predictions)

    def test_trees_beat_random_ranking(self):
        for candidates in self.results["tree"].values():
            for metrics in candidates.values():
                self.assertGreater(metrics["roc_auc"], 0.7)
                self.assertGreater(metrics["average_precision"], metrics["positive_rate"])

    def test_records_settings_and_runtime_for_every_candidate(self):
        for candidates in self.results["tree"].values():
            for metrics in candidates.values():
                self.assertIn("max_leaf_nodes", metrics["settings"])
                self.assertGreaterEqual(metrics["fit_seconds"], 0)
                self.assertGreater(metrics["iterations"], 0)

    def test_negative_sampling_correction_keeps_probabilities_calibrated(self):
        selected = self.results["tree"]["core"][self.results["selected"]["core"]]
        self.assertAlmostEqual(
            selected["mean_predicted_probability"], selected["positive_rate"], delta=0.02
        )

    def test_crash_history_matters_and_weather_does_not(self):
        # The synthetic crashes depend on the cell and time of day only.
        features = TREE.FEATURE_SETS["core"]
        table = TREE.importance_tables(
            TREE.tree_scorer(self.models["core"], features, self.levels, 0.5),
            features, self.validation, repeats=2, seed=0, model_name="tree_core",
        )
        groups = table[table["level"] == "group"].set_index("name")["average_precision_drop"]
        self.assertGreater(groups["Crash history"], groups["Weather"])
        self.assertEqual(set(table["level"]), {"group", "feature"})


class CommandLineTests(unittest.TestCase):
    def test_script_runs_without_a_test_file_or_logistic_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            make_panel("2021-01-01", "2021-03-01").to_parquet(
                processed / TREE.LOGREG.TRAIN_FILE
            )
            make_panel("2021-03-01", "2021-04-01", seed=1).to_parquet(
                processed / TREE.LOGREG.VALIDATION_FILE
            )

            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--root", str(root),
                    "--candidates", "shallow",
                    "--negative-sample-rate", "0.5",
                    "--burn-in-days", "7",
                    "--importance-rows", "500",
                    "--importance-repeats", "1",
                ],
                check=True,
                capture_output=True,
            )

            report = root / "reports" / "tree_model"
            metrics = json.loads((report / "validation_metrics.json").read_text())
            self.assertEqual(metrics["selected"], {"core": "shallow", "core_plus_location": "shallow"})
            self.assertIn("total_seconds", metrics["runtime"])
            importance = pd.read_csv(report / "feature_importance.csv")
            self.assertEqual(set(importance["model"]), {"tree_core", "tree_core_plus_location"})
            self.assertTrue((report / "model_comparison.csv").is_file())
            self.assertTrue((processed / "models" / "tree_has_crash.joblib").is_file())


if __name__ == "__main__":
    unittest.main()
