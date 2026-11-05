import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "12boosting_libraries.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOOSTING = load_module("boosting_libraries", SCRIPT)
TREE_TESTS = load_module("test_tree_model", ROOT / "tests" / "test_tree_model.py")


class SettingsTests(unittest.TestCase):
    def test_depth_allows_at_least_as_many_leaves(self):
        for leaves, depth in [(3, 2), (7, 3), (31, 5), (63, 6), (127, 7)]:
            self.assertEqual(BOOSTING.depth_for_leaves(leaves), depth)

    def test_libraries_share_the_scikit_learn_candidates_and_features(self):
        self.assertIs(BOOSTING.CANDIDATES, BOOSTING.TREE.CANDIDATES)
        self.assertIs(BOOSTING.FEATURE_SETS, BOOSTING.TREE.FEATURE_SETS)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = TREE_TESTS.prepare(TREE_TESTS.make_panel("2021-01-01", "2021-07-01"), "train")
        validation = TREE_TESTS.prepare(
            TREE_TESTS.make_panel("2021-07-01", "2021-10-01", seed=1), "validation"
        )
        cls.results, cls.models, cls.levels, cls.predictions = BOOSTING.fit_and_evaluate(
            train,
            validation,
            candidates=TREE_TESTS.FAST_CANDIDATES,
            negative_rate=0.5,
            burn_in_days=14,
            prior_strength=5,
            recent_window_days=14,
        )

    def candidate_metrics(self):
        for library in BOOSTING.LIBRARIES:
            for candidates in self.results[library].values():
                yield from candidates.values()

    def test_selects_each_model_by_validation_average_precision(self):
        for library in BOOSTING.LIBRARIES:
            for set_name in BOOSTING.FEATURE_SETS:
                candidates = self.results[library][set_name]
                selected = self.results["selected"][library][set_name]
                self.assertEqual(
                    candidates[selected]["average_precision"],
                    max(metrics["average_precision"] for metrics in candidates.values()),
                )
                self.assertIn(f"{library}_{set_name}", self.predictions)

    def test_both_libraries_beat_random_ranking(self):
        for metrics in self.candidate_metrics():
            self.assertGreater(metrics["roc_auc"], 0.7)
            self.assertGreater(metrics["average_precision"], metrics["positive_rate"])

    def test_early_stopping_keeps_fewer_trees_than_the_maximum(self):
        for metrics in self.candidate_metrics():
            self.assertGreater(metrics["iterations"], 0)
            self.assertLess(metrics["iterations"], BOOSTING.MAX_ROUNDS)

    def test_negative_sampling_correction_keeps_probabilities_calibrated(self):
        for library in BOOSTING.LIBRARIES:
            selected = self.results["selected"][library]["core"]
            metrics = self.results[library]["core"][selected]
            self.assertAlmostEqual(
                metrics["mean_predicted_probability"], metrics["positive_rate"], delta=0.02
            )

    def test_comparison_lists_saved_scikit_learn_trees_first(self):
        tree_metrics = {
            "selected": {"core": "shallow"},
            "tree": {"core": {"shallow": self.results["lightgbm"]["core"]["small"]}},
        }
        table = BOOSTING.comparison_table(self.results, tree_metrics, None)
        self.assertEqual(table["model"].iloc[0], "sklearn core (shallow)")
        self.assertEqual(len(table), 1 + len(BOOSTING.LIBRARIES) * len(BOOSTING.FEATURE_SETS))


class CommandLineTests(unittest.TestCase):
    def test_script_runs_without_a_test_file_or_earlier_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            TREE_TESTS.make_panel("2021-01-01", "2021-03-01").to_parquet(
                processed / BOOSTING.LOGREG.TRAIN_FILE
            )
            TREE_TESTS.make_panel("2021-03-01", "2021-04-01", seed=1).to_parquet(
                processed / BOOSTING.LOGREG.VALIDATION_FILE
            )

            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--root", str(root),
                    "--candidates", "shallow",
                    "--negative-sample-rate", "0.5",
                    "--burn-in-days", "7",
                ],
                check=True,
                capture_output=True,
            )

            report = root / "reports" / "boosting_libraries"
            metrics = json.loads((report / "validation_metrics.json").read_text())
            self.assertEqual(
                metrics["selected"],
                {
                    library: {"core": "shallow", "core_plus_location": "shallow"}
                    for library in BOOSTING.LIBRARIES
                },
            )
            self.assertIn("lightgbm", metrics["runtime"])
            comparison = pd.read_csv(report / "model_comparison.csv")
            self.assertEqual(len(comparison), 4)
            self.assertTrue((processed / "models" / "boosting_has_crash.joblib").is_file())


if __name__ == "__main__":
    unittest.main()
