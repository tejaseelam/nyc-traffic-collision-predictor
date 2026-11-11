import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "12compare_models.py"
SPEC = importlib.util.spec_from_file_location("compare_models", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def make_predictions(days=20, cells=30, seed=0):
    """Two prediction files for the same 2023 rows, with risk that varies by cell."""
    random = np.random.default_rng(seed)
    times = pd.date_range("2023-01-01", periods=days * 4, freq="6h")
    cell_ids = [f"NYC_{index}_0" for index in range(cells)]
    risk = np.tile(np.linspace(0.01, 0.3, cells), len(times))
    crashes = random.random(len(risk)) < risk
    keys = pd.DataFrame({
        "cell_id": np.tile(cell_ids, len(times)),
        "time_bin": np.repeat(times, cells),
        "has_crash": crashes,
    })

    def noisy(scale):
        return np.clip(risk * np.exp(random.normal(0, scale, len(risk))), 1e-4, 0.99)

    logreg = keys.assign(cell_hour_rate_rolling=noisy(0.6), logistic=noisy(0.4))
    tree = keys.assign(tree_core=noisy(0.3), tree_core_plus_location=noisy(0.2))
    return logreg, tree


class AlignmentTests(unittest.TestCase):
    def test_rows_are_matched_by_cell_and_period(self):
        logreg, tree = make_predictions()
        merged = COMPARE.align(logreg, tree.sample(frac=1, random_state=1))
        self.assertEqual(len(merged), len(logreg))

    def test_rejects_files_that_disagree_on_the_outcome(self):
        logreg, tree = make_predictions()
        tree.loc[0, "has_crash"] = not tree.loc[0, "has_crash"]
        with self.assertRaisesRegex(ValueError, "disagree"):
            COMPARE.align(logreg, tree)

    def test_rejects_rows_outside_2023(self):
        logreg, tree = make_predictions()
        for frame in (logreg, tree):
            frame["time_bin"] = frame["time_bin"] + pd.DateOffset(years=1)
        with self.assertRaisesRegex(ValueError, "2023"):
            COMPARE.align(logreg, tree)

    def test_rejects_files_covering_different_rows(self):
        logreg, tree = make_predictions()
        tree = tree.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "length"):
            COMPARE.align(logreg, tree)


class WeightedScoreTests(unittest.TestCase):
    def test_unit_weights_match_the_shared_evaluation(self):
        random = np.random.default_rng(3)
        y_true = random.random(5000) < 0.05
        probability = np.round(random.random(5000) * 0.5 + y_true * 0.1, 3)  # includes ties
        order, group_end = COMPARE.ranked(probability, y_true)
        average_precision, capture = COMPARE.weighted_scores(
            y_true[order].astype(float), np.ones(len(y_true)), group_end
        )
        expected = COMPARE.LOGREG.evaluate(y_true, probability)
        self.assertAlmostEqual(average_precision, average_precision_score(y_true, probability))
        self.assertAlmostEqual(capture, expected["recall_at_top_fraction"], delta=0.01)

    def test_weights_act_like_repeated_rows(self):
        y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
        probability = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        weights = np.array([2, 0, 1, 3, 1, 0, 2, 1], dtype=float)
        order, group_end = COMPARE.ranked(probability, y_true)
        average_precision, _ = COMPARE.weighted_scores(y_true[order], weights[order], group_end)
        repeated = np.repeat(np.arange(len(y_true)), weights.astype(int))
        self.assertAlmostEqual(
            average_precision, average_precision_score(y_true[repeated], probability[repeated])
        )


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = COMPARE.align(*make_predictions())
        cls.metrics, cls.intervals, cls.details = COMPARE.compare(cls.frame, repeats=50, seed=0)

    def test_every_model_is_scored_on_the_same_rows(self):
        self.assertEqual(self.details["n_rows"].nunique(), 1)
        self.assertEqual(self.details["n_positive"].nunique(), 1)

    def test_less_noisy_models_rank_better(self):
        self.assertGreater(
            self.metrics["tree_core_plus_location"]["average_precision"],
            self.metrics["cell_hour_rate_rolling"]["average_precision"],
        )

    def test_intervals_contain_the_full_sample_estimate(self):
        for _, column, *_ in COMPARE.MODELS:
            row = self.intervals[
                (self.intervals["prediction_column"] == column)
                & (self.intervals["measure"] == "average_precision")
            ].iloc[0]
            self.assertLessEqual(row["low_95"], self.metrics[column]["average_precision"])
            self.assertGreaterEqual(row["high_95"], self.metrics[column]["average_precision"])

    def test_shared_csv_follows_the_team_template(self):
        template = pd.read_csv(ROOT / "reports" / "evaluation" / "evaluation_results_template.csv")
        rows = COMPARE.shared_csv_rows(self.metrics, self.intervals, 50)
        self.assertEqual(list(rows.columns), list(template.columns))
        self.assertTrue(rows["model"].isin(["baseline", "logistic", "lightgbm"]).all())
        self.assertTrue((rows["split"] == "2023_validation").all())
        self.assertTrue(rows[["average_precision", "roc_auc", "top5_capture"]].le(1).all().all())


class CommandLineTests(unittest.TestCase):
    def test_script_writes_the_shared_csv_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "data" / "processed" / "predictions"
            predictions.mkdir(parents=True)
            logreg, tree = make_predictions()
            logreg.to_parquet(predictions / COMPARE.LOGREG_PREDICTIONS)
            tree.to_parquet(predictions / COMPARE.TREE_PREDICTIONS)

            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--bootstrap-repeats", "20"],
                check=True,
                capture_output=True,
            )

            output = root / "reports" / "model_comparison"
            self.assertEqual(len(pd.read_csv(output / "maria_model_comparison.csv")), len(COMPARE.MODELS))
            for name in ["gain_over_baseline.png", "capture_curve.png", "calibration.png"]:
                self.assertTrue((output / "figures" / name).is_file())


if __name__ == "__main__":
    unittest.main()
