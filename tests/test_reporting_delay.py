import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "13reporting_delay.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DELAY = load_module("reporting_delay", SCRIPT)
TREE_TESTS = load_module("test_tree_model", ROOT / "tests" / "test_tree_model.py")
LOGREG = DELAY.LOGREG


def history_frames(flip_from=None, flip_to=None):
    """Two consecutive splits with history features, optionally with some outcomes flipped."""
    train = TREE_TESTS.prepare(TREE_TESTS.make_panel("2022-12-01", "2023-01-01"), "train")
    validation = TREE_TESTS.prepare(
        TREE_TESTS.make_panel("2023-01-01", "2023-01-11", seed=1), "validation"
    )
    if flip_from is not None:
        for frame in (train, validation):
            flipped = (frame["time_bin"] >= flip_from) & (frame["time_bin"] < flip_to)
            frame.loc[flipped, "has_crash"] = ~frame.loc[flipped, "has_crash"]
    return LOGREG.add_history_features([train, validation], 5, 7)


class DelayedHistoryTests(unittest.TestCase):
    def test_zero_delay_keeps_the_history_features(self):
        frames = history_frames()
        for original, delayed in zip(frames, DELAY.delay_history(frames, 0)):
            pd.testing.assert_frame_equal(original, delayed)

    def test_delayed_features_are_the_values_from_that_many_days_earlier(self):
        frames = history_frames()
        combined = pd.concat(frames, ignore_index=True)
        delayed = pd.concat(DELAY.delay_history(frames, 3), ignore_index=True)
        earlier = combined.assign(time_bin=combined["time_bin"] + pd.Timedelta(days=3))
        merged = delayed.merge(earlier, on=["cell_id", "time_bin"], suffixes=("", "_earlier"))
        self.assertEqual(len(merged), len(combined) - 3 * 4 * combined["cell_id"].nunique())
        for column in DELAY.HISTORY_COLUMNS:
            np.testing.assert_array_equal(merged[column], merged[f"{column}_earlier"])

    def test_outcomes_inside_the_delay_are_not_used(self):
        last = pd.Timestamp("2023-01-10 18:00")
        delay = pd.Timedelta(days=3)
        original = DELAY.delay_history(history_frames(), 3)[1]
        # Every outcome in the three days before the last period, and the period itself.
        hidden = DELAY.delay_history(history_frames(last - delay, last + delay), 3)[1]
        # The day just before those three, which a three-day delay can still see.
        visible = DELAY.delay_history(
            history_frames(last - delay - pd.Timedelta(days=1), last - delay), 3
        )[1]

        final = original["time_bin"] == last
        for column in DELAY.HISTORY_COLUMNS:
            np.testing.assert_array_equal(original.loc[final, column], hidden.loc[final, column])
            self.assertFalse(np.allclose(original.loc[final, column], visible.loc[final, column]))

    def test_periods_without_earlier_data_get_the_no_history_value(self):
        frames = history_frames()
        train = DELAY.delay_history(frames, 3)[0]
        start = train["time_bin"].min()
        early = train["time_bin"] < start + pd.Timedelta(days=3)
        first = frames[0]["time_bin"] == start
        for column in DELAY.HISTORY_COLUMNS:
            no_history = frames[0].loc[first, column].unique()
            self.assertEqual(len(no_history), 1)
            self.assertTrue((train.loc[early, column] == no_history[0]).all())

    def test_rejects_panels_with_missing_periods(self):
        train, validation = history_frames()
        gap = validation["time_bin"] != pd.Timestamp("2023-01-05 06:00")
        with self.assertRaisesRegex(ValueError, "days earlier"):
            DELAY.delay_history([train, validation.loc[gap]], 3)


class FittingRowsTests(unittest.TestCase):
    def setUp(self):
        self.train = TREE_TESTS.prepare(TREE_TESTS.make_panel("2022-07-01", "2023-01-01"), "train")

    def rows(self, delay_days):
        return DELAY.fitting_rows(
            self.train, delay_days, negative_rate=0.5, burn_in_days=28,
            minimum_history_days=14, seed=0,
        )

    def test_short_delays_fit_on_the_same_rows_as_no_delay(self):
        start = self.train["time_bin"].min()
        for original, delayed in zip(self.rows(0), self.rows(14)):
            pd.testing.assert_frame_equal(original, delayed)
            self.assertGreaterEqual(original["time_bin"].min(), start + pd.Timedelta(days=28))

    def test_long_delays_drop_only_the_rows_left_without_history(self):
        first_fitted = self.train["time_bin"].min() + pd.Timedelta(days=60 + 14)
        for original, delayed in zip(self.rows(0), self.rows(60)):
            self.assertLess(len(delayed), len(original))
            pd.testing.assert_frame_equal(
                original.loc[original["time_bin"] >= first_fitted], delayed
            )


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = TREE_TESTS.prepare(TREE_TESTS.make_panel("2022-07-01", "2023-01-01"), "train")
        validation = TREE_TESTS.prepare(
            TREE_TESTS.make_panel("2023-01-01", "2023-03-01", seed=1), "validation"
        )
        cls.y_true = validation["has_crash"].to_numpy()
        cls.scores = DELAY.score_delays(
            train, validation, delays=[0, 7], negative_rate=0.5, burn_in_days=14,
            prior_strength=5, recent_window_days=14,
        )
        cls.details = DELAY.measure(
            cls.y_true, validation["time_bin"].dt.normalize(), cls.scores, repeats=30, seed=0
        )

    def test_scores_every_model_with_and_without_refitting(self):
        self.assertEqual(
            set(self.scores),
            {(model, delay, "refit") for model in ["baseline", "logistic", "lightgbm"] for delay in [0, 7]}
            | {(model, 7, "frozen") for model in ["logistic", "lightgbm"]},
        )

    def test_changes_are_measured_from_the_zero_day_delay(self):
        reference = self.details[self.details["delay_days"] == 0]
        self.assertTrue((reference["average_precision_change"] == 0).all())
        self.assertTrue((reference["average_precision_change_low_95"] == 0).all())
        delayed = self.details[self.details["delay_days"] == 7]
        self.assertTrue(
            (delayed["average_precision_change_low_95"] <= delayed["average_precision_change_high_95"]).all()
        )

    def test_rejects_delays_without_a_zero_day_reference(self):
        with self.assertRaisesRegex(ValueError, "include 0"):
            DELAY.score_delays(None, None, delays=[3, 7])

    def test_rejects_validation_rows_outside_2023(self):
        train = TREE_TESTS.prepare(TREE_TESTS.make_panel("2023-01-01", "2023-02-01"), "train")
        later = TREE_TESTS.prepare(TREE_TESTS.make_panel("2024-01-01", "2024-01-05"), "validation")
        with self.assertRaisesRegex(ValueError, "2023"):
            DELAY.score_delays(train, later)

    def test_shared_csv_follows_the_team_template(self):
        template = pd.read_csv(ROOT / "reports" / "evaluation" / "evaluation_results_template.csv")
        rows = DELAY.shared_csv_rows(self.details, 30)
        self.assertEqual(list(rows.columns), list(template.columns))
        self.assertEqual(len(rows), 6)
        self.assertFalse(rows.duplicated(["model", "group"]).any())
        self.assertEqual(set(rows["group"]), {"0_day_delay", "7_day_delay"})
        self.assertTrue((rows["experiment"] == "reporting_delay").all())
        self.assertTrue((rows["top5_scope"] == "global").all())
        self.assertTrue((rows["split"] == "2023_validation").all())
        self.assertTrue((rows["n_positive"] == int(self.y_true.sum())).all())


class CommandLineTests(unittest.TestCase):
    def test_script_writes_the_shared_csv_and_figure_without_a_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            TREE_TESTS.make_panel("2022-10-01", "2023-01-01").to_parquet(
                processed / LOGREG.TRAIN_FILE
            )
            TREE_TESTS.make_panel("2023-01-01", "2023-02-01", seed=1).to_parquet(
                processed / LOGREG.VALIDATION_FILE
            )

            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--root", str(root),
                    "--delays", "0,3",
                    "--negative-sample-rate", "0.5",
                    "--burn-in-days", "7",
                    "--bootstrap-repeats", "10",
                ],
                check=True,
                capture_output=True,
            )

            output = root / "reports" / "reporting_delay"
            shared = pd.read_csv(output / DELAY.SHARED_CSV)
            self.assertEqual(len(shared), 6)
            self.assertEqual(len(pd.read_csv(output / DELAY.DETAILS_CSV)), 8)
            self.assertTrue((output / "figures" / DELAY.FIGURE).is_file())


if __name__ == "__main__":
    unittest.main()
