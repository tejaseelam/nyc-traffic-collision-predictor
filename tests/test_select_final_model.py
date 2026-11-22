import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_final_model", ROOT / "scripts" / "15select_final_model.py"
)
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


class FinalModelSelectionTests(unittest.TestCase):
    def test_no_weather_selected_when_slightly_better_and_tied(self):
        selected, _ = SELECT.choose_model(
            0.0608, 0.0610, {"low_95": -0.0001, "high_95": 0.0005}
        )
        self.assertEqual(selected, "no_weather")

    def test_full_selected_when_no_weather_is_worse(self):
        selected, _ = SELECT.choose_model(
            0.0608, 0.0607, {"low_95": -0.0002, "high_95": 0.0001}
        )
        self.assertEqual(selected, "full")

    def test_candidate_features_remove_only_weather(self):
        sets = SELECT.candidate_features()
        self.assertTrue(set(SELECT.TREE.FEATURE_GROUPS["Weather"]).isdisjoint(sets["no_weather"]))
        self.assertGreater(len(sets["full"]), len(sets["no_weather"]))


if __name__ == "__main__":
    unittest.main()
