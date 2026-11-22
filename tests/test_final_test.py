import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("final_test", ROOT / "scripts" / "16final_test.py")
FINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINAL)


class FinalTestGuardTests(unittest.TestCase):
    def test_valid_freeze(self):
        FINAL.validate_freeze(
            {
                "status": "locked_before_2024_test",
                "test_data_opened": False,
                "selected_model": "no_weather",
                "features": ["hour"],
                "evaluation_rule": "Rank all NYC cell-periods together and report the share of positive periods in the global highest-risk 5%.",
            }
        )

    def test_rejects_unlocked_freeze(self):
        with self.assertRaises(RuntimeError):
            FINAL.validate_freeze({"status": "draft"})


if __name__ == "__main__":
    unittest.main()
