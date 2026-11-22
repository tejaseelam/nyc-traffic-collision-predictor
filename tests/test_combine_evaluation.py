import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_evaluation", ROOT / "scripts" / "17combine_evaluation_results.py"
)
COMBINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMBINE)


class CombineEvaluationTests(unittest.TestCase):
    def test_rejects_non_validation_split(self):
        row = {column: "x" for column in COMBINE.COLUMNS}
        row.update(
            average_precision=0.1,
            roc_auc=0.6,
            top5_capture=0.2,
            top5_scope="global",
            n_rows=COMBINE.CANONICAL_ROWS,
            n_positive=COMBINE.CANONICAL_POSITIVES,
            split="2024_test",
            status="complete",
        )
        with self.assertRaises(ValueError):
            COMBINE.validate(pd.DataFrame([row], columns=COMBINE.COLUMNS))


if __name__ == "__main__":
    unittest.main()
