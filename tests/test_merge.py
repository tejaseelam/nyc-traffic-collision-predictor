import importlib.util
from pathlib import Path
import unittest

import pandas as pd
import shapely


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_pipeline",
    ROOT / "scripts" / "4merge.py",
)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class MergePipelineTests(unittest.TestCase):
    def test_weather_merge_preserves_rows_and_marks_missing_dates(self):
        crashes = pd.DataFrame({
            "collision_id": [1, 2],
            "crash_date": ["2024-01-01", "2024-01-02"],
        })
        weather = pd.DataFrame({
            "weather_date": ["2024-01-01"],
            "weather_record_present": [True],
            "weather_any_missing": [False],
            "weather_precip_mm": [2.5],
        })

        result = MERGE.merge_weather(crashes, weather)

        self.assertEqual(result["collision_id"].tolist(), [1, 2])
        self.assertEqual(result["weather_matched"].tolist(), [True, False])
        self.assertEqual(result["weather_any_missing"].tolist(), [False, True])

    def test_traffic_merge_uses_cross_street_and_preserves_rows(self):
        crashes = pd.DataFrame({
            "collision_id": [1, 2],
            "borough_final": ["BROOKLYN", "BROOKLYN"],
            "crash_datetime": ["2024-01-01 08:15", "2024-01-01 09:15"],
            "on_street_name": ["ATLANTIC AVENUE", "ATLANTIC AVENUE"],
            "cross_street_name": ["FLATBUSH AVENUE", "FLATBUSH AVENUE"],
            "off_street_name": ["UNRELATED LOCATION", "UNRELATED LOCATION"],
        })
        traffic = pd.DataFrame({
            "segmentid": [101],
            "traffic_hour": ["2024-01-01 08:00"],
            "traffic_borough": ["BROOKLYN"],
            "traffic_street": ["ATLANTIC AVENUE"],
            "traffic_from_street": ["FLATBUSH AVENUE"],
            "traffic_to_street": ["4 AVENUE"],
            "traffic_nearby_segment_volume": [800],
        })

        result = MERGE.merge_traffic(crashes, traffic, "cross_street_name")

        self.assertEqual(result["collision_id"].tolist(), [1, 2])
        self.assertEqual(result["traffic_matched"].tolist(), [True, False])
        self.assertEqual(
            result["traffic_match_status"].tolist(),
            ["MATCHED_NEARBY_SEGMENT", "NO_COUNT_AT_CRASH_HOUR"],
        )

    def test_speed_merge_matches_same_name_nearby_road(self):
        crashes = pd.DataFrame({
            "collision_id": [1],
            "latitude": [40.7300],
            "longitude": [-73.9900],
            "on_street_name": ["BROADWAY"],
            "valid_location": [True],
        })
        road = shapely.LineString([
            (-73.9902, 40.7298),
            (-73.9898, 40.7302),
        ])
        speed = pd.DataFrame({
            "speed_geometry_wkb": [shapely.to_wkb(road)],
            "speed_street_normalized": ["BROADWAY"],
            "speed_limit_snapshot_mph": pd.Series([25], dtype="Int64"),
            "speed_source_row_id": pd.Series(["road-1"], dtype="string"),
        })

        result = MERGE.merge_speed(crashes, speed)

        self.assertEqual(result["collision_id"].tolist(), [1])
        self.assertEqual(result["speed_limit_snapshot_mph"].tolist(), [25])
        self.assertEqual(result["speed_match_status"].tolist(), ["MATCHED_CURRENT_SNAPSHOT"])
        self.assertTrue(result.loc[0, "speed_matched"])


if __name__ == "__main__":
    unittest.main()
