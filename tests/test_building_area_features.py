import unittest

import pandas as pd

from modeling.building_area_features import (
    add_building_area_features,
    build_building_area_table,
)


class BuildingAreaFeaturesTest(unittest.TestCase):
    def test_build_building_area_table_aggregates_floors_and_aliases(self):
        raw = pd.DataFrame(
            [
                {
                    "building_no": 1,
                    "building": "건축관",
                    "floor": "지상1층",
                    "private_area_sqm": 100.0,
                    "common_area_sqm": 30.0,
                    "total_area_sqm": 130.0,
                    "room_count": 10,
                },
                {
                    "building_no": 1,
                    "building": "건축관",
                    "floor": "지상2층",
                    "private_area_sqm": 120.0,
                    "common_area_sqm": 40.0,
                    "total_area_sqm": 160.0,
                    "room_count": 12,
                },
                {
                    "building_no": 2,
                    "building": "예술대학 음악관",
                    "floor": "지상1층",
                    "private_area_sqm": 200.0,
                    "common_area_sqm": 50.0,
                    "total_area_sqm": 250.0,
                    "room_count": 20,
                },
            ]
        )

        features = build_building_area_table(raw, ["건축관", "음대", "인문계식당"])

        architecture = features[features["building_name_recent"] == "건축관"].iloc[0]
        music = features[features["building_name_recent"] == "음대"].iloc[0]
        missing = features[features["building_name_recent"] == "인문계식당"].iloc[0]

        self.assertEqual(architecture["area_total_sqm"], 290.0)
        self.assertEqual(architecture["room_count"], 22)
        self.assertEqual(architecture["floor_count"], 2)
        self.assertEqual(music["area_total_sqm"], 250.0)
        self.assertEqual(missing["area_missing"], 1)

    def test_add_building_area_features_adds_intensity_and_missing_fallback(self):
        panel = pd.DataFrame(
            [
                {
                    "building_name_recent": "건축관",
                    "profile_monthly_kwh_mean": 2900.0,
                },
                {
                    "building_name_recent": "인문계식당",
                    "profile_monthly_kwh_mean": 1450.0,
                },
            ]
        )
        area_features = pd.DataFrame(
            [
                {
                    "building_name_recent": "건축관",
                    "area_total_sqm": 290.0,
                    "room_count": 22.0,
                    "floor_count": 2.0,
                    "area_missing": 0,
                },
                {
                    "building_name_recent": "인문계식당",
                    "area_total_sqm": None,
                    "room_count": None,
                    "floor_count": None,
                    "area_missing": 1,
                },
            ]
        )

        result = add_building_area_features(panel, area_features)

        architecture = result[result["building_name_recent"] == "건축관"].iloc[0]
        missing = result[result["building_name_recent"] == "인문계식당"].iloc[0]

        self.assertAlmostEqual(architecture["profile_kwh_per_sqm_month"], 10.0)
        self.assertAlmostEqual(architecture["rooms_per_1000sqm"], 22.0 / 290.0 * 1000.0)
        self.assertEqual(missing["area_missing"], 1)
        self.assertTrue(pd.notna(missing["area_total_sqm"]))


if __name__ == "__main__":
    unittest.main()
