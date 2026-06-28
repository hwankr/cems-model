import unittest

import pandas as pd

from modeling.weather_features import (
    add_degree_day_features,
    aggregate_forecast_weather,
    normalize_actual_weather,
    select_forecast_snapshots,
)


class WeatherFeaturesTest(unittest.TestCase):
    def test_normalize_actual_weather_adds_standard_daily_features(self):
        raw = pd.DataFrame(
            [
                {
                    "tm": "2026-03-31",
                    "avgTa": "20.5",
                    "maxTa": "26.0",
                    "minTa": "13.0",
                    "sumRn": "2.5",
                    "avgRhm": "64.0",
                    "avgWs": "1.8",
                }
            ],
        )

        normalized = normalize_actual_weather(raw)

        row = normalized.iloc[0]
        self.assertEqual(row["date"], pd.Timestamp("2026-03-31"))
        self.assertEqual(row["weather_source"], "actual")
        self.assertAlmostEqual(row["temp_mean_c"], 20.5)
        self.assertAlmostEqual(row["temp_max_c"], 26.0)
        self.assertAlmostEqual(row["temp_min_c"], 13.0)
        self.assertAlmostEqual(row["rainfall_mm"], 2.5)
        self.assertAlmostEqual(row["humidity_mean_pct"], 64.0)
        self.assertAlmostEqual(row["wind_mean_mps"], 1.8)
        self.assertAlmostEqual(row["cdd_18"], 2.5)
        self.assertAlmostEqual(row["hdd_18"], 0.0)
        self.assertEqual(row["is_rainy"], 1)

    def test_select_forecast_snapshots_uses_latest_issue_before_target_midnight(self):
        forecast = pd.DataFrame(
            [
                {
                    "target_date": "2026-04-02",
                    "issued_at": "2026-04-01 17:00",
                    "forecast_datetime": "2026-04-02 03:00",
                    "temp_c": 13.0,
                },
                {
                    "target_date": "2026-04-02",
                    "issued_at": "2026-04-02 02:00",
                    "forecast_datetime": "2026-04-02 03:00",
                    "temp_c": 15.0,
                },
                {
                    "target_date": "2026-04-02",
                    "issued_at": "2026-04-01 23:00",
                    "forecast_datetime": "2026-04-02 06:00",
                    "temp_c": 14.0,
                },
            ],
        )

        selected = select_forecast_snapshots(forecast, cutoff_hour=0)

        self.assertEqual(selected["issued_at"].nunique(), 1)
        self.assertEqual(selected["issued_at"].iloc[0], pd.Timestamp("2026-04-01 23:00"))
        self.assertEqual(selected["target_date"].iloc[0], pd.Timestamp("2026-04-02"))
        self.assertEqual(selected["weather_source"].iloc[0], "forecast")

    def test_aggregate_forecast_weather_builds_daily_feature_row(self):
        hourly = pd.DataFrame(
            [
                {
                    "target_date": "2026-04-02",
                    "issued_at": "2026-04-01 23:00",
                    "forecast_datetime": "2026-04-02 00:00",
                    "temp_c": 12.0,
                    "humidity_pct": 70.0,
                    "rainfall_mm": 0.0,
                    "wind_mps": 1.0,
                },
                {
                    "target_date": "2026-04-02",
                    "issued_at": "2026-04-01 23:00",
                    "forecast_datetime": "2026-04-02 12:00",
                    "temp_c": 24.0,
                    "humidity_pct": 55.0,
                    "rainfall_mm": 3.0,
                    "wind_mps": 2.0,
                },
            ],
        )

        aggregated = aggregate_forecast_weather(hourly)

        row = aggregated.iloc[0]
        self.assertEqual(row["date"], pd.Timestamp("2026-04-02"))
        self.assertEqual(row["weather_source"], "forecast")
        self.assertEqual(row["issued_at"], pd.Timestamp("2026-04-01 23:00"))
        self.assertAlmostEqual(row["temp_mean_c"], 18.0)
        self.assertAlmostEqual(row["temp_min_c"], 12.0)
        self.assertAlmostEqual(row["temp_max_c"], 24.0)
        self.assertAlmostEqual(row["humidity_mean_pct"], 62.5)
        self.assertAlmostEqual(row["rainfall_mm"], 3.0)
        self.assertAlmostEqual(row["wind_mean_mps"], 1.5)

        with_degrees = add_degree_day_features(aggregated)
        self.assertAlmostEqual(with_degrees.iloc[0]["cdd_18"], 0.0)
        self.assertAlmostEqual(with_degrees.iloc[0]["hdd_18"], 0.0)
        self.assertEqual(with_degrees.iloc[0]["is_rainy"], 1)


if __name__ == "__main__":
    unittest.main()
