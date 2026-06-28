import unittest

import numpy as np
import pandas as pd

from modeling.weather_school_model import (
    build_daily_building_base,
    build_school_training_frame,
    build_weather_school_predictions,
    calculate_weather_model_metrics,
)


class ConstantMultiplierModel:
    def fit(self, x_train, y_train):
        self.train_rows = len(x_train)
        return self

    def predict(self, x_test):
        return np.repeat(1.1, len(x_test))


def make_monthly_profiles() -> pd.DataFrame:
    rows = []
    for month, days in [(3, 31), (4, 30)]:
        rows.extend(
            [
                {
                    "building_name_recent": "A",
                    "month": month,
                    "profile_monthly_kwh_mean": 3100.0 if month == 3 else 3000.0,
                    "calendar_days_in_month": days,
                },
                {
                    "building_name_recent": "B",
                    "month": month,
                    "profile_monthly_kwh_mean": 6200.0 if month == 3 else 6000.0,
                    "calendar_days_in_month": days,
                },
            ],
        )
    return pd.DataFrame(rows)


def make_school_total() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-03-30",
                "report_month": "2026-03",
                "school_total_daily_kwh": 450.0,
                "school_total_day_weight_observed_month": 0.050,
                "school_total_is_usable": True,
            },
            {
                "date": "2026-03-31",
                "report_month": "2026-03",
                "school_total_daily_kwh": 420.0,
                "school_total_day_weight_observed_month": 0.045,
                "school_total_is_usable": True,
            },
            {
                "date": "2026-04-01",
                "report_month": "2026-04",
                "school_total_daily_kwh": 9999.0,
                "school_total_day_weight_observed_month": 0.040,
                "school_total_is_usable": True,
            },
        ],
    )


def make_weather() -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = pd.DataFrame(
        [
            {
                "date": "2026-03-30",
                "weather_source": "actual",
                "temp_mean_c": 18.0,
                "temp_min_c": 10.0,
                "temp_max_c": 25.0,
                "humidity_mean_pct": 55.0,
                "rainfall_mm": 0.0,
                "wind_mean_mps": 2.0,
                "cdd_18": 0.0,
                "hdd_18": 0.0,
                "is_rainy": 0,
            },
            {
                "date": "2026-03-31",
                "weather_source": "actual",
                "temp_mean_c": 20.0,
                "temp_min_c": 12.0,
                "temp_max_c": 26.0,
                "humidity_mean_pct": 65.0,
                "rainfall_mm": 2.0,
                "wind_mean_mps": 1.5,
                "cdd_18": 2.0,
                "hdd_18": 0.0,
                "is_rainy": 1,
            },
        ],
    )
    forecast = pd.DataFrame(
        [
            {
                "date": "2026-04-01",
                "issued_at": "2026-03-31 23:00",
                "weather_source": "forecast",
                "temp_mean_c": 22.0,
                "temp_min_c": 15.0,
                "temp_max_c": 28.0,
                "humidity_mean_pct": 60.0,
                "rainfall_mm": 0.0,
                "wind_mean_mps": 2.5,
                "cdd_18": 4.0,
                "hdd_18": 0.0,
                "is_rainy": 0,
            }
        ],
    )
    return actual, forecast


def make_validation_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-04-01",
                "report_month": "2026-04",
                "building_name_recent": "A",
                "usage_kwh_clean": 130.0,
                "is_validation_target_clean": True,
            },
            {
                "date": "2026-04-01",
                "report_month": "2026-04",
                "building_name_recent": "B",
                "usage_kwh_clean": 230.0,
                "is_validation_target_clean": True,
            },
        ],
    )


class WeatherSchoolModelTest(unittest.TestCase):
    def test_training_frame_stops_before_validation_start(self):
        training = build_school_training_frame(
            school_total=make_school_total(),
            monthly_profiles=make_monthly_profiles(),
            actual_weather=make_weather()[0],
            validation_start=pd.Timestamp("2026-04-01"),
        )

        self.assertEqual(training["date"].max(), pd.Timestamp("2026-03-31"))
        self.assertTrue((training["weather_source"] == "actual").all())
        self.assertIn("school_multiplier", training.columns)
        self.assertAlmostEqual(training.iloc[0]["school_multiplier"], 1.55)

    def test_building_base_uses_month_profile_for_each_target_date(self):
        base = build_daily_building_base(
            monthly_profiles=make_monthly_profiles(),
            dates=[pd.Timestamp("2026-04-01")],
        ).sort_values("building_name_recent")

        self.assertEqual(base["date"].nunique(), 1)
        self.assertEqual(base["building_name_recent"].tolist(), ["A", "B"])
        self.assertAlmostEqual(base["building_base_kwh"].sum(), 300.0)

    def test_predictions_use_forecast_weather_and_allocate_school_total(self):
        actual_weather, forecast_weather = make_weather()
        predictions = build_weather_school_predictions(
            school_total=make_school_total(),
            monthly_profiles=make_monthly_profiles(),
            actual_weather=actual_weather,
            forecast_weather=forecast_weather,
            validation_panel=make_validation_panel(),
            validation_start=pd.Timestamp("2026-04-01"),
            model_factory=ConstantMultiplierModel,
        )

        self.assertTrue((predictions["weather_source"] == "forecast").all())
        self.assertAlmostEqual(predictions["pred_school_total_kwh"].iloc[0], 330.0)
        self.assertAlmostEqual(predictions["pred_weather_school_kwh"].sum(), 330.0)
        self.assertAlmostEqual(
            predictions.loc[
                predictions["building_name_recent"] == "A",
                "pred_weather_school_kwh",
            ].iloc[0],
            110.0,
        )

        metrics = calculate_weather_model_metrics(predictions)
        self.assertEqual(metrics.iloc[0]["model"], "weather_school_shape")
        self.assertIn("wape", metrics.columns)


if __name__ == "__main__":
    unittest.main()
