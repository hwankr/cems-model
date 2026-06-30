from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.school_hourly_model import (
    DEFAULT_VALIDATION_END,
    DEFAULT_VALIDATION_START,
    LIGHTGBM_DAY_AHEAD_COLUMN,
    LIGHTGBM_DAY_AHEAD_WEATHER_ACADEMIC_COLUMN,
    LIGHTGBM_DAY_AHEAD_WEATHER_COLUMN,
    LIGHTGBM_OPERATIONAL_COLUMN,
    DAY_AHEAD_FEATURE_COLUMNS,
    OPERATIONAL_FEATURE_COLUMNS,
    add_academic_features,
    SchoolHourlyRunResult,
    add_hourly_features,
    add_weather_features,
    calculate_metric_row,
    load_hourly_usage,
    run_school_hourly_model,
    split_train_validation,
)


class ConstantRegressor:
    def fit(self, x_train, y_train):
        self.value_ = float(pd.to_numeric(y_train, errors="coerce").median())
        return self

    def predict(self, x_test):
        return np.repeat(self.value_, len(x_test))


def make_hourly_frame(start="2026-05-20 00:00:00", periods=24 * 44):
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    usage = 1000 + timestamps.hour * 10 + timestamps.dayofweek * 25
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "period_end": (timestamps + pd.Timedelta(hours=1)).astype(str),
            "measurement_date": timestamps.date.astype(str),
            "period_label": timestamps.hour + 1,
            "interval_minutes": 60,
            "year": timestamps.year,
            "month": timestamps.month,
            "day": timestamps.day,
            "day_of_week": timestamps.dayofweek,
            "is_weekend": timestamps.dayofweek >= 5,
            "hour": timestamps.hour,
            "minute": timestamps.minute,
            "usage_kwh": usage.astype(float),
            "max_demand_kw": usage.astype(float) * 4,
            "reactive_lag_kvarh": 0.0,
            "reactive_lead_kvarh": 0.0,
            "co2_tco2": usage.astype(float) * 0.00045,
            "power_factor_lag_pct": 99.0,
            "power_factor_lead_pct": 100.0,
            "source_type": "test",
            "source_table": "hour",
            "source_file": "test.csv",
            "is_observed": 1,
            "missing_value_count": 0,
        },
    )


def make_weather_frame(start="2026-05-30", periods=10, source="actual", temp_start=20.0):
    dates = pd.date_range(start=start, periods=periods, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.astype(str),
            "weather_source": source,
            "temp_mean_c": temp_start + np.arange(periods),
            "temp_min_c": temp_start - 4.0 + np.arange(periods),
            "temp_max_c": temp_start + 5.0 + np.arange(periods),
            "humidity_mean_pct": 60.0,
            "rainfall_mm": [0.0, 3.0] * (periods // 2) + ([0.0] if periods % 2 else []),
            "wind_mean_mps": 1.5,
            "cdd_18": np.maximum(temp_start + np.arange(periods) - 18.0, 0.0),
            "hdd_18": np.maximum(18.0 - (temp_start + np.arange(periods)), 0.0),
            "is_rainy": [0, 1] * (periods // 2) + ([0] if periods % 2 else []),
        },
    )
    if source == "forecast":
        frame["issued_at"] = (dates - pd.Timedelta(hours=1)).astype(str)
    return frame


def make_academic_frame(start="2026-05-30", periods=10):
    dates = pd.date_range(start=start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "academic_is_instruction_period": 1,
            "academic_is_vacation": 0,
            "academic_is_midterm": [1 if index == 2 else 0 for index in range(periods)],
            "academic_is_final": [1 if index >= periods - 2 else 0 for index in range(periods)],
            "academic_is_makeup_class_day": 0,
            "academic_is_course_eval": 0,
            "academic_is_education_practice": 0,
            "academic_days_since_semester_start": range(80, 80 + periods),
            "academic_days_to_final": list(reversed(range(periods))),
            "academic_semester_week": [12 + index // 7 for index in range(periods)],
        },
    )


class SchoolHourlyModelTest(unittest.TestCase):
    def test_load_hourly_usage_normalizes_timestamp_and_sorts(self):
        raw = make_hourly_frame(periods=3).iloc[[2, 0, 1]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hourly.csv"
            raw.to_csv(path, index=False)

            loaded = load_hourly_usage(path)

        expected = raw.copy()
        expected["timestamp"] = pd.to_datetime(expected["timestamp"])
        expected = expected.sort_values("timestamp")

        self.assertEqual(list(loaded["timestamp"]), sorted(loaded["timestamp"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["timestamp"]))
        self.assertEqual(loaded["usage_kwh"].tolist(), expected["usage_kwh"].tolist())

    def test_add_hourly_features_uses_only_past_observations(self):
        frame = make_hourly_frame(periods=24 * 8)

        featured = add_hourly_features(frame)

        self.assertEqual(featured.loc[1, "school_lag_1h_kwh"], featured.loc[0, "usage_kwh"])
        self.assertEqual(featured.loc[24, "school_lag_24h_kwh"], featured.loc[0, "usage_kwh"])
        self.assertEqual(featured.loc[168, "school_lag_168h_kwh"], featured.loc[0, "usage_kwh"])
        self.assertAlmostEqual(
            featured.loc[24, "school_rolling_24h_mean_kwh"],
            featured.loc[:23, "usage_kwh"].mean(),
        )
        self.assertNotEqual(
            featured.loc[24, "school_rolling_24h_mean_kwh"],
            featured.loc[1:24, "usage_kwh"].mean(),
        )

    def test_day_ahead_features_exclude_intra_day_observations(self):
        self.assertNotIn("school_lag_1h_kwh", DAY_AHEAD_FEATURE_COLUMNS)
        self.assertNotIn("school_rolling_24h_mean_kwh", DAY_AHEAD_FEATURE_COLUMNS)
        self.assertNotIn("school_rolling_168h_mean_kwh", DAY_AHEAD_FEATURE_COLUMNS)
        self.assertIn("school_lag_24h_kwh", DAY_AHEAD_FEATURE_COLUMNS)
        self.assertIn("school_lag_168h_kwh", DAY_AHEAD_FEATURE_COLUMNS)
        self.assertIn("school_lag_1h_kwh", OPERATIONAL_FEATURE_COLUMNS)

    def test_add_weather_features_uses_actual_before_validation_and_forecast_during_validation(self):
        frame = add_hourly_features(make_hourly_frame(start="2026-05-31 00:00:00", periods=72))
        actual = make_weather_frame(start="2026-05-31", periods=3, source="actual", temp_start=99.0)
        forecast = make_weather_frame(start="2026-05-31", periods=3, source="forecast", temp_start=10.0)

        with_weather = add_weather_features(
            frame,
            actual_weather=actual,
            forecast_weather=forecast,
            validation_start="2026-06-01 00:00:00",
        )

        train_row = with_weather[with_weather["timestamp"] == pd.Timestamp("2026-05-31 12:00:00")].iloc[0]
        validation_row = with_weather[
            with_weather["timestamp"] == pd.Timestamp("2026-06-01 12:00:00")
        ].iloc[0]
        self.assertEqual(train_row["weather_feature_source"], "actual")
        self.assertEqual(train_row["weather_temp_mean_c"], 99.0)
        self.assertEqual(validation_row["weather_feature_source"], "forecast")
        self.assertEqual(validation_row["weather_temp_mean_c"], 11.0)
        self.assertIn("weather_cdd_18", with_weather.columns)

    def test_add_academic_features_merges_daily_calendar_values(self):
        frame = add_hourly_features(make_hourly_frame(start="2026-06-01 00:00:00", periods=48))
        academic = make_academic_frame(start="2026-06-01", periods=2)

        with_academic = add_academic_features(frame, academic)

        first_day = with_academic[with_academic["timestamp"] == pd.Timestamp("2026-06-01 09:00:00")].iloc[0]
        second_day = with_academic[with_academic["timestamp"] == pd.Timestamp("2026-06-02 09:00:00")].iloc[0]
        self.assertEqual(first_day["academic_days_since_semester_start"], 80)
        self.assertEqual(second_day["academic_days_since_semester_start"], 81)
        self.assertIn("academic_is_instruction_period", with_academic.columns)

    def test_default_split_excludes_incomplete_june_30(self):
        frame = add_hourly_features(make_hourly_frame())

        train, validation = split_train_validation(frame)

        self.assertLess(train["timestamp"].max(), pd.Timestamp(DEFAULT_VALIDATION_START))
        self.assertGreaterEqual(validation["timestamp"].min(), pd.Timestamp(DEFAULT_VALIDATION_START))
        self.assertLessEqual(validation["timestamp"].max(), pd.Timestamp(DEFAULT_VALIDATION_END))
        self.assertFalse((validation["timestamp"].dt.date.astype(str) == "2026-06-30").any())

    def test_metric_row_calculates_wape_and_bias(self):
        frame = pd.DataFrame({"usage_kwh": [100.0, 200.0], "pred": [90.0, 220.0]})

        row = calculate_metric_row(frame, "example", "pred")

        self.assertEqual(row["model"], "example")
        self.assertEqual(row["n_rows"], 2)
        self.assertAlmostEqual(row["mae_kwh"], 15.0)
        self.assertAlmostEqual(row["wape"], 30.0 / 300.0)
        self.assertAlmostEqual(row["bias_pct_sum_pred_minus_actual"], 10.0 / 300.0)

    def test_run_school_hourly_model_writes_outputs(self):
        data = make_hourly_frame(start="2026-05-01 00:00:00", periods=24 * 42)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "hourly.csv"
            output_dir = tmp_path / "outputs"
            data.to_csv(data_path, index=False)

            result = run_school_hourly_model(
                data_path=data_path,
                output_dir=output_dir,
                validation_start="2026-06-01 00:00:00",
                validation_end="2026-06-05 23:00:00",
                forecast_weather_path=None,
                actual_weather_path=None,
                academic_features_path=None,
                model_factory=lambda: ConstantRegressor(),
            )

            self.assertIsInstance(result, SchoolHourlyRunResult)
            self.assertEqual(result.validation_start, "2026-06-01 00:00:00")
            self.assertEqual(result.validation_end, "2026-06-05 23:00:00")
            self.assertEqual(result.champion_model, "lightgbm_school_hourly_day_ahead")
            self.assertGreater(result.validation_rows, 0)
            self.assertTrue((output_dir / "school_hourly_predictions.csv").exists())
            self.assertTrue((output_dir / "school_hourly_model_comparison.csv").exists())
            self.assertTrue((output_dir / "school_hourly_metrics_by_hour.csv").exists())
            self.assertTrue((output_dir / "school_hourly_metrics_by_day.csv").exists())
            self.assertTrue((output_dir / "school_hourly_top_errors.csv").exists())
            self.assertTrue((output_dir / "school_hourly_run_summary.json").exists())
            predictions = pd.read_csv(output_dir / "school_hourly_predictions.csv")
            self.assertIn(LIGHTGBM_DAY_AHEAD_COLUMN, predictions.columns)
            self.assertIn(LIGHTGBM_OPERATIONAL_COLUMN, predictions.columns)

    def test_run_school_hourly_model_writes_weather_academic_candidates_when_sources_exist(self):
        data = make_hourly_frame(start="2026-05-01 00:00:00", periods=24 * 42)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "hourly.csv"
            actual_path = tmp_path / "actual_weather.csv"
            forecast_path = tmp_path / "forecast_weather.csv"
            academic_path = tmp_path / "academic.csv"
            output_dir = tmp_path / "outputs"
            data.to_csv(data_path, index=False)
            make_weather_frame(start="2026-05-01", periods=42, source="actual", temp_start=18.0).to_csv(
                actual_path,
                index=False,
            )
            make_weather_frame(
                start="2026-06-01",
                periods=5,
                source="forecast",
                temp_start=22.0,
            ).to_csv(forecast_path, index=False)
            make_academic_frame(start="2026-05-01", periods=42).to_csv(academic_path, index=False)

            run_school_hourly_model(
                data_path=data_path,
                output_dir=output_dir,
                validation_start="2026-06-01 00:00:00",
                validation_end="2026-06-05 23:00:00",
                actual_weather_path=actual_path,
                forecast_weather_path=forecast_path,
                academic_features_path=academic_path,
                model_factory=lambda: ConstantRegressor(),
            )

            predictions = pd.read_csv(output_dir / "school_hourly_predictions.csv")
            comparison = pd.read_csv(output_dir / "school_hourly_model_comparison.csv")
            self.assertIn(LIGHTGBM_DAY_AHEAD_WEATHER_COLUMN, predictions.columns)
            self.assertIn(LIGHTGBM_DAY_AHEAD_WEATHER_ACADEMIC_COLUMN, predictions.columns)
            self.assertIn("lightgbm_school_hourly_day_ahead_weather", comparison["model"].tolist())
            self.assertIn(
                "lightgbm_school_hourly_day_ahead_weather_academic",
                comparison["model"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()
