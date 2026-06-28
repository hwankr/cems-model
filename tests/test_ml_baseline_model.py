from pathlib import Path
import unittest
import json

import numpy as np
import pandas as pd

import modeling.ml_baseline_model as ml_model

from modeling.ml_baseline_model import (
    MLPredictionSpec,
    add_ml_features,
    build_walk_forward_predictions,
    calculate_comparison_metrics,
    run_ml_baseline,
)


class DummyQuantileModel:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.center = 0.0

    def fit(self, x_train, y_train):
        self.center = float(pd.Series(y_train).median())
        return self

    def predict(self, x_test):
        return np.repeat(self.center + (self.alpha - 0.5) * 10.0, len(x_test))


def dummy_model_factory(alpha: float):
    return DummyQuantileModel(alpha)


def make_forecast_weather(day_count: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2026-04-01", periods=day_count, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "issued_at": (dates - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "weather_source": "forecast",
            "temp_mean_c": np.linspace(12.0, 23.0, day_count),
            "temp_min_c": np.linspace(7.0, 18.0, day_count),
            "temp_max_c": np.linspace(17.0, 28.0, day_count),
            "humidity_mean_pct": np.linspace(45.0, 80.0, day_count),
            "rainfall_mm": [0.0, 3.0] * (day_count // 2) + ([0.0] if day_count % 2 else []),
            "wind_mean_mps": np.linspace(0.8, 2.0, day_count),
            "cdd_18": np.maximum(np.linspace(12.0, 23.0, day_count) - 18.0, 0.0),
            "hdd_18": np.maximum(18.0 - np.linspace(12.0, 23.0, day_count), 0.0),
            "is_rainy": [0, 1] * (day_count // 2) + ([0] if day_count % 2 else []),
        }
    )


def make_area_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "building_name_recent": "A",
                "area_total_sqm": 1000.0,
                "room_count": 20.0,
                "floor_count": 4.0,
                "area_missing": 0,
            },
            {
                "building_name_recent": "B",
                "area_total_sqm": 2000.0,
                "room_count": 30.0,
                "floor_count": 5.0,
                "area_missing": 0,
            },
        ]
    )


def make_academic_features(day_count: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2026-04-01", periods=day_count, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "academic_is_instruction_period": 1,
            "academic_is_vacation": 0,
            "academic_is_midterm": [1 if 5 <= index <= 7 else 0 for index in range(day_count)],
            "academic_is_final": [1 if index >= day_count - 3 else 0 for index in range(day_count)],
            "academic_is_makeup_class_day": [1 if index == 10 else 0 for index in range(day_count)],
            "academic_is_course_eval": [1 if index >= 12 else 0 for index in range(day_count)],
            "academic_is_education_practice": 0,
            "academic_days_since_semester_start": range(30, 30 + day_count),
            "academic_days_to_final": list(reversed(range(day_count))),
            "academic_semester_week": [5 + index // 7 for index in range(day_count)],
        }
    )


def make_panel(day_count: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2026-04-01", periods=day_count, freq="D")
    rows = []
    for building_index, building in enumerate(["A", "B"]):
        for index, date in enumerate(dates):
            usage = 100.0 + building_index * 50.0 + index * 2.0
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "report_month": date.strftime("%Y-%m"),
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "building_name_recent": building,
                    "usage_kwh_clean": usage,
                    "is_validation_target_clean": True,
                    "profile_monthly_kwh_mean": 3000.0 + building_index * 1500.0,
                    "profile_monthly_kwh_median": 3000.0 + building_index * 1500.0,
                    "profile_monthly_kwh_std": 20.0,
                    "profile_monthly_kwh_min": 2900.0,
                    "profile_monthly_kwh_max": 3100.0,
                    "profile_source_month_count": 2,
                    "profile_source_months": "2024-04,2025-04",
                    "overall_avg_monthly_kwh": 3200.0 + building_index * 1200.0,
                    "month_factor_vs_overall": 1.0,
                    "calendar_days_in_month": 30,
                    "baseline_uniform_daily_kwh": 100.0 + building_index * 50.0,
                }
            )
    return pd.DataFrame(rows)


class MLBaselineModelTest(unittest.TestCase):
    def test_add_weather_features_uses_forecast_weather_for_validation_dates(self):
        panel = add_ml_features(make_panel(day_count=2))
        actual_weather = make_forecast_weather(day_count=2).drop(columns=["issued_at"]).copy()
        actual_weather["weather_source"] = "actual"
        actual_weather["temp_mean_c"] = 99.0
        forecast_weather = make_forecast_weather(day_count=2)

        with_weather = ml_model.add_weather_features(
            panel,
            actual_weather=actual_weather,
            forecast_weather=forecast_weather,
        )

        first_row = with_weather[with_weather["date"] == pd.Timestamp("2026-04-01")].iloc[0]
        self.assertEqual(first_row["weather_feature_source"], "forecast")
        self.assertEqual(first_row["weather_temp_mean_c"], 12.0)
        self.assertNotEqual(first_row["weather_temp_mean_c"], 99.0)
        self.assertIn("weather_cdd_18", with_weather.columns)
        self.assertIn("weather_hdd_18", with_weather.columns)

    def test_add_ml_features_uses_only_previous_usage_for_lags(self):
        panel = make_panel(day_count=10)
        poisoned = panel.copy()
        poisoned.loc[poisoned["date"] >= "2026-04-08", "usage_kwh_clean"] = 9999.0

        base_features = add_ml_features(panel)
        poisoned_features = add_ml_features(poisoned)
        base_row = base_features[
            (base_features["building_name_recent"] == "A")
            & (base_features["date"] == pd.Timestamp("2026-04-08"))
        ].iloc[0]
        poisoned_row = poisoned_features[
            (poisoned_features["building_name_recent"] == "A")
            & (poisoned_features["date"] == pd.Timestamp("2026-04-08"))
        ].iloc[0]

        for column in [
            "ml_lag_1d_kwh",
            "ml_lag_3d_mean_kwh",
            "ml_lag_7d_mean_kwh",
            "ml_lag_7d_same_weekday_kwh",
        ]:
            self.assertEqual(base_row[column], poisoned_row[column], column)

    def test_walk_forward_predictions_train_only_on_past_dates(self):
        panel = add_ml_features(make_panel(day_count=18))
        predictions = build_walk_forward_predictions(
            panel,
            spec=MLPredictionSpec(min_train_days=14),
            model_factory=dummy_model_factory,
        )

        self.assertEqual(predictions["date"].min(), pd.Timestamp("2026-04-15"))
        self.assertTrue((predictions["walk_forward_train_end"] < predictions["date"]).all())
        self.assertIn("pred_lightgbm_q50_kwh", predictions.columns)
        self.assertIn("pred_lightgbm_q05_kwh", predictions.columns)
        self.assertIn("pred_lightgbm_q95_kwh", predictions.columns)

    def test_walk_forward_predictions_train_weather_model_with_weather_features(self):
        recorded_columns = []

        class RecordingQuantileModel(DummyQuantileModel):
            def fit(self, x_train, y_train):
                recorded_columns.append(tuple(x_train.columns))
                return super().fit(x_train, y_train)

        panel = add_ml_features(make_panel(day_count=18))
        panel = ml_model.add_weather_features(
            panel,
            forecast_weather=make_forecast_weather(day_count=18),
        )

        predictions = build_walk_forward_predictions(
            panel,
            spec=MLPredictionSpec(min_train_days=14),
            model_factory=lambda alpha: RecordingQuantileModel(alpha),
        )

        self.assertIn("pred_lightgbm_weather_q50_kwh", predictions.columns)
        self.assertIn("pred_lightgbm_weather_q05_kwh", predictions.columns)
        self.assertIn("pred_lightgbm_weather_q95_kwh", predictions.columns)
        self.assertTrue(
            any("weather_cdd_18" in columns for columns in recorded_columns),
            recorded_columns,
        )
        self.assertTrue(
            any("weather_cdd_18" not in columns for columns in recorded_columns),
            recorded_columns,
        )

    def test_walk_forward_predictions_train_area_and_academic_candidate_models(self):
        panel = add_ml_features(make_panel(day_count=18))
        panel = ml_model.add_weather_features(
            panel,
            forecast_weather=make_forecast_weather(day_count=18),
        )
        panel = ml_model.add_area_features(panel, make_area_features())
        panel = ml_model.add_academic_features(panel, make_academic_features(day_count=18))

        predictions = build_walk_forward_predictions(
            panel,
            spec=MLPredictionSpec(min_train_days=14),
            model_factory=dummy_model_factory,
        )

        self.assertIn("pred_lightgbm_weather_area_q50_kwh", predictions.columns)
        self.assertIn("pred_lightgbm_weather_academic_q50_kwh", predictions.columns)
        self.assertIn("pred_lightgbm_weather_area_academic_q50_kwh", predictions.columns)

    def test_comparison_metrics_report_wape_and_existing_baselines(self):
        panel = add_ml_features(make_panel(day_count=18))
        predictions = build_walk_forward_predictions(
            panel,
            spec=MLPredictionSpec(min_train_days=14),
            model_factory=dummy_model_factory,
        )
        comparison = calculate_comparison_metrics(predictions)

        self.assertIn("lightgbm_quantile_q50_walk_forward", comparison["model"].tolist())
        self.assertIn("current_weekday_recent_heuristic", comparison["model"].tolist())
        self.assertIn("naive_last_week_same_weekday", comparison["model"].tolist())
        self.assertIn("wape", comparison.columns)
        self.assertNotIn("mape", comparison.columns)

    def test_comparison_metrics_include_weather_lightgbm_when_available(self):
        panel = add_ml_features(make_panel(day_count=18))
        panel = ml_model.add_weather_features(
            panel,
            forecast_weather=make_forecast_weather(day_count=18),
        )
        predictions = build_walk_forward_predictions(
            panel,
            spec=MLPredictionSpec(min_train_days=14),
            model_factory=dummy_model_factory,
        )
        comparison = calculate_comparison_metrics(predictions)

        self.assertIn(
            "lightgbm_weather_quantile_q50_walk_forward",
            comparison["model"].tolist(),
        )

    def test_comparison_metrics_include_area_academic_variants_when_available(self):
        panel = add_ml_features(make_panel(day_count=18))
        panel = ml_model.add_weather_features(
            panel,
            forecast_weather=make_forecast_weather(day_count=18),
        )
        panel = ml_model.add_area_features(panel, make_area_features())
        panel = ml_model.add_academic_features(panel, make_academic_features(day_count=18))
        predictions = build_walk_forward_predictions(
            panel,
            spec=MLPredictionSpec(min_train_days=14),
            model_factory=dummy_model_factory,
        )
        comparison = calculate_comparison_metrics(predictions)

        self.assertIn(
            "lightgbm_weather_area_quantile_q50_walk_forward",
            comparison["model"].tolist(),
        )
        self.assertIn(
            "lightgbm_weather_academic_quantile_q50_walk_forward",
            comparison["model"].tolist(),
        )
        self.assertIn(
            "lightgbm_weather_area_academic_quantile_q50_walk_forward",
            comparison["model"].tolist(),
        )

    def test_run_ml_baseline_writes_comparison_outputs(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "panel.csv"
            weather_path = tmp_path / "weather.csv"
            area_path = tmp_path / "area.csv"
            academic_path = tmp_path / "academic.csv"
            output_dir = tmp_path / "outputs"
            make_panel(day_count=18).to_csv(data_path, index=False)
            make_forecast_weather(day_count=18).to_csv(weather_path, index=False)
            make_area_features().to_csv(area_path, index=False)
            make_academic_features(day_count=18).to_csv(academic_path, index=False)

            result = run_ml_baseline(
                data_path=data_path,
                output_dir=output_dir,
                forecast_weather_path=weather_path,
                area_features_path=area_path,
                academic_features_path=academic_path,
                spec=MLPredictionSpec(min_train_days=14),
                model_factory=dummy_model_factory,
            )

            self.assertEqual(result.evaluation_rows, 8)
            predictions = pd.read_csv(output_dir / "ml_baseline_predictions.csv")
            self.assertIn("pred_lightgbm_weather_q50_kwh", predictions.columns)
            self.assertIn("pred_lightgbm_weather_area_q50_kwh", predictions.columns)
            self.assertTrue((output_dir / "ml_baseline_model_comparison.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_champion.json").exists())
            champion = json.loads((output_dir / "ml_baseline_champion.json").read_text())
            self.assertIn("prediction_column", champion)
            self.assertTrue((output_dir / "ml_baseline_coverage.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_pinball_loss.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_residual_correlations.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_group_bias_by_building.csv").exists())
            group_bias = pd.read_csv(output_dir / "ml_baseline_group_bias_by_building.csv")
            self.assertEqual(set(group_bias["prediction_column"]), {champion["prediction_column"]})
            self.assertFalse((group_bias["model"] == group_bias["building_name_recent"]).any())


if __name__ == "__main__":
    unittest.main()
