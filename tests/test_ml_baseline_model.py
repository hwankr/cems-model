from pathlib import Path
import unittest

import numpy as np
import pandas as pd

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

    def test_run_ml_baseline_writes_comparison_outputs(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "panel.csv"
            output_dir = tmp_path / "outputs"
            make_panel(day_count=18).to_csv(data_path, index=False)

            result = run_ml_baseline(
                data_path=data_path,
                output_dir=output_dir,
                spec=MLPredictionSpec(min_train_days=14),
                model_factory=dummy_model_factory,
            )

            self.assertEqual(result.evaluation_rows, 8)
            self.assertTrue((output_dir / "ml_baseline_predictions.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_model_comparison.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_coverage.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_pinball_loss.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_residual_correlations.csv").exists())
            self.assertTrue((output_dir / "ml_baseline_group_bias_by_building.csv").exists())


if __name__ == "__main__":
    unittest.main()
