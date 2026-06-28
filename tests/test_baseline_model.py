from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from modeling.baseline_model import (
    OBSERVED_SCHOOL_FLOW_COLUMNS,
    PREDICTION_COLUMNS,
    add_baseline_predictions,
    build_validation_frame,
    calculate_metrics,
    run_baseline,
    validate_panel_schema,
)


WEEKDAY_RECENT_FEATURE_COLUMNS = [
    "weekday_recent_prev_day_kwh",
    "weekday_recent_recent_3d_mean_kwh",
    "weekday_recent_recent_7d_mean_kwh",
    "weekday_recent_prev_week_same_weekday_kwh",
    "weekday_recent_weekday_profile_kwh",
]


class BaselineModelTest(unittest.TestCase):
    def test_add_baseline_predictions_uses_only_realtime_available_inputs(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
                "report_month": ["2026-04", "2026-04"],
                "building_name_recent": ["A", "A"],
                "usage_kwh_clean": [10.0, 20.0],
                "profile_monthly_kwh_mean": [300.0, 620.0],
                "calendar_days_in_month": [30, 31],
                "school_total_day_weight_observed_month": [0.10, 0.25],
            }
        )

        result = add_baseline_predictions(panel)

        self.assertEqual(result["pred_realtime_uniform_daily_kwh"].tolist(), [10.0, 20.0])
        self.assertIn("pred_weekday_recent_kwh", result.columns)
        self.assertNotIn("pred_school_shape_kwh", result.columns)
        self.assertEqual(
            PREDICTION_COLUMNS,
            ["pred_realtime_uniform_daily_kwh", "pred_weekday_recent_kwh"],
        )

    def test_weekday_recent_prediction_for_april_10_ignores_current_and_future_usage(self):
        dates = pd.date_range("2026-04-07", "2026-04-12")
        base_panel = pd.DataFrame(
            {
                "date": dates,
                "report_month": ["2026-04"] * len(dates),
                "building_name_recent": ["A"] * len(dates),
                "usage_kwh_clean": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                "profile_monthly_kwh_mean": [300.0] * len(dates),
                "calendar_days_in_month": [30] * len(dates),
                "school_total_daily_kwh": [9000.0, 9100.0, 9200.0, 9300.0, 9400.0, 9500.0],
                "school_total_day_weight_observed_month": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
                "baseline_school_shape_kwh_observed_norm": [30.0, 33.0, 36.0, 39.0, 42.0, 45.0],
            }
        )
        poisoned_panel = base_panel.copy()
        poisoned_panel.loc[poisoned_panel["date"] >= pd.Timestamp("2026-04-10"), "usage_kwh_clean"] = [
            9000.0,
            9100.0,
            9200.0,
        ]
        poisoned_panel.loc[
            poisoned_panel["date"] >= pd.Timestamp("2026-04-10"),
            [
                "school_total_daily_kwh",
                "school_total_day_weight_observed_month",
                "baseline_school_shape_kwh_observed_norm",
            ],
        ] = [
            [99000.0, 0.91, 273.0],
            [99100.0, 0.92, 276.0],
            [99200.0, 0.93, 279.0],
        ]

        base_result = add_baseline_predictions(base_panel)
        poisoned_result = add_baseline_predictions(poisoned_panel)
        base_april_10 = base_result.loc[base_result["date"] == pd.Timestamp("2026-04-10")].iloc[0]
        poisoned_april_10 = poisoned_result.loc[
            poisoned_result["date"] == pd.Timestamp("2026-04-10")
        ].iloc[0]

        self.assertEqual(base_april_10["weekday_recent_prev_day_kwh"], 120.0)
        self.assertEqual(base_april_10["weekday_recent_recent_3d_mean_kwh"], 110.0)
        self.assertEqual(base_april_10["weekday_recent_recent_7d_mean_kwh"], 10.0)
        self.assertEqual(base_april_10["weekday_recent_prev_week_same_weekday_kwh"], 10.0)
        self.assertEqual(base_april_10["weekday_recent_weekday_profile_kwh"], 10.0)

        comparison_columns = ["pred_weekday_recent_kwh", *WEEKDAY_RECENT_FEATURE_COLUMNS]
        for column in comparison_columns:
            self.assertEqual(base_april_10[column], poisoned_april_10[column], column)

    def test_validate_panel_schema_rejects_future_profile_sources(self):
        panel = pd.DataFrame(
            {
                "date": ["2026-04-01"],
                "report_month": ["2026-04"],
                "building_name_recent": ["A"],
                "usage_kwh_clean": [10.0],
                "is_validation_target_clean": [True],
                "profile_monthly_kwh_mean": [300.0],
                "calendar_days_in_month": [30],
                "school_total_day_weight_observed_month": [0.10],
                "profile_source_months": ["2026-04"],
            }
        )

        with self.assertRaisesRegex(ValueError, "future profile source"):
            validate_panel_schema(panel)

    def test_build_validation_frame_accepts_boolean_and_string_true_values(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-06-25"]),
                "building_name_recent": ["A", "A", "A"],
                "is_validation_target_clean": [True, "true", False],
            }
        )

        result = build_validation_frame(panel)

        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            [
                "2026-04-01",
                "2026-04-02",
            ],
        )

    def test_calculate_metrics_ignores_zero_actual_rows_for_mape_only(self):
        frame = pd.DataFrame(
            {
                "building_name_recent": ["A", "A"],
                "usage_kwh_clean": [10.0, 0.0],
                "pred_realtime_uniform_daily_kwh": [15.0, 5.0],
            }
        )

        metrics = calculate_metrics(
            frame,
            "pred_realtime_uniform_daily_kwh",
            ["building_name_recent"],
        )

        row = metrics.iloc[0]
        self.assertEqual(row["building_name_recent"], "A")
        self.assertEqual(row["n_rows"], 2)
        self.assertEqual(row["actual_sum_kwh_clean"], 10.0)
        self.assertEqual(row["pred_sum_kwh"], 20.0)
        self.assertEqual(row["mae_kwh"], 5.0)
        self.assertEqual(row["rmse_kwh"], 5.0)
        self.assertEqual(row["mape"], 0.5)
        self.assertEqual(row["bias_kwh_mean"], 5.0)
        self.assertEqual(row["bias_pct_sum"], 1.0)

    def test_run_baseline_writes_prediction_metric_and_ranking_outputs(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "panel.csv"
            output_dir = tmp_path / "outputs"
            panel = pd.DataFrame(
                {
                    "date": ["2026-04-01", "2026-04-02", "2026-06-25"],
                    "report_month": ["2026-04", "2026-04", "2026-06"],
                    "building_name_recent": ["A", "A", "A"],
                    "usage_kwh_clean": [10.0, 20.0, np.nan],
                    "is_validation_target_clean": [True, True, False],
                    "profile_monthly_kwh_mean": [300.0, 300.0, 300.0],
                    "calendar_days_in_month": [30, 30, 30],
                    "school_total_day_weight_observed_month": [0.05, 0.10, 0.20],
                    "profile_source_months": ["2025-04", "2025-04", "2025-06"],
                }
            )
            panel.to_csv(data_path, index=False)

            result = run_baseline(data_path, output_dir)

            self.assertEqual(result.validation_rows, 2)
            self.assertTrue((output_dir / "baseline_predictions.csv").exists())
            self.assertTrue((output_dir / "baseline_metrics_by_building.csv").exists())
            self.assertTrue((output_dir / "baseline_metrics_by_month.csv").exists())
            self.assertTrue((output_dir / "baseline_error_rankings.csv").exists())

            predictions = pd.read_csv(output_dir / "baseline_predictions.csv")
            self.assertIn("pred_realtime_uniform_daily_kwh", predictions.columns)
            self.assertIn("pred_weekday_recent_kwh", predictions.columns)
            self.assertNotIn("pred_school_shape_kwh", predictions.columns)
            for column in OBSERVED_SCHOOL_FLOW_COLUMNS:
                self.assertNotIn(column, predictions.columns)

            metrics_by_month = pd.read_csv(output_dir / "baseline_metrics_by_month.csv")
            self.assertEqual(
                metrics_by_month["prediction_method"].unique().tolist(),
                ["pred_realtime_uniform_daily_kwh", "pred_weekday_recent_kwh"],
            )

            error_rankings = pd.read_csv(output_dir / "baseline_error_rankings.csv")
            self.assertEqual(
                error_rankings["prediction_method"].unique().tolist(),
                ["pred_realtime_uniform_daily_kwh", "pred_weekday_recent_kwh"],
            )


if __name__ == "__main__":
    unittest.main()
