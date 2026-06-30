from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling import school_overuse_model as som


def _toy_predictions():
    # Deliberately unsorted quantiles in one row to test crossing fix is applied upstream.
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-01 00:00:00", "2026-06-01 01:00:00", "2026-06-01 02:00:00"]
            ),
            "report_date": ["2026-06-01", "2026-06-01", "2026-06-01"],
            "usage_kwh": [100.0, 250.0, 50.0],
            "p10_kwh": [80.0, 120.0, 60.0],
            "p50_kwh": [100.0, 150.0, 75.0],
            "p90_kwh": [130.0, 200.0, 90.0],
        }
    )


def test_day_ahead_features_have_no_same_hour_leakage():
    leaks = {"school_lag_1h_kwh", "school_rolling_24h_mean_kwh", "school_rolling_168h_mean_kwh"}
    assert leaks.isdisjoint(set(som.DAY_AHEAD_FEATURES))
    assert "school_lag_24h_kwh" in som.DAY_AHEAD_FEATURES
    assert "school_same_hour_7d_mean_kwh" in som.DAY_AHEAD_FEATURES


def test_flag_overuse_math():
    flagged = som.flag_overuse(_toy_predictions())
    # row0 actual 100 inside [80,130]; row1 actual 250 > 200 over-use; row2 actual 50 < 60 under-use
    assert flagged.loc[0, "in_normal_band"] == True  # noqa: E712
    assert flagged.loc[0, "is_overuse"] == False  # noqa: E712
    assert flagged.loc[1, "is_overuse"] == True  # noqa: E712
    assert flagged.loc[1, "exceedance_kwh"] == pytest.approx(50.0)
    assert flagged.loc[1, "exceedance_pct"] == pytest.approx(50.0 / 200.0)
    assert flagged.loc[2, "is_underuse"] == True  # noqa: E712
    assert flagged.loc[2, "exceedance_kwh"] == pytest.approx(0.0)
    # band_position = (actual - p10) / (p90 - p10)
    assert flagged.loc[0, "band_position"] == pytest.approx((100 - 80) / (130 - 80))


def test_compute_band_metrics_keys_and_coverage():
    flagged = som.flag_overuse(_toy_predictions())
    metrics = som.compute_band_metrics(flagged)
    for key in [
        "coverage", "pinball_p10", "pinball_p50", "pinball_p90",
        "p50_wape", "p50_mae_kwh", "p50_rmse_kwh", "p50_bias_pct",
        "overuse_hours", "underuse_hours", "overuse_total_exceedance_kwh",
        "mean_band_width_kwh", "actual_sum_kwh", "n_rows",
    ]:
        assert key in metrics
    # one of three rows inside band -> coverage 1/3
    assert metrics["coverage"] == pytest.approx(1 / 3)
    assert metrics["overuse_hours"] == 1
    assert metrics["underuse_hours"] == 1
    assert metrics["overuse_total_exceedance_kwh"] == pytest.approx(50.0)
    assert metrics["n_rows"] == 3


def test_train_quantile_band_monotone_and_shape():
    # Build a small but realistic frame through the real feature pipeline.
    data_path = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")
    frame = som.build_modeling_frame(data_path)
    train, validation = som.split_train_validation(
        frame, "2026-06-01 00:00:00", "2026-06-20 23:00:00"
    )
    result = som.train_quantile_band(train, validation)
    preds = result.predictions
    assert len(preds) == len(validation)
    # monotonic quantiles after crossing fix
    assert (preds["p10_kwh"] <= preds["p50_kwh"] + 1e-6).all()
    assert (preds["p50_kwh"] <= preds["p90_kwh"] + 1e-6).all()
    # predictions are non-negative
    assert (preds["p10_kwh"] >= 0).all()


def test_fallback_regressor_used_when_no_lightgbm(monkeypatch):
    # Force the fallback path and confirm it still yields a monotone band.
    monkeypatch.setattr(som, "default_quantile_model_factory", som.fallback_quantile_model_factory)
    data_path = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")
    frame = som.build_modeling_frame(data_path)
    train, validation = som.split_train_validation(
        frame, "2026-06-01 00:00:00", "2026-06-20 23:00:00"
    )
    result = som.train_quantile_band(train, validation, model_factory=som.fallback_quantile_model_factory)
    preds = result.predictions
    assert (preds["p10_kwh"] <= preds["p90_kwh"] + 1e-6).all()
    assert result.used_fallback is True


def test_explain_band_schema(tmp_path):
    data_path = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")
    frame = som.build_modeling_frame(data_path)
    train, validation = som.split_train_validation(frame, "2026-06-01 00:00:00", "2026-06-20 23:00:00")
    band = som.train_quantile_band(train, validation)
    flagged = som.flag_overuse(band.predictions)
    explanations, importance, shap_available = som.explain_band(
        band.p50_model, band.x_validation, flagged, band.feature_columns, top_k=5
    )
    assert set(["timestamp", "rank", "feature", "feature_label_ko", "shap_value_kwh", "feature_value", "direction"]).issubset(explanations.columns)
    assert (explanations["rank"] >= 1).all() and (explanations["rank"] <= 5).all()
    assert set(["feature", "feature_label_ko", "mean_abs_shap"]).issubset(importance.columns)
    # every explained timestamp is an over-use hour
    overuse_ts = set(flagged.loc[flagged["is_overuse"], "timestamp"])
    assert set(explanations["timestamp"]).issubset(overuse_ts)


def test_calibration_widens_band_and_sets_calib_coverage():
    """CQR calibration must widen the raw band and report ~80% calib coverage."""
    data_path = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")
    frame = som.build_modeling_frame(data_path)
    train, validation = som.split_train_validation(
        frame, "2026-06-01 00:00:00", "2026-06-20 23:00:00"
    )
    result = som.train_quantile_band(train, validation, calibrate=True, calibration_days=28, target_coverage=0.8)

    # Calibration block is present and applied
    assert result.calibration.get("applied") is True
    # Q must widen the band (raw coverage < 0.8, so Q > 0)
    assert result.calibration["q_kwh"] > 0, "Expected Q>0 since LightGBM raw bands under-cover"
    # calib_coverage_after should be close to 0.8 (±0.06 tolerance)
    calib_cov = result.calibration["calib_coverage_after"]
    assert 0.76 <= calib_cov <= 0.86, f"calib_coverage_after={calib_cov} not in [0.76, 0.86]"
    # calibrated validation coverage should be strictly greater than raw coverage
    flagged = som.flag_overuse(result.predictions)
    metrics = som.compute_band_metrics(flagged)
    calibrated_cov = metrics["coverage"]
    raw_cov = metrics["coverage_raw"]
    assert calibrated_cov > raw_cov, (
        f"Calibrated coverage {calibrated_cov:.3f} should exceed raw {raw_cov:.3f}"
    )


def test_compute_band_metrics_coverage_raw_nan_without_raw_columns():
    """When p10_raw_kwh / p90_raw_kwh are absent, coverage_raw must be NaN."""
    flagged = som.flag_overuse(_toy_predictions())
    metrics = som.compute_band_metrics(flagged)
    # toy predictions have no p10_raw_kwh column → coverage_raw must be NaN
    assert "coverage_raw" in metrics, "coverage_raw key must always be present"
    assert np.isnan(metrics["coverage_raw"]), f"Expected NaN, got {metrics['coverage_raw']}"
    # coverage (calibrated) still works: 1/3 in band
    assert metrics["coverage"] == pytest.approx(1 / 3)


def test_run_school_overuse_model_writes_artifacts(tmp_path):
    out = tmp_path / "outputs"
    web = tmp_path / "school_overuse_web"
    result = som.run_school_overuse_model(output_dir=out, web_dir=web)
    for name in ["school_overuse_predictions.csv", "school_overuse_explanations.csv",
                 "school_overuse_feature_importance.csv", "school_overuse_daily_summary.csv",
                 "school_overuse_metrics.json", "school_overuse_run_summary.json"]:
        assert (out / name).exists()
    monitor = web / "data" / "monitor.json"
    assert monitor.exists()
    import json
    bundle = json.loads(monitor.read_text(encoding="utf-8"))
    assert set(["meta", "metrics", "baselines", "series", "overuse", "feature_importance", "glossary"]).issubset(bundle)
    assert bundle["meta"]["validation_rows"] == len(bundle["series"])
    assert 0.0 <= bundle["metrics"]["coverage"] <= 1.0
    assert result.validation_rows == bundle["meta"]["validation_rows"]

    # Verify SHAP path is genuine: shap_available must be True and overuse items
    # must carry real explanations (not empty fallback stubs).
    assert bundle["meta"]["shap_available"] is True, (
        "Expected shap_available=True in monitor.json meta"
    )
    assert len(bundle["overuse"]) > 0, "Expected at least one over-use hour in bundle"
    first_overuse = bundle["overuse"][0]
    assert "explanations" in first_overuse, "Over-use entry must have 'explanations' key"
    assert len(first_overuse["explanations"]) > 0, "Over-use explanations list must be non-empty"
    required_keys = {"feature", "label_ko", "shap_kwh", "feature_value", "direction"}
    first_expl = first_overuse["explanations"][0]
    assert required_keys.issubset(first_expl.keys()), (
        f"First explanation missing keys: {required_keys - first_expl.keys()}"
    )
