from pathlib import Path
import numpy as np, pandas as pd, pytest
from modeling import school_savings_baseline as ssb

DATA = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")


def test_frozen_features_have_no_usage_lags():
    banned = {"school_lag_1h_kwh","school_lag_24h_kwh","school_lag_168h_kwh",
              "school_rolling_24h_mean_kwh","school_rolling_168h_mean_kwh","school_same_hour_7d_mean_kwh"}
    assert banned.isdisjoint(set(ssb.FROZEN_FEATURES))
    assert "frozen_profile_kwh" in ssb.FROZEN_FEATURES


def test_frozen_profile_ignores_reporting_period():
    frame = ssb.build_savings_frame(DATA)
    out = ssb.add_frozen_profile(frame, reference_end=ssb.REFERENCE_END)
    # frozen_profile for a (dow,hour) equals the baseline-only mean, not the all-data mean
    ref = out[out["timestamp"] <= pd.Timestamp(ssb.REFERENCE_END)]
    g = ref.groupby(["day_of_week","hour"])["usage_kwh"].mean()
    sample = out.dropna(subset=["frozen_profile_kwh"]).iloc[0]
    expected = g.loc[(sample["day_of_week"], sample["hour"])]
    assert abs(sample["frozen_profile_kwh"] - expected) < 1e-6


def test_train_frozen_band_shapes_and_accuracy():
    frame = ssb.build_savings_frame(DATA)
    frame = ssb.add_frozen_profile(frame)
    result = ssb.train_frozen_band(frame)
    p = result.predictions
    assert (p["p10_kwh"] <= p["p50_kwh"] + 1e-6).all()
    assert (p["p50_kwh"] <= p["p90_kwh"] + 1e-6).all()
    assert (p["p10_kwh"] >= 0).all()
    # reporting period length = 480 hours
    assert len(p) == 480
    # held-out accuracy reported and sane
    assert 0 < result.accuracy["wape"] < 0.5
    assert 0.0 <= result.accuracy["coverage"] <= 1.0
    assert result.calibration["applied"] is True


def test_load_total_area_positive():
    assert ssb.load_total_area_sqm() > 100000  # campus is hundreds of thousands of m^2
