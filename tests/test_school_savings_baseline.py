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


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------

def test_compute_savings_math():
    df = pd.DataFrame({"usage_kwh":[100.0,200.0], "p10_kwh":[90.0,150.0],
                       "p50_kwh":[120.0,180.0], "p90_kwh":[140.0,210.0]})
    out = ssb.compute_savings(df, total_area_sqm=1000.0)
    # row0: avoided = p50 - actual = 120 - 100 = 20 (saved)
    assert out.loc[0,"avoided_kwh"] == pytest.approx(20.0)
    # row0: actual 100 < p10 90? No (100 > 90) -> NOT confirmed saving
    assert out.loc[0,"is_confirmed_saving"] == False
    # row1: avoided = 180 - 200 = -20 (over-used)
    assert out.loc[1,"avoided_kwh"] == pytest.approx(-20.0)
    # row0: avoided_kwh_per_sqm = 20 / 1000 = 0.02
    assert out.loc[0,"avoided_kwh_per_sqm"] == pytest.approx(20.0/1000.0)
    # row1: actual 200 > p90 210? No -> not overuse
    assert out.loc[1,"is_overuse"] == False


def test_scorecard_and_leaderboard_and_bundle(tmp_path):
    res = ssb.run_savings_demo(output_dir=tmp_path/"outputs", web_dir=tmp_path/"web")
    import json
    bundle = json.loads((tmp_path/"web"/"data"/"savings.json").read_text(encoding="utf-8"))
    for k in ["meta","accuracy","scorecard","series","daily","leaderboard","glossary"]:
        assert k in bundle
    assert bundle["meta"]["reporting_rows"] == len(bundle["series"]) == 480
    assert len(bundle["daily"]) == 20
    assert len(bundle["leaderboard"]) >= 3
    assert all("rank" in r for r in bundle["leaderboard"])
    assert bundle["scorecard"]["area_sqm"] > 100000
    assert res.reporting_rows == 480
