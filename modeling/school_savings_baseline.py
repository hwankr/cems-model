"""Frozen (M&V-style) baseline model for school energy savings estimation.

The baseline is FROZEN: it uses NO usage lags or rolling features, so it
cannot chase savings. FROZEN_FEATURES = calendar + frozen_profile_kwh +
weather + academic only.

Key design:
- frozen_profile_kwh: mean usage by (day_of_week, hour) computed using ONLY
  rows with timestamp <= REFERENCE_END (never reporting-period rows).
- train_frozen_band: split reference period into proper + held-out calib
  (last HELDOUT_DAYS days). Fit 3 LightGBM quantile models (alpha 0.1/0.5/0.9)
  on proper only. CQR calibration on calib set. Report held-out accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.school_hourly_model import (
    DEFAULT_ACADEMIC_FEATURES_PATH,
    DEFAULT_ACTUAL_WEATHER_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_FORECAST_WEATHER_PATH,
    TARGET_COLUMN,
    CALENDAR_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    add_academic_features,
    add_hourly_features,
    add_weather_features,
    build_feature_frame,
    load_hourly_usage,
    load_optional_table,
)
from modeling.academic_calendar_features import ACADEMIC_FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_END = "2026-05-31 23:00:00"
REPORTING_START = "2026-06-01 00:00:00"
REPORTING_END = "2026-06-20 23:00:00"
HELDOUT_DAYS = 28
TOTAL_AREA_SQM = 447761.8
QUANTILES = (0.1, 0.5, 0.9)

# Frozen feature set: NO usage lags, NO rolling stats — frozen profile only.
FROZEN_FEATURES: list[str] = (
    list(CALENDAR_FEATURE_COLUMNS)
    + ["frozen_profile_kwh"]
    + list(WEATHER_FEATURE_COLUMNS)
    + list(ACADEMIC_FEATURE_COLUMNS)
)


# ---------------------------------------------------------------------------
# Area loading
# ---------------------------------------------------------------------------

def load_total_area_sqm(path: Path = Path("inputs/yu_building_area.xlsx")) -> float:
    """Read total floor area from Excel (header=2, sum total_area_sqm col c6).

    Falls back to TOTAL_AREA_SQM constant on any error.
    """
    try:
        raw = pd.read_excel(path, sheet_name=0, header=2)
        # Normalize column names to c0..c7 as in building_area_features
        raw.columns = [f"c{i}" for i in range(len(raw.columns))]
        # c6 = total_area_sqm per the _normalize_excel_area_columns mapping
        total = pd.to_numeric(raw["c6"], errors="coerce").sum()
        if np.isfinite(total) and total > 0:
            return float(total)
    except Exception:
        pass
    return float(TOTAL_AREA_SQM)


# ---------------------------------------------------------------------------
# LightGBM-free fallback (mirrors school_overuse_model.QuantileFallbackRegressor)
# ---------------------------------------------------------------------------

class QuantileFallbackRegressor:
    """Predict training median shifted by the residual quantile offset."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.center_: float = 0.0
        self.offset_: float = 0.0

    def fit(self, x, y):
        y = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce")
        self.center_ = float(y.median()) if len(y) else 0.0
        resid = y - self.center_
        self.offset_ = float(resid.quantile(self.alpha)) if len(resid) else 0.0
        return self

    def predict(self, x):
        return np.repeat(self.center_ + self.offset_, len(x))


def _make_quantile_model(alpha: float):
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=24,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=20260630,
            verbosity=-1,
        )
    except ImportError:
        return QuantileFallbackRegressor(alpha)


# ---------------------------------------------------------------------------
# Frame building
# ---------------------------------------------------------------------------

def build_savings_frame(
    data_path: Path = DEFAULT_DATA_PATH,
    actual_weather_path: Path | None = DEFAULT_ACTUAL_WEATHER_PATH,
    forecast_weather_path: Path | None = DEFAULT_FORECAST_WEATHER_PATH,
    academic_features_path: Path | None = DEFAULT_ACADEMIC_FEATURES_PATH,
    reporting_start: str = REPORTING_START,
) -> pd.DataFrame:
    """Build the full modeling frame (mirrors school_overuse_model.build_modeling_frame).

    Loads usage → adds hourly features → weather → academic.
    The reporting_start boundary is passed to add_weather_features so that
    forecast weather is used for the reporting period.
    """
    hourly = load_hourly_usage(data_path)
    featured = add_hourly_features(hourly)

    actual_weather = load_optional_table(actual_weather_path)
    forecast_weather = load_optional_table(forecast_weather_path)
    if actual_weather is not None or forecast_weather is not None:
        featured = add_weather_features(
            featured,
            actual_weather=actual_weather,
            forecast_weather=forecast_weather,
            validation_start=reporting_start,
        )

    academic = load_optional_table(academic_features_path)
    if academic is not None:
        featured = add_academic_features(featured, academic)

    return featured


def add_frozen_profile(
    frame: pd.DataFrame,
    reference_end: str = REFERENCE_END,
) -> pd.DataFrame:
    """Add frozen_profile_kwh: mean usage_kwh by (day_of_week, hour) using
    ONLY rows with timestamp <= reference_end.

    This is the M&V anchor: it never uses reporting-period actuals.
    """
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    ref_end_ts = pd.Timestamp(reference_end)

    ref = result[result["timestamp"] <= ref_end_ts]
    profile = (
        ref.groupby(["day_of_week", "hour"])[TARGET_COLUMN]
        .mean()
        .rename("frozen_profile_kwh")
    )
    result = result.merge(
        profile.reset_index(),
        on=["day_of_week", "hour"],
        how="left",
    )
    return result


# ---------------------------------------------------------------------------
# FrozenBandResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrozenBandResult:
    """Result of train_frozen_band."""

    predictions: pd.DataFrame          # reporting rows with p10/p50/p90_kwh
    p50_model: object                   # fitted P50 model
    x_reporting: pd.DataFrame          # feature matrix for reporting period
    feature_columns: list[str]
    calibration: dict = field(default_factory=dict)
    accuracy: dict = field(default_factory=dict)   # held-out wape/mae/rmse/coverage
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_frozen_band(
    frame: pd.DataFrame,
    reference_end: str = REFERENCE_END,
    reporting_start: str = REPORTING_START,
    reporting_end: str = REPORTING_END,
    heldout_days: int = HELDOUT_DAYS,
    quantiles: tuple = QUANTILES,
) -> FrozenBandResult:
    """Train 3 LightGBM quantile models (P10/P50/P90) on the proper subset of
    the reference period, apply CQR calibration using the held-out calib set,
    and predict the calibrated band on the reporting period.

    Frozen leakage guard: only FROZEN_FEATURES are used — no usage lags.

    Returns FrozenBandResult with reporting predictions, held-out accuracy,
    and calibration metadata.
    """
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    ref_end_ts = pd.Timestamp(reference_end)
    rep_start_ts = pd.Timestamp(reporting_start)
    rep_end_ts = pd.Timestamp(reporting_end)

    # Split into reference and reporting
    reference = frame[frame["timestamp"] <= ref_end_ts].copy()
    reporting = frame[
        (frame["timestamp"] >= rep_start_ts) & (frame["timestamp"] <= rep_end_ts)
    ].copy()

    if reporting.empty:
        raise RuntimeError(
            f"No reporting rows found between {reporting_start} and {reporting_end}."
        )

    # Within reference: calib = last heldout_days, proper = earlier
    ref_max_ts = reference["timestamp"].max()
    calib_cutoff = ref_max_ts - pd.Timedelta(days=heldout_days)
    proper = reference[reference["timestamp"] <= calib_cutoff].dropna(
        subset=[TARGET_COLUMN]
    ).copy()
    calib = reference[reference["timestamp"] > calib_cutoff].dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    if proper.empty or calib.empty:
        raise RuntimeError("Insufficient reference rows for proper/calib split.")

    # Resolve which FROZEN_FEATURES are actually present in proper
    feature_columns = [c for c in FROZEN_FEATURES if c in proper.columns]

    # Build feature matrices
    x_proper = build_feature_frame(proper, feature_columns=feature_columns)
    x_calib = build_feature_frame(
        calib, list(x_proper.columns), reference=proper, feature_columns=feature_columns
    )
    x_reporting = build_feature_frame(
        reporting, list(x_proper.columns), reference=proper, feature_columns=feature_columns
    )

    y_proper = pd.to_numeric(proper[TARGET_COLUMN], errors="coerce")
    y_calib = pd.to_numeric(calib[TARGET_COLUMN], errors="coerce").to_numpy()

    # Fit 3 quantile models on proper only
    used_fallback = False
    raw_calib: dict[float, np.ndarray] = {}
    raw_rep: dict[float, np.ndarray] = {}
    p50_model = None

    for q in quantiles:
        model = _make_quantile_model(q)
        if isinstance(model, QuantileFallbackRegressor):
            used_fallback = True
        model.fit(x_proper, y_proper)
        if q == 0.5:
            p50_model = model
        raw_calib[q] = model.predict(x_calib)
        raw_rep[q] = model.predict(x_reporting)

    # CQR: conformity scores on held-out calib (unclipped)
    p10_raw_calib = raw_calib[0.1]
    p90_raw_calib = raw_calib[0.9]
    s = np.maximum(p10_raw_calib - y_calib, y_calib - p90_raw_calib)
    n = len(s)
    level = min(math.ceil((n + 1) * 0.8) / n, 1.0)
    Q = float(np.quantile(s, level, method="higher"))

    # Calibrated reporting band (unclipped basis → then clip ≥0, re-sort monotone)
    p10_cal = raw_rep[0.1] - Q
    p50_cal = raw_rep[0.5]
    p90_cal = raw_rep[0.9] + Q

    band = np.stack([p10_cal, p50_cal, p90_cal], axis=1)
    band.sort(axis=1)
    band = np.maximum(band, 0.0)

    predictions = reporting.copy().reset_index(drop=True)
    predictions["p10_kwh"] = band[:, 0]
    predictions["p50_kwh"] = band[:, 1]
    predictions["p90_kwh"] = band[:, 2]

    # Held-out accuracy: P50 WAPE/MAE/RMSE on calib + calibrated-band coverage on calib
    p50_raw_calib = raw_calib[0.5]
    abs_err = np.abs(p50_raw_calib - y_calib)
    actual_sum = float(y_calib.sum())
    wape = float(abs_err.sum() / actual_sum) if actual_sum > 0 else float("nan")
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((p50_raw_calib - y_calib) ** 2).mean()))

    # Coverage using the calibrated calib band
    p10_calib_cal = p10_raw_calib - Q
    p90_calib_cal = p90_raw_calib + Q
    coverage = float(((y_calib >= p10_calib_cal) & (y_calib <= p90_calib_cal)).mean())

    accuracy = {
        "wape": round(wape, 6),
        "mae_kwh": round(mae, 4),
        "rmse_kwh": round(rmse, 4),
        "coverage": round(coverage, 6),
    }

    calibration = {
        "applied": True,
        "q_kwh": round(Q, 4),
        "n_proper": int(len(proper)),
        "n_calib": int(n),
    }

    return FrozenBandResult(
        predictions=predictions,
        p50_model=p50_model,
        x_reporting=x_reporting.reset_index(drop=True),
        feature_columns=list(x_proper.columns),
        calibration=calibration,
        accuracy=accuracy,
        used_fallback=used_fallback,
    )
