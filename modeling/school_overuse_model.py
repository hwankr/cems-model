from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from modeling.school_hourly_model import (
    DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS,
    DEFAULT_ACADEMIC_FEATURES_PATH,
    DEFAULT_ACTUAL_WEATHER_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_FORECAST_WEATHER_PATH,
    TARGET_COLUMN,
    add_academic_features,
    add_hourly_features,
    add_weather_features,
    build_feature_frame,
    load_hourly_usage,
    load_optional_table,
    split_train_validation,
)

DEFAULT_VALIDATION_START = "2026-06-01 00:00:00"
DEFAULT_VALIDATION_END = "2026-06-20 23:00:00"
DEFAULT_OUTPUT_DIR = Path("outputs")
QUANTILES = (0.1, 0.5, 0.9)
DAY_AHEAD_FEATURES = list(DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS)

FEATURE_LABELS_KO = {
    "month": "월", "day": "일", "day_of_week": "요일", "is_weekend": "주말 여부", "hour": "시간",
    "hour_sin": "시간 주기(sin)", "hour_cos": "시간 주기(cos)", "dow_sin": "요일 주기(sin)", "dow_cos": "요일 주기(cos)",
    "school_lag_24h_kwh": "24시간 전 사용량", "school_lag_168h_kwh": "1주 전 같은시간 사용량",
    "school_same_hour_7d_mean_kwh": "최근 7일 같은시간 평균",
    "weather_temp_mean_c": "평균기온", "weather_temp_min_c": "최저기온", "weather_temp_max_c": "최고기온",
    "weather_humidity_mean_pct": "평균습도", "weather_rainfall_mm": "강수량", "weather_wind_mean_mps": "평균풍속",
    "weather_cdd_18": "냉방도일(CDD)", "weather_hdd_18": "난방도일(HDD)", "weather_is_rainy": "강우 여부",
    "academic_is_instruction_period": "수업기간", "academic_is_vacation": "방학", "academic_is_midterm": "중간고사",
    "academic_is_final": "기말고사", "academic_is_makeup_class_day": "보강일", "academic_is_course_eval": "강의평가기간",
    "academic_is_education_practice": "교육실습", "academic_days_since_semester_start": "학기 경과일",
    "academic_days_to_final": "기말까지 일수", "academic_semester_week": "학기 주차",
}


def feature_label_ko(feature: str) -> str:
    return FEATURE_LABELS_KO.get(feature, feature)


@dataclass
class QuantileBandResult:
    predictions: pd.DataFrame
    p50_model: object
    x_validation: pd.DataFrame
    feature_columns: list[str]
    used_fallback: bool = False
    shap_available: bool = False


class QuantileFallbackRegressor:
    """LightGBM-free fallback: predict training median, then offset by the
    training residual quantile so the band is non-degenerate."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.center_ = 0.0
        self.offset_ = 0.0

    def fit(self, x, y):
        y = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce")
        self.center_ = float(y.median()) if len(y) else 0.0
        resid = y - self.center_
        self.offset_ = float(resid.quantile(self.alpha)) if len(resid) else 0.0
        return self

    def predict(self, x):
        return np.repeat(self.center_ + self.offset_, len(x))


def default_quantile_model_factory(alpha: float) -> object:
    try:
        import lightgbm as lgb
    except ImportError:
        return QuantileFallbackRegressor(alpha)
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=300, learning_rate=0.04, num_leaves=31,
        min_child_samples=24, subsample=0.9, colsample_bytree=0.9,
        random_state=20260630, verbosity=-1,
    )


def fallback_quantile_model_factory(alpha: float) -> object:
    return QuantileFallbackRegressor(alpha)


def build_modeling_frame(
    data_path: Path = DEFAULT_DATA_PATH,
    actual_weather_path: Path | None = DEFAULT_ACTUAL_WEATHER_PATH,
    forecast_weather_path: Path | None = DEFAULT_FORECAST_WEATHER_PATH,
    academic_features_path: Path | None = DEFAULT_ACADEMIC_FEATURES_PATH,
    validation_start: str = DEFAULT_VALIDATION_START,
) -> pd.DataFrame:
    hourly = load_hourly_usage(data_path)
    featured = add_hourly_features(hourly)
    actual_weather = load_optional_table(actual_weather_path)
    forecast_weather = load_optional_table(forecast_weather_path)
    if actual_weather is not None or forecast_weather is not None:
        featured = add_weather_features(
            featured, actual_weather=actual_weather,
            forecast_weather=forecast_weather, validation_start=validation_start,
        )
    academic = load_optional_table(academic_features_path)
    if academic is not None:
        featured = add_academic_features(featured, academic)
    return featured


def train_quantile_band(train, validation, quantiles=QUANTILES,
                        model_factory: Callable[[float], object] = default_quantile_model_factory):
    feature_columns = [c for c in DAY_AHEAD_FEATURES if c in train.columns and c in validation.columns]
    clean_train = train.dropna(subset=[TARGET_COLUMN]).copy()
    if clean_train.empty or validation.empty:
        raise RuntimeError("Insufficient rows to train quantile band.")
    y_train = pd.to_numeric(clean_train[TARGET_COLUMN], errors="coerce")
    x_train = build_feature_frame(clean_train, feature_columns=feature_columns)
    x_validation = build_feature_frame(validation, list(x_train.columns), reference=clean_train, feature_columns=feature_columns)

    preds = validation.copy()
    quantile_cols = {0.1: "p10_kwh", 0.5: "p50_kwh", 0.9: "p90_kwh"}
    p50_model = None
    used_fallback = False
    for q in quantiles:
        model = model_factory(q)
        if isinstance(model, QuantileFallbackRegressor):
            used_fallback = True
        model.fit(x_train, y_train)
        preds[quantile_cols[q]] = np.maximum(model.predict(x_validation), 0)
        if q == 0.5:
            p50_model = model
    # crossing fix: sort the three quantiles row-wise
    band = preds[["p10_kwh", "p50_kwh", "p90_kwh"]].to_numpy()
    band.sort(axis=1)
    preds[["p10_kwh", "p50_kwh", "p90_kwh"]] = band
    return QuantileBandResult(
        predictions=preds.reset_index(drop=True), p50_model=p50_model,
        x_validation=x_validation.reset_index(drop=True),
        feature_columns=list(x_train.columns), used_fallback=used_fallback,
    )


def flag_overuse(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    actual = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    result["actual_kwh"] = actual
    p10, p50, p90 = result["p10_kwh"], result["p50_kwh"], result["p90_kwh"]
    result["in_normal_band"] = (actual >= p10) & (actual <= p90)
    result["is_overuse"] = actual > p90
    result["is_underuse"] = actual < p10
    result["exceedance_kwh"] = (actual - p90).clip(lower=0)
    result["exceedance_pct"] = result["exceedance_kwh"] / p90.replace(0, np.nan)
    width = (p90 - p10).replace(0, np.nan)
    result["band_position"] = (actual - p10) / width
    return result


def _pinball(actual, pred, alpha):
    diff = actual - pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def compute_band_metrics(predictions: pd.DataFrame) -> dict:
    df = predictions.dropna(subset=[TARGET_COLUMN, "p50_kwh"]).copy()
    actual = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    p50 = df["p50_kwh"]
    abs_err = (p50 - actual).abs()
    actual_sum = float(actual.sum())
    return {
        "coverage": float(df["in_normal_band"].mean()) if "in_normal_band" in df else float("nan"),
        "pinball_p10": _pinball(actual, df["p10_kwh"], 0.1),
        "pinball_p50": _pinball(actual, df["p50_kwh"], 0.5),
        "pinball_p90": _pinball(actual, df["p90_kwh"], 0.9),
        "p50_wape": float(abs_err.sum() / actual_sum) if actual_sum else float("nan"),
        "p50_mae_kwh": float(abs_err.mean()),
        "p50_rmse_kwh": float(np.sqrt(((p50 - actual) ** 2).mean())),
        "p50_bias_pct": float((p50 - actual).sum() / actual_sum) if actual_sum else float("nan"),
        "overuse_hours": int(df["is_overuse"].sum()),
        "underuse_hours": int(df["is_underuse"].sum()),
        "overuse_total_exceedance_kwh": float(df["exceedance_kwh"].sum()),
        "mean_band_width_kwh": float((df["p90_kwh"] - df["p10_kwh"]).mean()),
        "actual_sum_kwh": actual_sum,
        "n_rows": int(len(df)),
    }
