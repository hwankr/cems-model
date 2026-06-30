from __future__ import annotations

import warnings
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
    calibration: dict = field(default_factory=dict)


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


def train_quantile_band(
    train, validation, quantiles=QUANTILES,
    model_factory: Callable[[float], object] = default_quantile_model_factory,
    calibrate: bool = True,
    calibration_days: int = 28,
    target_coverage: float = 0.8,
):
    feature_columns = [c for c in DAY_AHEAD_FEATURES if c in train.columns and c in validation.columns]
    clean_train = train.dropna(subset=[TARGET_COLUMN]).copy()
    if clean_train.empty or validation.empty:
        raise RuntimeError("Insufficient rows to train quantile band.")

    # ------------------------------------------------------------------
    # CQR split: only when calibrate=True and we have enough history
    # ------------------------------------------------------------------
    calibration_info: dict = {"applied": False}
    do_calibrate = False

    if calibrate:
        # Determine timestamps from the index or a timestamp column
        if "timestamp" in clean_train.columns:
            ts_series = pd.to_datetime(clean_train["timestamp"])
        else:
            ts_series = pd.to_datetime(clean_train.index)
        ts_max = ts_series.max()
        calib_cutoff = ts_max - pd.Timedelta(days=calibration_days)
        calib_mask = ts_series > calib_cutoff
        proper_mask = ~calib_mask
        if proper_mask.sum() > 0 and calib_mask.sum() > 0:
            do_calibrate = True
            proper_train = clean_train.loc[proper_mask.values]
            calib_train = clean_train.loc[calib_mask.values]

    if do_calibrate:
        # Fit models on proper subset only
        y_proper = pd.to_numeric(proper_train[TARGET_COLUMN], errors="coerce")
        x_proper = build_feature_frame(proper_train, feature_columns=feature_columns)
        x_calib = build_feature_frame(calib_train, list(x_proper.columns), reference=proper_train, feature_columns=feature_columns)
        x_validation = build_feature_frame(validation, list(x_proper.columns), reference=proper_train, feature_columns=feature_columns)
        y_train = y_proper  # used for fallback check below
        x_train = x_proper
    else:
        y_train = pd.to_numeric(clean_train[TARGET_COLUMN], errors="coerce")
        x_train = build_feature_frame(clean_train, feature_columns=feature_columns)
        x_validation = build_feature_frame(validation, list(x_train.columns), reference=clean_train, feature_columns=feature_columns)

    preds = validation.copy()
    quantile_cols = {0.1: "p10_kwh", 0.5: "p50_kwh", 0.9: "p90_kwh"}
    p50_model = None
    used_fallback = False

    # Store raw predictions on calib and validation when doing CQR
    if do_calibrate:
        calib_preds: dict = {}
        val_raw: dict = {}

    for q in quantiles:
        model = model_factory(q)
        if isinstance(model, QuantileFallbackRegressor):
            used_fallback = True
        model.fit(x_train, y_train)
        if q == 0.5:
            p50_model = model
        if do_calibrate:
            calib_preds[q] = model.predict(x_calib)  # raw, may be negative
            val_raw[q] = model.predict(x_validation)  # unclipped raw, symmetric with calib
        else:
            val_pred = np.maximum(model.predict(x_validation), 0)
            preds[quantile_cols[q]] = val_pred

    if do_calibrate:
        # Raw band on calib (unclipped, same basis as conformity scores)
        p10_raw_calib = calib_preds[0.1]
        p90_raw_calib = calib_preds[0.9]
        y_calib = pd.to_numeric(calib_train[TARGET_COLUMN], errors="coerce").to_numpy()

        # Conformity scores: positive = outside band
        s = np.maximum(p10_raw_calib - y_calib, y_calib - p90_raw_calib)

        n = len(s)
        level = min(np.ceil((n + 1) * target_coverage) / n, 1.0)
        Q = float(np.quantile(s, level, method="higher"))

        # Raw validation band columns (clip >=0)
        preds["p10_raw_kwh"] = np.maximum(val_raw[0.1], 0)
        preds["p90_raw_kwh"] = np.maximum(val_raw[0.9], 0)

        # Calibrated validation band: apply CQR offset on unclipped basis, then clip final result
        p10_cal = val_raw[0.1] - Q  # unclipped, symmetric with calibration
        p90_cal = val_raw[0.9] + Q  # unclipped, symmetric with calibration
        p50_val = val_raw[0.5]  # unclipped median

        # Row-wise sort [p10, p50, p90] to enforce monotonicity, clip >=0
        band = np.stack([p10_cal, p50_val, p90_cal], axis=1)
        band.sort(axis=1)
        band = np.maximum(band, 0)
        preds["p10_kwh"] = band[:, 0]
        preds["p50_kwh"] = band[:, 1]
        preds["p90_kwh"] = band[:, 2]

        # Calib coverage AFTER applying Q
        p10_calib_cal = p10_raw_calib - Q
        p90_calib_cal = p90_raw_calib + Q
        calib_inside = ((y_calib >= p10_calib_cal) & (y_calib <= p90_calib_cal)).mean()

        # Raw validation coverage (before Q adjustment)
        val_actual = pd.to_numeric(validation[TARGET_COLUMN], errors="coerce").to_numpy()
        raw_inside_mask = ~np.isnan(val_actual)
        if raw_inside_mask.sum() > 0:
            raw_val_cov = float(
                ((val_actual[raw_inside_mask] >= preds["p10_raw_kwh"].to_numpy()[raw_inside_mask]) &
                 (val_actual[raw_inside_mask] <= preds["p90_raw_kwh"].to_numpy()[raw_inside_mask])).mean()
            )
        else:
            raw_val_cov = float("nan")

        calibration_info = {
            "applied": True,
            "calibration_days": calibration_days,
            "target_coverage": target_coverage,
            "q_kwh": round(Q, 4),
            "n_proper": int(len(proper_train)),
            "n_calib": int(n),
            "calib_coverage_after": round(float(calib_inside), 6),
            "raw_validation_coverage": round(raw_val_cov, 6),
        }
    else:
        # No calibration: crossing fix on the raw band
        band = preds[["p10_kwh", "p50_kwh", "p90_kwh"]].to_numpy()
        band.sort(axis=1)
        preds[["p10_kwh", "p50_kwh", "p90_kwh"]] = band

    return QuantileBandResult(
        predictions=preds.reset_index(drop=True), p50_model=p50_model,
        x_validation=x_validation.reset_index(drop=True),
        feature_columns=list(x_train.columns), used_fallback=used_fallback,
        calibration=calibration_info,
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

    # coverage_raw: fraction inside [p10_raw_kwh, p90_raw_kwh] if those columns exist
    if "p10_raw_kwh" in df.columns and "p90_raw_kwh" in df.columns:
        coverage_raw = float(
            ((actual >= df["p10_raw_kwh"]) & (actual <= df["p90_raw_kwh"])).mean()
        )
    else:
        coverage_raw = float("nan")

    return {
        "coverage": float(df["in_normal_band"].mean()) if "in_normal_band" in df else float("nan"),
        "coverage_raw": coverage_raw,
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


# ---------------------------------------------------------------------------
# Task 2: SHAP explanations
# ---------------------------------------------------------------------------

GLOSSARY: list[dict] = [
    {
        "term": "분위수(Quantile)",
        "desc": "데이터를 크기순으로 줄 세웠을 때 특정 비율 위치의 값. P90 = 하위 90% 지점.",
    },
    {
        "term": "P10 / P50 / P90",
        "desc": "이 시간에 '정상이라면' 사용량이 P10 이상일 확률 90%, P50(중앙값) 근처가 가장 그럴듯, P90 이하일 확률 90%. [P10,P90] 안에 정상의 약 80%가 들어옴.",
    },
    {
        "term": "정상 밴드 / 과다사용",
        "desc": "[P10,P90]가 정상 범위. 실제 > P90이면 과다사용, 초과량 = 실제 − P90.",
    },
    {
        "term": "day-ahead(하루 전 기준)",
        "desc": "당일 직전 관측을 안 쓰고 날짜·요일·시간·예보날씨·학사일정·과거 부하만으로 기대치를 만든 것. 그래서 '지금 평소보다 많이 쓰는 중'을 잡아낼 수 있음.",
    },
    {
        "term": "SHAP",
        "desc": "트리 모델의 예측을 각 피처가 얼마나 끌어올리고/내렸는지 kWh 단위로 분해한 기여도. '기온 +120, 시험기간 +80 …' 식.",
    },
    {
        "term": "WAPE",
        "desc": "가중 절대 백분율 오차 = Σ|예측−실제| / Σ실제. 낮을수록 정확.",
    },
    {
        "term": "Pinball loss",
        "desc": "분위수 예측 품질 지표. 낮을수록 분위수가 잘 맞음.",
    },
    {
        "term": "Coverage(적중률)",
        "desc": "실제값이 [P10,P90] 안에 든 비율. 잘 보정되면 ≈80%.",
    },
    {
        "term": "CDD/HDD",
        "desc": "냉방·난방 도일(degree-day). 기준온도 대비 더운/추운 정도의 누적량 → 냉난방 전력 수요 대리지표.",
    },
]


def explain_band(
    p50_model,
    x_validation: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_columns: list[str],
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Compute SHAP explanations for over-use rows.

    Returns (explanations_long_df, global_importance_df, shap_available).
    When shap_available is False, explanations_long will have the correct columns
    but ZERO rows — callers should rely on the meta.shap_available flag or the
    fallback banner rather than per-row attribution.  Global importance uses
    feature_importances_ when available; otherwise all-zero weights are returned.
    """
    # Reset indices so that positional access into shap_values (numpy array)
    # and x_validation always aligns with row labels — robust to non-default index.
    predictions = predictions.reset_index(drop=True)
    x_validation = x_validation.reset_index(drop=True)

    overuse_mask = predictions["is_overuse"].fillna(False).astype(bool)
    overuse_idx = predictions.index[overuse_mask].tolist()  # now 0-based integers

    shap_available = False
    shap_values = None

    if not isinstance(p50_model, QuantileFallbackRegressor):
        try:
            import shap as _shap

            # Use the underlying booster for LightGBM to avoid shap version issues
            booster = getattr(p50_model, "booster_", p50_model)
            explainer = _shap.TreeExplainer(booster)
            # Compute SHAP for all validation rows (needed for global importance)
            x_val_arr = x_validation[feature_columns].values
            shap_raw = explainer.shap_values(x_val_arr)
            # shap_raw may be 2-D array (n_rows, n_features) for regression
            if isinstance(shap_raw, list):
                shap_raw = shap_raw[0]
            shap_values = shap_raw  # shape (n_rows, n_features)
            shap_available = True
        except Exception as exc:
            warnings.warn(
                f"SHAP unavailable, falling back to feature_importances_: {exc}",
                stacklevel=2,
            )
            shap_values = None

    # Build global importance
    if shap_values is not None:
        mean_abs = np.abs(shap_values).mean(axis=0)
        global_importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "feature_label_ko": [feature_label_ko(f) for f in feature_columns],
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    else:
        # Fallback: use feature_importances_ if available
        fi = getattr(p50_model, "feature_importances_", None)
        if fi is not None and len(fi) == len(feature_columns):
            mean_abs = fi.astype(float)
        else:
            mean_abs = np.zeros(len(feature_columns))
        global_importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "feature_label_ko": [feature_label_ko(f) for f in feature_columns],
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Build per-overuse-row explanations in long format
    rows = []
    for idx in overuse_idx:
        # idx is a 0-based positional integer after reset_index above
        row_pred = predictions.loc[idx]
        ts = row_pred.get("timestamp", None)
        rd = row_pred.get("report_date", None)
        hr = row_pred.get("hour", None)

        if shap_values is not None:
            sv = shap_values[idx]  # shape (n_features,) — positional, safe now
            # top_k by |shap|
            abs_sv = np.abs(sv)
            top_indices = np.argsort(abs_sv)[::-1][:top_k]
            for rank, fi_idx in enumerate(top_indices, start=1):
                feat = feature_columns[fi_idx]
                sv_val = float(sv[fi_idx])
                # Use .loc with the label (== idx after reset) for unambiguous access
                fv = float(x_validation.loc[idx, feature_columns[fi_idx]]) if fi_idx < len(feature_columns) else float("nan")
                rows.append(
                    {
                        "timestamp": ts,
                        "report_date": rd,
                        "hour": hr,
                        "rank": rank,
                        "feature": feat,
                        "feature_label_ko": feature_label_ko(feat),
                        "shap_value_kwh": sv_val,
                        "feature_value": fv,
                        "direction": "up" if sv_val > 0 else "down",
                    }
                )
        # else: shap_values is None — no per-row attribution without SHAP; skip row.

    if rows:
        explanations_long = pd.DataFrame(rows)
    else:
        explanations_long = pd.DataFrame(
            columns=[
                "timestamp", "report_date", "hour", "rank", "feature",
                "feature_label_ko", "shap_value_kwh", "feature_value", "direction",
            ]
        )

    return explanations_long, global_importance_df, shap_available


# ---------------------------------------------------------------------------
# Task 2: run pipeline + output writers
# ---------------------------------------------------------------------------

@dataclass
class SchoolOveruseRunResult:
    output_dir: Path
    validation_rows: int
    validation_days: int
    coverage: float
    overuse_hours: int
    overuse_total_exceedance_kwh: float
    p50_wape: float
    used_fallback: bool
    shap_available: bool


def _score_naive(predictions: pd.DataFrame, col: str, actual_col: str = "actual_kwh") -> dict:
    """WAPE/MAE/RMSE for a naive prediction column."""
    df = predictions.dropna(subset=[actual_col, col]).copy()
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    pred = pd.to_numeric(df[col], errors="coerce")
    abs_err = (pred - actual).abs()
    actual_sum = float(actual.sum())
    wape = float(abs_err.sum() / actual_sum) if actual_sum else float("nan")
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((pred - actual) ** 2).mean()))
    return {"wape": round(wape, 6), "mae_kwh": round(mae, 4), "rmse_kwh": round(rmse, 4)}


def _build_monitor_bundle(
    predictions_flagged: pd.DataFrame,
    metrics: dict,
    explanations_long: pd.DataFrame,
    global_importance: pd.DataFrame,
    validation_start: str,
    validation_end: str,
    shap_available: bool,
    used_fallback: bool,
    calibration: dict | None = None,
) -> dict:
    """Build the monitor.json schema dict."""

    validation_rows = int(len(predictions_flagged))
    validation_days = int(predictions_flagged["report_date"].nunique()) if "report_date" in predictions_flagged.columns else 0

    # series: sorted by timestamp
    series_rows = []
    for _, r in predictions_flagged.sort_values("timestamp").iterrows():
        series_rows.append(
            {
                "timestamp": str(r["timestamp"]) if not pd.isnull(r["timestamp"]) else "",
                "report_date": str(r.get("report_date", "")),
                "hour": int(r.get("hour", 0)) if not pd.isnull(r.get("hour", 0)) else 0,
                "actual": round(float(r["actual_kwh"]), 2) if not pd.isnull(r.get("actual_kwh")) else None,
                "p10": round(float(r["p10_kwh"]), 2),
                "p50": round(float(r["p50_kwh"]), 2),
                "p90": round(float(r["p90_kwh"]), 2),
                "in_band": bool(r.get("in_normal_band", False)),
                "is_overuse": bool(r.get("is_overuse", False)),
                "exceedance_kwh": round(float(r.get("exceedance_kwh", 0.0)), 2),
                "exceedance_pct": round(float(r.get("exceedance_pct", 0.0)), 2) if not pd.isnull(r.get("exceedance_pct", 0.0)) else 0.0,
                "band_position": round(float(r.get("band_position", 0.0)), 2) if not pd.isnull(r.get("band_position", 0.0)) else 0.0,
            }
        )

    # overuse: sorted by exceedance_kwh desc, with explanations embedded
    overuse_rows = []
    overuse_df = predictions_flagged[predictions_flagged["is_overuse"].fillna(False)].sort_values(
        "exceedance_kwh", ascending=False
    )
    for _, r in overuse_df.iterrows():
        ts = r["timestamp"]
        row_expl = []
        if not explanations_long.empty and "timestamp" in explanations_long.columns:
            sub = explanations_long[explanations_long["timestamp"] == ts].sort_values("rank")
            for _, e in sub.iterrows():
                row_expl.append(
                    {
                        "feature": str(e["feature"]),
                        "label_ko": str(e["feature_label_ko"]),
                        "shap_kwh": round(float(e["shap_value_kwh"]), 4),
                        "feature_value": round(float(e["feature_value"]), 4) if not pd.isnull(e["feature_value"]) else None,
                        "direction": str(e["direction"]),
                    }
                )
        overuse_rows.append(
            {
                "timestamp": str(ts),
                "report_date": str(r.get("report_date", "")),
                "hour": int(r.get("hour", 0)) if not pd.isnull(r.get("hour", 0)) else 0,
                "actual": round(float(r["actual_kwh"]), 2) if not pd.isnull(r.get("actual_kwh")) else None,
                "p50": round(float(r["p50_kwh"]), 2),
                "p90": round(float(r["p90_kwh"]), 2),
                "exceedance_kwh": round(float(r.get("exceedance_kwh", 0.0)), 2),
                "exceedance_pct": round(float(r.get("exceedance_pct", 0.0)), 2) if not pd.isnull(r.get("exceedance_pct", 0.0)) else 0.0,
                "explanations": row_expl,
            }
        )

    # baselines
    actual_col = "actual_kwh"
    naive_cols = {
        "naive_last_day_same_hour": "pred_last_day_same_hour_kwh",
        "naive_last_week_same_hour": "pred_last_week_same_hour_kwh",
        "naive_same_hour_7d_mean": "pred_same_hour_7d_mean_kwh",
    }
    baselines = [
        {
            "model": "P50 (quantile median)",
            "wape": round(metrics["p50_wape"], 6),
            "mae_kwh": round(metrics["p50_mae_kwh"], 4),
            "rmse_kwh": round(metrics["p50_rmse_kwh"], 4),
        }
    ]
    for label, col in naive_cols.items():
        if col in predictions_flagged.columns:
            sc = _score_naive(predictions_flagged, col, actual_col)
            baselines.append({"model": label, **sc})

    # feature importance
    fi_list = []
    for _, row in global_importance.iterrows():
        fi_list.append(
            {
                "feature": str(row["feature"]),
                "label_ko": str(row["feature_label_ko"]),
                "mean_abs_shap": round(float(row["mean_abs_shap"]), 4),
            }
        )

    # metrics – round all float values to 4 decimals (NaN stays NaN)
    metrics_rounded = {}
    for k, v in metrics.items():
        if isinstance(v, float) and not np.isnan(v):
            metrics_rounded[k] = round(v, 4)
        else:
            metrics_rounded[k] = v
    metrics_rounded["coverage_target"] = 0.8

    meta_block = {
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_rows": validation_rows,
        "validation_days": validation_days,
        "quantiles": list(QUANTILES),
        "base_model": "lightgbm_quantile_day_ahead_weather_academic",
        "feature_contract": "day-ahead",
        "shap_available": shap_available,
        "used_fallback": used_fallback,
        "generated_note": "Generated by modeling/school_overuse_model.py",
    }
    if calibration:
        meta_block["calibration"] = calibration

    return {
        "meta": meta_block,
        "metrics": metrics_rounded,
        "baselines": baselines,
        "series": series_rows,
        "overuse": overuse_rows,
        "feature_importance": fi_list,
        "glossary": GLOSSARY,
    }


def run_school_overuse_model(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    validation_start: str = DEFAULT_VALIDATION_START,
    validation_end: str = DEFAULT_VALIDATION_END,
    web_dir: Path = Path("school_overuse_web"),
    actual_weather_path: Path | None = DEFAULT_ACTUAL_WEATHER_PATH,
    forecast_weather_path: Path | None = DEFAULT_FORECAST_WEATHER_PATH,
    academic_features_path: Path | None = DEFAULT_ACADEMIC_FEATURES_PATH,
) -> SchoolOveruseRunResult:
    """Full training → prediction → explanation → output pipeline."""
    import json as _json

    output_dir = Path(output_dir)
    web_dir = Path(web_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build features
    frame = build_modeling_frame(
        data_path=data_path,
        actual_weather_path=actual_weather_path,
        forecast_weather_path=forecast_weather_path,
        academic_features_path=academic_features_path,
        validation_start=validation_start,
    )

    # 2. Train/validation split
    train, validation = split_train_validation(frame, validation_start, validation_end)

    # 3. Train quantile band
    band_result = train_quantile_band(train, validation)

    # 4. Flag over-use
    flagged = flag_overuse(band_result.predictions)

    # 5. Compute metrics
    metrics = compute_band_metrics(flagged)

    # 6. SHAP explanations
    explanations_long, global_importance, shap_available = explain_band(
        band_result.p50_model,
        band_result.x_validation,
        flagged,
        band_result.feature_columns,
    )

    # 7. Build daily summary
    daily_cols = ["report_date", "is_overuse", "exceedance_kwh", "actual_kwh", "in_normal_band"]
    daily_avail = [c for c in daily_cols if c in flagged.columns]
    daily_summary = (
        flagged[daily_avail]
        .groupby("report_date")
        .agg(
            overuse_hours=("is_overuse", "sum"),
            total_exceedance_kwh=("exceedance_kwh", "sum"),
            max_exceedance_kwh=("exceedance_kwh", "max"),
            total_actual_kwh=("actual_kwh", "sum"),
            coverage=("in_normal_band", "mean"),
        )
        .reset_index()
    )

    # 8. Save CSVs
    flagged.to_csv(output_dir / "school_overuse_predictions.csv", index=False, encoding="utf-8-sig")
    explanations_long.to_csv(output_dir / "school_overuse_explanations.csv", index=False, encoding="utf-8-sig")
    global_importance.to_csv(output_dir / "school_overuse_feature_importance.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(output_dir / "school_overuse_daily_summary.csv", index=False, encoding="utf-8-sig")

    # 9. Save metrics JSON (include coverage_raw and coverage_target)
    metrics_with_target = dict(metrics)
    metrics_with_target["coverage_target"] = 0.8
    with open(output_dir / "school_overuse_metrics.json", "w", encoding="utf-8") as f:
        _json.dump(metrics_with_target, f, ensure_ascii=False, indent=2)

    # 10. Save run summary JSON
    run_summary = {
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_rows": metrics["n_rows"],
        "validation_days": int(flagged["report_date"].nunique()) if "report_date" in flagged.columns else 0,
        "coverage": round(metrics["coverage"], 6),
        "coverage_raw": round(metrics["coverage_raw"], 6) if not (isinstance(metrics["coverage_raw"], float) and np.isnan(metrics["coverage_raw"])) else None,
        "coverage_target": 0.8,
        "overuse_hours": metrics["overuse_hours"],
        "underuse_hours": metrics["underuse_hours"],
        "overuse_total_exceedance_kwh": round(metrics["overuse_total_exceedance_kwh"], 4),
        "p50_wape": round(metrics["p50_wape"], 6),
        "used_fallback": band_result.used_fallback,
        "shap_available": shap_available,
        "calibration": band_result.calibration if band_result.calibration else {"applied": False},
    }
    with open(output_dir / "school_overuse_run_summary.json", "w", encoding="utf-8") as f:
        _json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # 11. Build and save monitor.json
    bundle = _build_monitor_bundle(
        predictions_flagged=flagged,
        metrics=metrics,
        explanations_long=explanations_long,
        global_importance=global_importance,
        validation_start=validation_start,
        validation_end=validation_end,
        shap_available=shap_available,
        used_fallback=band_result.used_fallback,
        calibration=band_result.calibration if band_result.calibration else None,
    )
    web_data_dir = web_dir / "data"
    web_data_dir.mkdir(parents=True, exist_ok=True)
    with open(web_data_dir / "monitor.json", "w", encoding="utf-8") as f:
        _json.dump(bundle, f, ensure_ascii=False, indent=2)

    return SchoolOveruseRunResult(
        output_dir=output_dir,
        validation_rows=metrics["n_rows"],
        validation_days=run_summary["validation_days"],
        coverage=metrics["coverage"],
        overuse_hours=metrics["overuse_hours"],
        overuse_total_exceedance_kwh=metrics["overuse_total_exceedance_kwh"],
        p50_wape=metrics["p50_wape"],
        used_fallback=band_result.used_fallback,
        shap_available=shap_available,
    )
