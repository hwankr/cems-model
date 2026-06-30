# School Over-Consumption Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LightGBM quantile-band model that reports an hourly normal range `[P10, P90]` for school-wide electricity use, flags + quantifies over-consumption (`actual > P90`), explains each flagged hour with SHAP, and ships a new static website that visualizes the band, the over-use hours, the per-hour explanations, and a plain-Korean glossary.

**Architecture:** A new `modeling/school_overuse_model.py` reuses the feature engineering already in `modeling/school_hourly_model.py` (import, do not duplicate), trains three LightGBM quantile regressors (alpha 0.1/0.5/0.9) on the **day-ahead** feature contract, flags over-use, runs SHAP TreeExplainer on the P50 model, writes `outputs/school_overuse_*` artifacts plus a compact `school_overuse_web/data/monitor.json`, and a new `school_overuse_web/` static viewer renders that JSON. Nothing in the existing 17-building model, `web/`, `school_hourly_web/`, or `school_hourly_model.py` is modified.

**Tech Stack:** Python 3.12 (`.venv\Scripts\python.exe`), pandas 3.0, numpy, LightGBM 4.6 (`objective="quantile"`), SHAP 0.52 (`TreeExplainer`), unittest/pytest, static HTML/CSS/vanilla JS (no build step), SVG charts.

## Global Constraints

- Python is ONLY available via the project venv: run everything as `.venv\Scripts\python.exe ...` (PowerShell) — there is no system Python.
- Validation/scoring window: `2026-06-01 00:00:00` through `2026-06-20 23:00:00` (480 hours, 20 days). Training rows are strictly before `2026-06-01 00:00:00`.
- Day-ahead feature contract: features are calendar + `school_lag_24h_kwh` + `school_lag_168h_kwh` + `school_same_hour_7d_mean_kwh` + forecast weather + academic. The columns `school_lag_1h_kwh`, `school_rolling_24h_mean_kwh`, `school_rolling_168h_mean_kwh` MUST NOT appear in the feature set (same-hour leakage guard).
- Quantiles `(0.1, 0.5, 0.9)`. After prediction, enforce row-wise monotonicity `p10 ≤ p50 ≤ p90` by sorting the three values per row.
- Over-use definition: `is_overuse = actual > p90`; `exceedance_kwh = max(actual - p90, 0)`; `exceedance_pct = exceedance_kwh / p90`.
- Do NOT modify: any existing `modeling/*` file, `web/**`, `school_hourly_web/**`, or any existing `outputs/*` file. Only ADD new `school_overuse_*` artifacts and the new `school_overuse_web/` folder.
- Korean-first user-facing copy (web text, run doc). Code identifiers stay English.
- All new outputs written with `encoding="utf-8-sig"` for CSV (match existing modules) and UTF-8 for JSON.
- Each task: `git add` ONLY the files that task creates/changes (exact paths). Do not stage the pre-existing untracked `school_hourly_*` files or `tests/test_web_content.py`.
- Tests live in `tests/test_school_overuse_model.py` and `tests/test_school_overuse_web_content.py`. Run them with `.venv\Scripts\python.exe -m pytest <path> -q`.

### Reused interfaces from `modeling/school_hourly_model.py` (import these)

```python
from modeling.school_hourly_model import (
    load_hourly_usage,                 # (path) -> DataFrame with timestamp, usage_kwh, calendar cols, report_date/report_month
    add_hourly_features,               # (frame) -> adds lags, rolling, same_hour_7d_mean, sin/cos, naive preds
    add_weather_features,              # (frame, actual_weather, forecast_weather, validation_start) -> merged weather cols
    add_academic_features,             # (frame, academic_features) -> merged academic cols
    split_train_validation,            # (frame, validation_start, validation_end) -> (train, validation)
    build_feature_frame,               # (frame, columns=None, reference=None, feature_columns=None) -> numeric feature matrix (median-imputed)
    load_optional_table,               # (path|None) -> DataFrame|None
    DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS,  # the day-ahead feature list to reuse
    TARGET_COLUMN,                     # "usage_kwh"
)
```

The day-ahead feature list already EXCLUDES the operational leakage columns. Reuse it directly as `DAY_AHEAD_FEATURES`.

---

### Task 1: Quantile band model + flags + metrics (TDD)

**Files:**
- Create: `modeling/school_overuse_model.py`
- Create: `tests/test_school_overuse_model.py`

**Interfaces:**
- Consumes: reused interfaces above.
- Produces:
  - `DAY_AHEAD_FEATURES: list[str]` (= `DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS`)
  - `QUANTILES = (0.1, 0.5, 0.9)`
  - `FEATURE_LABELS_KO: dict[str, str]` (English feature -> Korean label; see code below)
  - `default_quantile_model_factory(alpha: float) -> object` (LGBMRegressor quantile, or `QuantileFallbackRegressor`)
  - `train_quantile_band(train, validation, quantiles=QUANTILES, model_factory=default_quantile_model_factory) -> QuantileBandResult`
    where `QuantileBandResult` is a dataclass with fields: `predictions: pd.DataFrame` (validation rows + columns `p10_kwh,p50_kwh,p90_kwh`), `p50_model: object`, `x_validation: pd.DataFrame`, `feature_columns: list[str]`, `shap_available: bool` (set later), `used_fallback: bool`.
  - `flag_overuse(predictions: pd.DataFrame) -> pd.DataFrame` adds `actual_kwh,in_normal_band,is_overuse,is_underuse,exceedance_kwh,exceedance_pct,band_position`.
  - `compute_band_metrics(predictions: pd.DataFrame) -> dict` with keys `coverage, pinball_p10, pinball_p50, pinball_p90, p50_wape, p50_mae_kwh, p50_rmse_kwh, p50_bias_pct, overuse_hours, underuse_hours, overuse_total_exceedance_kwh, mean_band_width_kwh, actual_sum_kwh, n_rows`.

- [ ] **Step 1: Write failing tests** — `tests/test_school_overuse_model.py`

```python
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
```

- [ ] **Step 2: Run tests, confirm they fail** — `.venv\Scripts\python.exe -m pytest tests/test_school_overuse_model.py -q` → expect ImportError / AttributeError (module + functions missing).

- [ ] **Step 3: Implement `modeling/school_overuse_model.py` (data + features + band + flags + metrics)**

Key content to implement:

```python
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
```

- [ ] **Step 4: Run tests, confirm pass** — `.venv\Scripts\python.exe -m pytest tests/test_school_overuse_model.py -q` → all pass.

- [ ] **Step 5: Commit** — `git add modeling/school_overuse_model.py tests/test_school_overuse_model.py` then commit `feat: add quantile band model with over-use flags and metrics`.

---

### Task 2: SHAP explanations, run pipeline, outputs + web JSON bundle (TDD)

**Files:**
- Modify: `modeling/school_overuse_model.py`
- Create: `scripts/run_school_overuse_model.py`
- Test: `tests/test_school_overuse_model.py` (add cases)

**Interfaces:**
- Consumes: Task 1 interfaces.
- Produces:
  - `explain_band(p50_model, x_validation, predictions, feature_columns, top_k=5) -> tuple[pd.DataFrame, pd.DataFrame, bool]` returning `(explanations_long, global_importance, shap_available)`. `explanations_long` columns: `timestamp, report_date, hour, rank, feature, feature_label_ko, shap_value_kwh, feature_value, direction` (only over-use rows). `global_importance` columns: `feature, feature_label_ko, mean_abs_shap`.
  - `run_school_overuse_model(data_path=..., output_dir=..., validation_start=..., validation_end=..., web_dir=Path("school_overuse_web")) -> SchoolOveruseRunResult` dataclass: `output_dir, validation_rows, validation_days, coverage, overuse_hours, overuse_total_exceedance_kwh, p50_wape, used_fallback, shap_available`.
  - Writes: `outputs/school_overuse_predictions.csv`, `outputs/school_overuse_explanations.csv`, `outputs/school_overuse_feature_importance.csv`, `outputs/school_overuse_daily_summary.csv`, `outputs/school_overuse_metrics.json`, `outputs/school_overuse_run_summary.json`, and `school_overuse_web/data/monitor.json`.

**`monitor.json` schema (the model↔web contract — implement EXACTLY):**

```json
{
  "meta": {"validation_start","validation_end","validation_rows","validation_days",
           "quantiles":[0.1,0.5,0.9],"base_model":"lightgbm_quantile_day_ahead_weather_academic",
           "feature_contract":"day-ahead","shap_available":true,"used_fallback":false,
           "generated_note":"..."},
  "metrics": { ...all keys from compute_band_metrics..., "coverage_target":0.8 },
  "baselines": [ {"model":"P50 (quantile median)","wape":..,"mae_kwh":..,"rmse_kwh":..},
                 {"model":"naive_last_day_same_hour","wape":..,"mae_kwh":..,"rmse_kwh":..},
                 {"model":"naive_last_week_same_hour","wape":..,"mae_kwh":..,"rmse_kwh":..},
                 {"model":"naive_same_hour_7d_mean","wape":..,"mae_kwh":..,"rmse_kwh":..} ],
  "series": [ {"timestamp":"2026-06-01 00:00:00","report_date":"2026-06-01","hour":0,
               "actual":..,"p10":..,"p50":..,"p90":..,"in_band":true,"is_overuse":false,
               "exceedance_kwh":0.0,"exceedance_pct":0.0,"band_position":0.4}, ... ],
  "overuse": [ {"timestamp":..,"report_date":..,"hour":..,"actual":..,"p50":..,"p90":..,
                "exceedance_kwh":..,"exceedance_pct":..,
                "explanations":[ {"feature":"weather_temp_mean_c","label_ko":"평균기온",
                                  "shap_kwh":120.5,"feature_value":31.2,"direction":"up"}, ... ]}, ... ],
  "feature_importance": [ {"feature":..,"label_ko":..,"mean_abs_shap":..}, ... ],
  "glossary": [ {"term":"분위수(Quantile)","desc":"..."}, ... ]
}
```

`series` is sorted by timestamp; `overuse` is sorted by `exceedance_kwh` desc; numbers rounded to 2 decimals; `baselines` reuse the naive prediction columns already present on the validation frame (`pred_last_day_same_hour_kwh`, `pred_last_week_same_hour_kwh`, `pred_same_hour_7d_mean_kwh`).

- [ ] **Step 1: Write failing tests** (append to `tests/test_school_overuse_model.py`)

```python
def test_explain_band_schema(tmp_path):
    data_path = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")
    frame = som.build_modeling_frame(data_path)
    train, validation = som.split_train_validation(frame, "2026-06-01 00:00:00", "2026-06-20 23:00:00")
    band = som.train_quantile_band(train, validation)
    flagged = som.flag_overuse(band.predictions)
    explanations, importance, shap_available = som.explain_band(
        band.p50_model, band.x_validation, flagged, band.feature_columns, top_k=5
    )
    assert set(["timestamp","rank","feature","feature_label_ko","shap_value_kwh","feature_value","direction"]).issubset(explanations.columns)
    assert (explanations["rank"] >= 1).all() and (explanations["rank"] <= 5).all()
    assert set(["feature","feature_label_ko","mean_abs_shap"]).issubset(importance.columns)
    # every explained timestamp is an over-use hour
    overuse_ts = set(flagged.loc[flagged["is_overuse"], "timestamp"])
    assert set(explanations["timestamp"]).issubset(overuse_ts)


def test_run_school_overuse_model_writes_artifacts(tmp_path):
    out = tmp_path / "outputs"
    web = tmp_path / "school_overuse_web"
    result = som.run_school_overuse_model(output_dir=out, web_dir=web)
    for name in ["school_overuse_predictions.csv","school_overuse_explanations.csv",
                 "school_overuse_feature_importance.csv","school_overuse_daily_summary.csv",
                 "school_overuse_metrics.json","school_overuse_run_summary.json"]:
        assert (out / name).exists()
    monitor = web / "data" / "monitor.json"
    assert monitor.exists()
    import json
    bundle = json.loads(monitor.read_text(encoding="utf-8"))
    assert set(["meta","metrics","baselines","series","overuse","feature_importance","glossary"]).issubset(bundle)
    assert bundle["meta"]["validation_rows"] == len(bundle["series"])
    assert 0.0 <= bundle["metrics"]["coverage"] <= 1.0
    assert result.validation_rows == bundle["meta"]["validation_rows"]
```

- [ ] **Step 2: Run tests, confirm they fail** (AttributeError: `explain_band`/`run_school_overuse_model` missing).

- [ ] **Step 3: Implement `explain_band`, JSON bundle builder, `run_school_overuse_model`, output saving.**

`explain_band` notes: use `shap.TreeExplainer(p50_model)`; compute SHAP for the over-use subset of `x_validation` (align by index with `flagged`). For each over-use row, take top_k features by `|shap|`, `direction = "up" if shap>0 else "down"`, `feature_value` from `x_validation`. If shap import fails OR `p50_model` is `QuantileFallbackRegressor`, fall back to `getattr(p50_model, "feature_importances_", None)`; if that's also missing, return empty `explanations_long`, a `global_importance` of zeros, and `shap_available=False`. Build glossary list from the terms in the spec §4. Provide `GLOSSARY: list[dict]` constant in the module so the web test and bundle share one source.

- [ ] **Step 4: Implement `scripts/run_school_overuse_model.py`** (mirror `scripts/run_school_hourly_model.py` shape):

```python
from pathlib import Path
from modeling.school_overuse_model import run_school_overuse_model

if __name__ == "__main__":
    result = run_school_overuse_model()
    print("output_dir", result.output_dir)
    print("validation_rows", result.validation_rows)
    print("coverage", round(result.coverage, 4))
    print("overuse_hours", result.overuse_hours)
    print("p50_wape", round(result.p50_wape, 4))
    print("shap_available", result.shap_available)
```

- [ ] **Step 5: Run tests + the script for real**
  - `.venv\Scripts\python.exe -m pytest tests/test_school_overuse_model.py -q` → pass
  - `.venv\Scripts\python.exe scripts/run_school_overuse_model.py` → prints metrics; creates real `outputs/school_overuse_*` and `school_overuse_web/data/monitor.json`.

- [ ] **Step 6: Commit** — `git add modeling/school_overuse_model.py tests/test_school_overuse_model.py scripts/run_school_overuse_model.py outputs/school_overuse_*.csv outputs/school_overuse_*.json school_overuse_web/data/monitor.json` then commit `feat: add SHAP explanations, run pipeline, and web data bundle`.

---

### Task 3: New static website (TDD on content)

**Files:**
- Create: `school_overuse_web/index.html`
- Create: `school_overuse_web/styles.css`
- Create: `school_overuse_web/app.js`
- Create: `tests/test_school_overuse_web_content.py`

**Interfaces:**
- Consumes: `school_overuse_web/data/monitor.json` (schema from Task 2).
- Produces: a static viewer opened via a local static server (fetch reads `data/monitor.json`).

The site has 5 tabs: 개요 / 시계열 모니터 / 과다사용 분석 / 모델 정확도 / 용어 설명. Vanilla JS, no framework, no build. Charts drawn as inline SVG. Korean-first copy. Load JSON with `fetch('data/monitor.json')`.

- [ ] **Step 1: Write failing web-content test** — `tests/test_school_overuse_web_content.py`

```python
from pathlib import Path

WEB = Path("school_overuse_web")


def test_web_files_exist():
    assert (WEB / "index.html").exists()
    assert (WEB / "styles.css").exists()
    assert (WEB / "app.js").exists()


def test_index_has_tabs_and_title():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for needle in ["과다사용", "정상 범위", "개요", "시계열", "과다사용 분석", "모델 정확도", "용어 설명"]:
        assert needle in html


def test_app_reads_monitor_json_and_key_terms():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "data/monitor.json" in js
    # references the contract fields it must render
    for field in ["series", "overuse", "feature_importance", "glossary", "p10", "p90", "exceedance"]:
        assert field in js


def test_glossary_terms_present_in_html_or_js():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    blob = html + js
    for term in ["분위수", "SHAP", "WAPE", "day-ahead", "CDD"]:
        assert term in blob
```

- [ ] **Step 2: Run test, confirm fail** (files missing).

- [ ] **Step 3: Implement the three web files.** Requirements:
  - **개요 tab:** KPI cards — 검증기간, 검증 시간 수, 정상밴드 적중률(coverage, %), 과다사용 시간 수, 총 초과 kWh, P50 정확도(WAPE %). Include a one-paragraph plain-Korean explanation of what the page shows.
  - **시계열 모니터 tab:** date `<select>` populated from distinct `report_date` in `series`; an inline SVG chart per selected day showing the `[p10,p90]` band as a shaded area, `p50` as a dashed line, `actual` as a solid line, over-use hours marked with red dots; hover/tooltip showing actual/p10/p50/p90/exceedance. A legend.
  - **과다사용 분석 tab:** table of `overuse` rows (시각, 실제, 정상상한 P90, 초과 kWh, 초과 %) sorted desc; clicking a row renders a horizontal SHAP bar chart of its `explanations` (up=red, down=blue) with Korean `label_ko`, plus a summary sentence: `"정상 기대 P50 {p50}kWh, 정상상한 P90 {p90}kWh, 실제 {actual}kWh → 상한보다 {exceedance}kWh 초과"`. Include the SHAP-framing note from spec §5.
  - **모델 정확도 tab:** table from `baselines` (model, WAPE %, MAE, RMSE) highlighting P50; a coverage calibration line ("정상밴드 적중률 {coverage}% / 목표 80%").
  - **용어 설명 tab:** render `glossary` as term/description cards.
  - If `meta.used_fallback` or `!meta.shap_available`, show a small banner noting the explanation fell back to feature importance.
  - `styles.css`: clean responsive dashboard, readable Korean typography, accessible color contrast. No external CDN dependencies.

- [ ] **Step 4: Verify** — `.venv\Scripts\python.exe -m pytest tests/test_school_overuse_web_content.py -q` → pass. Then browser smoke check: serve with `.venv\Scripts\python.exe -m http.server 8765 --directory school_overuse_web` and confirm the page loads `monitor.json`, the chart renders, and clicking an over-use row shows its SHAP bars (controller will drive the browser check).

- [ ] **Step 5: Commit** — `git add school_overuse_web/index.html school_overuse_web/styles.css school_overuse_web/app.js tests/test_school_overuse_web_content.py` then commit `feat: add over-use monitor static web viewer`.

---

### Task 4: Run record + full verification

**Files:**
- Create: `docs/runs/2026-06-30-school-overuse-v1.md`

**Interfaces:**
- Consumes: `outputs/school_overuse_metrics.json`, `outputs/school_overuse_run_summary.json`, and the comparison/coverage numbers.
- Produces: Korean-first run documentation.

- [ ] **Step 1: Generate/confirm outputs** — ensure `.venv\Scripts\python.exe scripts/run_school_overuse_model.py` ran and outputs exist; read the real metrics.
- [ ] **Step 2: Write the run record** from ACTUAL metrics (no invented numbers): 목적, 검증 기준(2026-06-01~06-20, 480시간/20일), 모델(LightGBM 분위수 P10/P50/P90, day-ahead 계약), 정상밴드 적중률(coverage)·pinball·P50 WAPE vs naive, 과다사용 시간 수·총 초과 kWh, SHAP 설명 방식과 §5 해석 주의, 산출물 목록, 웹 사용법, 한계(밴드는 day-ahead 기준 기대치이며 operational 모델과 다름).
- [ ] **Step 3: Full verification** — run `.venv\Scripts\python.exe -m pytest tests/test_school_overuse_model.py tests/test_school_overuse_web_content.py -q` (all pass) AND `.venv\Scripts\python.exe -m pytest tests/ -q` to confirm no regressions in the existing suite. Record results in the run doc.
- [ ] **Step 4: Commit** — `git add docs/runs/2026-06-30-school-overuse-v1.md` then commit `docs: add school over-use run record v1`.

---

## Self-Review (author)

- Spec coverage: band model (T1), flags/metrics (T1), SHAP + outputs + bundle (T2), web with glossary + term explanations (T3), run doc + verification (T4). ✔
- Leakage guard tested (T1). ✔ Coverage interpretation documented (T4). ✔
- JSON contract pinned in T2 and consumed in T3 with matching field names (`series/overuse/feature_importance/glossary/p10/p90/exceedance`). ✔
- No placeholders; test code and key implementation provided. ✔
