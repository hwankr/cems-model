from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from modeling.academic_calendar_features import ACADEMIC_FEATURE_COLUMNS


DEFAULT_DATA_PATH = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_ACTUAL_WEATHER_PATH = Path("outputs/weather_actual_daily.csv")
DEFAULT_FORECAST_WEATHER_PATH = Path("outputs/weather_forecast_daily.csv")
DEFAULT_ACADEMIC_FEATURES_PATH = Path("outputs/yu_academic_calendar_daily_features.csv")
DEFAULT_VALIDATION_START = "2026-06-01 00:00:00"
DEFAULT_VALIDATION_END = "2026-06-29 23:00:00"

TARGET_COLUMN = "usage_kwh"
LIGHTGBM_DAY_AHEAD_COLUMN = "pred_lightgbm_school_hourly_day_ahead_kwh"
LIGHTGBM_DAY_AHEAD_WEATHER_COLUMN = "pred_lightgbm_school_hourly_day_ahead_weather_kwh"
LIGHTGBM_DAY_AHEAD_WEATHER_ACADEMIC_COLUMN = (
    "pred_lightgbm_school_hourly_day_ahead_weather_academic_kwh"
)
LIGHTGBM_OPERATIONAL_COLUMN = "pred_lightgbm_school_hourly_operational_kwh"
LIGHTGBM_PREDICTION_COLUMN = LIGHTGBM_DAY_AHEAD_COLUMN
BASELINE_PREDICTION_COLUMNS = {
    "naive_last_day_same_hour": "pred_last_day_same_hour_kwh",
    "naive_last_week_same_hour": "pred_last_week_same_hour_kwh",
    "naive_same_hour_7d_mean": "pred_same_hour_7d_mean_kwh",
}
COMPARISON_COLUMNS = {
    **BASELINE_PREDICTION_COLUMNS,
    "lightgbm_school_hourly_day_ahead": LIGHTGBM_DAY_AHEAD_COLUMN,
    "lightgbm_school_hourly_day_ahead_weather": LIGHTGBM_DAY_AHEAD_WEATHER_COLUMN,
    "lightgbm_school_hourly_day_ahead_weather_academic": (
        LIGHTGBM_DAY_AHEAD_WEATHER_ACADEMIC_COLUMN
    ),
    "lightgbm_school_hourly_operational": LIGHTGBM_OPERATIONAL_COLUMN,
}

CALENDAR_FEATURE_COLUMNS = [
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

DAY_AHEAD_FEATURE_COLUMNS = [
    *CALENDAR_FEATURE_COLUMNS,
    "school_lag_24h_kwh",
    "school_lag_168h_kwh",
    "school_same_hour_7d_mean_kwh",
]

WEATHER_FEATURE_COLUMNS = [
    "weather_temp_mean_c",
    "weather_temp_min_c",
    "weather_temp_max_c",
    "weather_humidity_mean_pct",
    "weather_rainfall_mm",
    "weather_wind_mean_mps",
    "weather_cdd_18",
    "weather_hdd_18",
    "weather_is_rainy",
]

DAY_AHEAD_WEATHER_FEATURE_COLUMNS = [
    *DAY_AHEAD_FEATURE_COLUMNS,
    *WEATHER_FEATURE_COLUMNS,
]

DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS = [
    *DAY_AHEAD_WEATHER_FEATURE_COLUMNS,
    *ACADEMIC_FEATURE_COLUMNS,
]

OPERATIONAL_FEATURE_COLUMNS = [
    *CALENDAR_FEATURE_COLUMNS,
    "school_lag_1h_kwh",
    "school_lag_24h_kwh",
    "school_lag_168h_kwh",
    "school_rolling_24h_mean_kwh",
    "school_rolling_168h_mean_kwh",
    "school_same_hour_7d_mean_kwh",
]
FEATURE_COLUMNS = OPERATIONAL_FEATURE_COLUMNS
OFFICIAL_DAY_AHEAD_PREDICTION_COLUMNS = [
    LIGHTGBM_DAY_AHEAD_COLUMN,
    LIGHTGBM_DAY_AHEAD_WEATHER_COLUMN,
    LIGHTGBM_DAY_AHEAD_WEATHER_ACADEMIC_COLUMN,
]

REQUIRED_COLUMNS = {
    "timestamp",
    "period_end",
    "measurement_date",
    "interval_minutes",
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "hour",
    "minute",
    TARGET_COLUMN,
}


@dataclass(frozen=True)
class SchoolHourlyRunResult:
    data_path: Path
    output_dir: Path
    training_rows: int
    validation_rows: int
    validation_start: str | None
    validation_end: str | None
    champion_model: str
    champion_wape: float


class MedianRegressor:
    def __init__(self) -> None:
        self.median_: float = 0.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "MedianRegressor":
        median = pd.to_numeric(y_train, errors="coerce").median()
        self.median_ = float(median) if pd.notna(median) else 0.0
        return self

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.median_, len(x_test))


ModelFactory = Callable[[], object]


def default_model_factory() -> object:
    try:
        import lightgbm as lgb
    except ImportError:
        return MedianRegressor()

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.92,
        colsample_bytree=0.92,
        random_state=20260630,
        verbosity=-1,
    )


def load_hourly_usage(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    result["measurement_date"] = pd.to_datetime(result["measurement_date"], errors="coerce")
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    result["interval_minutes"] = pd.to_numeric(result["interval_minutes"], errors="coerce")

    result = result.dropna(subset=["timestamp", TARGET_COLUMN])
    result = result[result["interval_minutes"] == 60].copy()
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    result["date"] = result["timestamp"].dt.normalize()
    result["report_date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["report_month"] = result["timestamp"].dt.strftime("%Y-%m")
    result["year"] = result["timestamp"].dt.year
    result["month"] = result["timestamp"].dt.month
    result["day"] = result["timestamp"].dt.day
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["hour"] = result["timestamp"].dt.hour
    result["minute"] = result["timestamp"].dt.minute
    return result.reset_index(drop=True)


def add_hourly_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    result = result.sort_values("timestamp").reset_index(drop=True)

    if "date" not in result.columns:
        result["date"] = result["timestamp"].dt.normalize()
    if "report_date" not in result.columns:
        result["report_date"] = result["date"].dt.strftime("%Y-%m-%d")
    if "report_month" not in result.columns:
        result["report_month"] = result["timestamp"].dt.strftime("%Y-%m")

    result["year"] = result["timestamp"].dt.year
    result["month"] = result["timestamp"].dt.month
    result["day"] = result["timestamp"].dt.day
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["hour"] = result["timestamp"].dt.hour
    result["minute"] = result["timestamp"].dt.minute
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

    usage = result[TARGET_COLUMN]
    result["school_lag_1h_kwh"] = usage.shift(1)
    result["school_lag_24h_kwh"] = usage.shift(24)
    result["school_lag_168h_kwh"] = usage.shift(168)
    result["school_rolling_24h_mean_kwh"] = usage.shift(1).rolling(24, min_periods=6).mean()
    result["school_rolling_168h_mean_kwh"] = usage.shift(1).rolling(168, min_periods=24).mean()

    same_hour_lags = [usage.shift(24 * days) for days in range(1, 8)]
    result["school_same_hour_7d_mean_kwh"] = pd.concat(same_hour_lags, axis=1).mean(
        axis=1,
        skipna=True,
    )

    result["pred_last_day_same_hour_kwh"] = result["school_lag_24h_kwh"]
    result["pred_last_week_same_hour_kwh"] = result["school_lag_168h_kwh"]
    result["pred_same_hour_7d_mean_kwh"] = result["school_same_hour_7d_mean_kwh"]
    return result


def add_weather_features(
    frame: pd.DataFrame,
    actual_weather: pd.DataFrame | None = None,
    forecast_weather: pd.DataFrame | None = None,
    validation_start: str = DEFAULT_VALIDATION_START,
) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    validation_start_ts = pd.Timestamp(validation_start).normalize()

    weather_frames = []
    if actual_weather is not None and not actual_weather.empty:
        actual = _normalize_weather(actual_weather, default_source="actual")
        weather_frames.append(actual[actual["date"] < validation_start_ts])
    if forecast_weather is not None and not forecast_weather.empty:
        forecast = _normalize_weather(forecast_weather, default_source="forecast")
        weather_frames.append(forecast[forecast["date"] >= validation_start_ts])

    weather_frames = [weather for weather in weather_frames if not weather.empty]
    if not weather_frames:
        return result

    weather = (
        pd.concat(weather_frames, ignore_index=True)
        .sort_values(["date", "weather_feature_source"])
        .drop_duplicates(subset=["date"], keep="last")
    )
    return result.merge(weather, on="date", how="left")


def add_academic_features(
    frame: pd.DataFrame,
    academic_features: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    academic = academic_features.copy()
    academic["date"] = pd.to_datetime(academic["date"])
    keep_columns = [
        "date",
        *[column for column in ACADEMIC_FEATURE_COLUMNS if column in academic.columns],
    ]
    return result.merge(academic[keep_columns], on="date", how="left")


def _normalize_weather(weather: pd.DataFrame, default_source: str) -> pd.DataFrame:
    result = weather.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["weather_feature_source"] = result.get("weather_source", default_source)
    result["weather_feature_source"] = result["weather_feature_source"].fillna(default_source)
    keep_columns = ["date", "weather_feature_source"]

    if "issued_at" in result.columns:
        result["weather_issued_at"] = result["issued_at"]
        keep_columns.append("weather_issued_at")

    rename = {}
    for column in [
        "temp_mean_c",
        "temp_min_c",
        "temp_max_c",
        "humidity_mean_pct",
        "rainfall_mm",
        "wind_mean_mps",
        "cdd_18",
        "hdd_18",
        "is_rainy",
    ]:
        if column in result.columns:
            rename[column] = f"weather_{column}"
            keep_columns.append(column)

    return result[keep_columns].rename(columns=rename)


def split_train_validation(
    frame: pd.DataFrame,
    validation_start: str = DEFAULT_VALIDATION_START,
    validation_end: str = DEFAULT_VALIDATION_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    train = result[result["timestamp"] < start].copy()
    validation = result[
        (result["timestamp"] >= start)
        & (result["timestamp"] <= end)
        & result[TARGET_COLUMN].notna()
    ].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def build_feature_frame(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    reference: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_columns = feature_columns or FEATURE_COLUMNS
    selected_columns = [column for column in selected_columns if column in frame.columns]
    features = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
    if reference is None:
        reference_features = features
    else:
        reference_columns = [column for column in selected_columns if column in reference.columns]
        reference_features = reference[reference_columns].apply(pd.to_numeric, errors="coerce")

    medians = reference_features.median(numeric_only=True)
    features = features.fillna(medians).fillna(0)
    if columns is not None:
        features = features.reindex(columns=columns, fill_value=0)
    return features


def build_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_factory: ModelFactory = default_model_factory,
) -> pd.DataFrame:
    clean_train = train.dropna(subset=[TARGET_COLUMN]).copy()
    if clean_train.empty:
        raise RuntimeError("No training rows are available before validation_start.")
    if validation.empty:
        raise RuntimeError("No validation rows are available in the requested window.")

    y_train = pd.to_numeric(clean_train[TARGET_COLUMN], errors="coerce")

    predictions = validation.copy()
    for prediction_column, feature_columns in _available_variant_specs(clean_train, validation):
        x_train = build_feature_frame(clean_train, feature_columns=feature_columns)
        x_validation = build_feature_frame(
            validation,
            list(x_train.columns),
            reference=clean_train,
            feature_columns=feature_columns,
        )
        model = model_factory()
        model.fit(x_train, y_train)
        predictions[prediction_column] = np.maximum(model.predict(x_validation), 0)

    # Backward-compatible alias for the original operational model name. New
    # consumers should use the explicit day_ahead or operational columns above.
    predictions["pred_lightgbm_school_hourly_kwh"] = predictions[LIGHTGBM_OPERATIONAL_COLUMN]
    predictions["is_validation_target"] = True
    return predictions.reset_index(drop=True)


def _available_variant_specs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[tuple[str, list[str]]]:
    specs = [
        (LIGHTGBM_DAY_AHEAD_COLUMN, DAY_AHEAD_FEATURE_COLUMNS),
        (LIGHTGBM_DAY_AHEAD_WEATHER_COLUMN, DAY_AHEAD_WEATHER_FEATURE_COLUMNS),
        (
            LIGHTGBM_DAY_AHEAD_WEATHER_ACADEMIC_COLUMN,
            DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS,
        ),
        (LIGHTGBM_OPERATIONAL_COLUMN, OPERATIONAL_FEATURE_COLUMNS),
    ]

    available = []
    for prediction_column, feature_columns in specs:
        if all(column in train.columns and column in validation.columns for column in feature_columns):
            available.append((prediction_column, feature_columns))
    return available


def calculate_metric_row(
    frame: pd.DataFrame,
    model_name: str,
    prediction_column: str,
) -> dict[str, float | int | str]:
    scored = frame.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    actual = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
    pred = pd.to_numeric(scored[prediction_column], errors="coerce")
    valid = actual.notna() & pred.notna()
    actual = actual[valid]
    pred = pred[valid]
    residual = pred - actual
    abs_error = residual.abs()
    squared_error = residual**2
    actual_sum = actual.sum()

    return {
        "model": model_name,
        "prediction_method": prediction_column,
        "prediction_column": prediction_column,
        "n_rows": int(len(actual)),
        "actual_sum_kwh": float(actual_sum),
        "pred_sum_kwh": float(pred.sum()),
        "mean_actual_kwh": float(actual.mean()) if len(actual) else np.nan,
        "mae_kwh": float(abs_error.mean()) if len(abs_error) else np.nan,
        "rmse_kwh": float(np.sqrt(squared_error.mean())) if len(squared_error) else np.nan,
        "wape": float(abs_error.sum() / actual_sum) if actual_sum else np.nan,
        "mape": float((abs_error / actual.replace(0, np.nan)).mean()) if len(actual) else np.nan,
        "bias_kwh_mean_pred_minus_actual": float(residual.mean()) if len(residual) else np.nan,
        "bias_pct_sum_pred_minus_actual": float(residual.sum() / actual_sum)
        if actual_sum
        else np.nan,
    }


def calculate_comparison_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, prediction_column in COMPARISON_COLUMNS.items():
        if prediction_column in predictions.columns:
            rows.append(calculate_metric_row(predictions, model_name, prediction_column))
    return pd.DataFrame(rows)


def calculate_group_metrics(
    predictions: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    rows = []
    group_columns = list(group_columns)
    for model_name, prediction_column in COMPARISON_COLUMNS.items():
        if prediction_column not in predictions.columns:
            continue
        for keys, group in predictions.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys, strict=True))
            row.update(calculate_metric_row(group, model_name, prediction_column))
            rows.append(row)
    return pd.DataFrame(rows)


def create_top_errors(
    predictions: pd.DataFrame,
    prediction_column: str = LIGHTGBM_PREDICTION_COLUMN,
    limit: int = 120,
) -> pd.DataFrame:
    ranked = predictions.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    ranked["pred_kwh"] = pd.to_numeric(ranked[prediction_column], errors="coerce")
    ranked["error_kwh"] = ranked["pred_kwh"] - pd.to_numeric(ranked[TARGET_COLUMN], errors="coerce")
    ranked["abs_error_kwh"] = ranked["error_kwh"].abs()
    keep_columns = [
        "timestamp",
        "report_date",
        "report_month",
        "hour",
        TARGET_COLUMN,
        "pred_kwh",
        "error_kwh",
        "abs_error_kwh",
    ]
    return ranked.sort_values("abs_error_kwh", ascending=False)[keep_columns].head(limit)


def save_outputs(
    predictions: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = calculate_comparison_metrics(predictions)
    champion = select_champion_model(comparison)

    predictions.to_csv(
        output_dir / "school_hourly_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        output_dir / "school_hourly_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_group_metrics(predictions, ["hour"]).to_csv(
        output_dir / "school_hourly_metrics_by_hour.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_group_metrics(predictions, ["report_date"]).to_csv(
        output_dir / "school_hourly_metrics_by_day.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_top_errors(predictions, str(champion["prediction_column"])).to_csv(
        output_dir / "school_hourly_top_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model_family": "school_hourly",
        "target_column": TARGET_COLUMN,
        "prediction_column": champion["prediction_column"],
        "champion_model": champion["model"],
        "validation_start": predictions["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_end": predictions["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_rows": int(len(predictions)),
        "validation_days": int(predictions["report_date"].nunique()),
        "champion_wape": champion["wape"],
        "champion_mae_kwh": champion["mae_kwh"],
        "champion_rmse_kwh": champion["rmse_kwh"],
        "official_scoring_note": "2026-06-30 is excluded because the source day is incomplete.",
        "forecast_contract": (
            "Official score uses a day-ahead feature contract that excludes same-day "
            "one-hour lag and rolling features. The operational variant is reported "
            "separately for one-hour rolling updates."
        ),
    }
    (output_dir / "school_hourly_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def select_champion_model(comparison: pd.DataFrame) -> dict[str, object]:
    if comparison.empty:
        raise RuntimeError("No model comparison rows were produced.")
    official = comparison[
        comparison["prediction_column"].astype(str).isin(OFFICIAL_DAY_AHEAD_PREDICTION_COLUMNS)
    ].copy()
    ranked = official.dropna(subset=["wape", "mae_kwh"]).copy()
    if ranked.empty:
        ranked = comparison.dropna(subset=["wape", "mae_kwh"]).copy()
    if ranked.empty:
        raise RuntimeError("No scored model comparison rows were produced.")
    champion = ranked.sort_values(["wape", "mae_kwh"], ascending=True).iloc[0]
    return {
        "model": str(champion["model"]),
        "prediction_column": str(champion["prediction_column"]),
        "n_rows": int(champion["n_rows"]),
        "mae_kwh": float(champion["mae_kwh"]),
        "rmse_kwh": float(champion["rmse_kwh"]),
        "wape": float(champion["wape"]),
        "bias_pct_sum_pred_minus_actual": float(champion["bias_pct_sum_pred_minus_actual"]),
    }


def run_school_hourly_model(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    validation_start: str = DEFAULT_VALIDATION_START,
    validation_end: str = DEFAULT_VALIDATION_END,
    actual_weather_path: Path | None = DEFAULT_ACTUAL_WEATHER_PATH,
    forecast_weather_path: Path | None = DEFAULT_FORECAST_WEATHER_PATH,
    academic_features_path: Path | None = DEFAULT_ACADEMIC_FEATURES_PATH,
    model_factory: ModelFactory = default_model_factory,
) -> SchoolHourlyRunResult:
    hourly = load_hourly_usage(data_path)
    featured = add_hourly_features(hourly)
    actual_weather = load_optional_table(actual_weather_path)
    forecast_weather = load_optional_table(forecast_weather_path)
    if actual_weather is not None or forecast_weather is not None:
        featured = add_weather_features(
            featured,
            actual_weather=actual_weather,
            forecast_weather=forecast_weather,
            validation_start=validation_start,
        )
    academic_features = load_optional_table(academic_features_path)
    if academic_features is not None:
        featured = add_academic_features(featured, academic_features)
    train, validation = split_train_validation(
        featured,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    predictions = build_predictions(train, validation, model_factory=model_factory)
    summary = save_outputs(predictions, output_dir)

    return SchoolHourlyRunResult(
        data_path=data_path,
        output_dir=output_dir,
        training_rows=int(len(train)),
        validation_rows=int(summary["validation_rows"]),
        validation_start=str(summary["validation_start"]),
        validation_end=str(summary["validation_end"]),
        champion_model=str(summary["champion_model"]),
        champion_wape=float(summary["champion_wape"]),
    )


def load_optional_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run school-wide hourly electricity forecasting model.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--actual-weather", type=Path, default=DEFAULT_ACTUAL_WEATHER_PATH)
    parser.add_argument("--forecast-weather", type=Path, default=DEFAULT_FORECAST_WEATHER_PATH)
    parser.add_argument("--academic-features", type=Path, default=DEFAULT_ACADEMIC_FEATURES_PATH)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_school_hourly_model(
        data_path=args.data,
        output_dir=args.output_dir,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        actual_weather_path=args.actual_weather,
        forecast_weather_path=args.forecast_weather,
        academic_features_path=args.academic_features,
    )
    print(f"data_path={result.data_path}")
    print(f"output_dir={result.output_dir}")
    print(f"training_rows={result.training_rows}")
    print(f"validation_rows={result.validation_rows}")
    print(f"validation_range={result.validation_start}..{result.validation_end}")
    print(f"champion_model={result.champion_model}")
    print(f"champion_wape={result.champion_wape:.6f}")


if __name__ == "__main__":
    main()
