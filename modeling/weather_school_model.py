from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from modeling.weather_features import (
    aggregate_forecast_weather,
    normalize_actual_weather,
    select_forecast_snapshots,
)


DEFAULT_PANEL_PATH = Path("10_baseline_ready_panel_primary_reliable.csv")
DEFAULT_SCHOOL_TOTAL_PATH = Path("13_school_total_daily_clean.csv")
DEFAULT_MONTHLY_PROFILE_PATH = Path("06_monthly_profile_primary_reliable.csv")
DEFAULT_ACTUAL_WEATHER_PATH = Path("outputs/weather_actual_daily.csv")
DEFAULT_FORECAST_WEATHER_PATH = Path("outputs/weather_forecast_daily.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")

VALIDATION_START = pd.Timestamp("2026-04-01")
VALIDATION_END = pd.Timestamp("2026-06-23")
PREDICTION_COLUMN = "pred_weather_school_kwh"

WEATHER_FEATURE_COLUMNS = [
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "humidity_mean_pct",
    "rainfall_mm",
    "wind_mean_mps",
    "cdd_18",
    "hdd_18",
    "is_rainy",
]

MODEL_FEATURE_COLUMNS = [
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "day_of_year",
    "school_multiplier_lag_1d",
    "school_multiplier_lag_7d_mean",
    *WEATHER_FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class WeatherSchoolRunResult:
    output_dir: Path
    evaluation_rows: int
    evaluation_start: str | None
    evaluation_end: str | None
    building_count: int


class MedianMultiplierRegressor:
    def __init__(self) -> None:
        self.median_: float = 1.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "MedianMultiplierRegressor":
        median = pd.to_numeric(y_train, errors="coerce").median()
        if pd.notna(median):
            self.median_ = float(median)
        return self

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.median_, len(x_test))


ModelFactory = Callable[[], object]


def default_model_factory() -> object:
    try:
        import lightgbm as lgb
    except ImportError:
        return MedianMultiplierRegressor()
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=9,
        min_child_samples=10,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=20260628,
        verbosity=-1,
    )


def build_daily_building_base(
    monthly_profiles: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    profiles = monthly_profiles.copy()
    profiles["month"] = pd.to_numeric(profiles["month"], errors="coerce").astype("Int64")
    profiles["profile_monthly_kwh_mean"] = pd.to_numeric(
        profiles["profile_monthly_kwh_mean"],
        errors="coerce",
    )

    rows: list[pd.DataFrame] = []
    for date_value in sorted(pd.to_datetime(pd.Series(list(dates))).dt.normalize().unique()):
        date = pd.Timestamp(date_value)
        month_profiles = profiles[profiles["month"] == date.month].copy()
        if month_profiles.empty:
            continue
        days = (
            pd.to_numeric(month_profiles.get("calendar_days_in_month"), errors="coerce")
            if "calendar_days_in_month" in month_profiles.columns
            else pd.Series(np.nan, index=month_profiles.index)
        )
        month_profiles["calendar_days_in_month"] = days.fillna(date.days_in_month)
        month_profiles["building_base_kwh"] = (
            month_profiles["profile_monthly_kwh_mean"]
            / month_profiles["calendar_days_in_month"]
        )
        month_profiles["date"] = date
        month_profiles["report_month"] = date.strftime("%Y-%m")
        rows.append(
            month_profiles[
                [
                    "date",
                    "report_month",
                    "building_name_recent",
                    "profile_monthly_kwh_mean",
                    "calendar_days_in_month",
                    "building_base_kwh",
                ]
            ],
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "report_month",
                "building_name_recent",
                "profile_monthly_kwh_mean",
                "calendar_days_in_month",
                "building_base_kwh",
            ],
        )
    return pd.concat(rows, ignore_index=True)


def build_school_training_frame(
    school_total: pd.DataFrame,
    monthly_profiles: pd.DataFrame,
    actual_weather: pd.DataFrame,
    validation_start: pd.Timestamp = VALIDATION_START,
) -> pd.DataFrame:
    school = _prepare_school_total_with_multipliers(school_total, monthly_profiles)
    training = school[
        (school["date"] < pd.Timestamp(validation_start))
        & school["school_total_daily_kwh"].notna()
        & school["school_total_is_usable"].map(_is_truthy)
    ].copy()
    weather = _prepare_actual_weather(actual_weather)
    training = training.merge(weather, on="date", how="inner")
    training = _add_calendar_features(training)
    return _fill_model_features(training)


def build_weather_school_predictions(
    school_total: pd.DataFrame,
    monthly_profiles: pd.DataFrame,
    actual_weather: pd.DataFrame,
    forecast_weather: pd.DataFrame,
    validation_panel: pd.DataFrame,
    validation_start: pd.Timestamp = VALIDATION_START,
    validation_end: pd.Timestamp = VALIDATION_END,
    model_factory: ModelFactory = default_model_factory,
) -> pd.DataFrame:
    validation = _prepare_validation_panel(validation_panel, validation_start, validation_end)
    target_dates = sorted(validation["date"].drop_duplicates())
    if not target_dates:
        raise RuntimeError("No validation target dates were found.")

    training = build_school_training_frame(
        school_total=school_total,
        monthly_profiles=monthly_profiles,
        actual_weather=actual_weather,
        validation_start=validation_start,
    )
    if training.empty:
        raise RuntimeError("No pre-validation training rows were produced.")

    forecast = _prepare_forecast_weather(forecast_weather)
    school = _prepare_school_total_with_multipliers(school_total, monthly_profiles)
    date_features = (
        pd.DataFrame({"date": pd.to_datetime(target_dates)})
        .merge(school[_school_lag_columns()], on="date", how="left")
        .merge(forecast, on="date", how="inner")
    )
    date_features = _add_calendar_features(date_features)
    date_features = _fill_model_features(date_features, training)

    model = model_factory()
    x_train = training[MODEL_FEATURE_COLUMNS]
    y_train = pd.to_numeric(training["school_multiplier"], errors="coerce")
    model.fit(x_train, y_train)

    x_eval = date_features[MODEL_FEATURE_COLUMNS]
    date_features["pred_school_multiplier"] = np.maximum(model.predict(x_eval), 0)

    building_base = build_daily_building_base(monthly_profiles, target_dates)
    school_base = _calculate_school_base_totals(building_base)
    date_predictions = date_features.merge(school_base, on="date", how="left")
    date_predictions["pred_school_total_kwh"] = (
        date_predictions["school_base_total_kwh"]
        * date_predictions["pred_school_multiplier"]
    )

    predictions = building_base.merge(
        date_predictions[
            [
                "date",
                "weather_source",
                "issued_at",
                "pred_school_multiplier",
                "school_base_total_kwh",
                "pred_school_total_kwh",
            ]
        ],
        on="date",
        how="inner",
    )
    predictions[PREDICTION_COLUMN] = (
        predictions["building_base_kwh"] * predictions["pred_school_multiplier"]
    )
    predictions = predictions.merge(
        validation[
            [
                "date",
                "report_month",
                "building_name_recent",
                "usage_kwh_clean",
                "is_validation_target_clean",
            ]
        ],
        on=["date", "report_month", "building_name_recent"],
        how="inner",
    )
    return predictions.sort_values(["date", "building_name_recent"]).reset_index(drop=True)


def calculate_weather_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.dropna(subset=["usage_kwh_clean", PREDICTION_COLUMN]).copy()
    actual = pd.to_numeric(scored["usage_kwh_clean"], errors="coerce")
    pred = pd.to_numeric(scored[PREDICTION_COLUMN], errors="coerce")
    residual = pred - actual
    abs_error = residual.abs()
    actual_sum = actual.sum()
    return pd.DataFrame(
        [
            {
                "model": "weather_school_shape",
                "prediction_column": PREDICTION_COLUMN,
                "n_rows": int(len(scored)),
                "actual_sum_kwh": float(actual_sum),
                "pred_sum_kwh": float(pred.sum()),
                "mae_kwh": float(abs_error.mean()),
                "rmse_kwh": float(np.sqrt((residual**2).mean())),
                "wape": float(abs_error.sum() / actual_sum) if actual_sum else np.nan,
                "bias_kwh_mean_pred_minus_actual": float(residual.mean()),
                "bias_pct_sum_pred_minus_actual": float(residual.sum() / actual_sum)
                if actual_sum
                else np.nan,
            }
        ],
    )


def calculate_weather_group_metrics(
    predictions: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(calculate_weather_model_metrics(group).iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def save_weather_outputs(
    predictions: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_dir / "weather_school_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_weather_model_metrics(predictions).to_csv(
        output_dir / "weather_school_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_weather_group_metrics(predictions, ["report_month"]).to_csv(
        output_dir / "weather_school_metrics_by_month.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_weather_group_metrics(predictions, ["building_name_recent"]).to_csv(
        output_dir / "weather_school_metrics_by_building.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_weather_school_model(
    panel_path: Path = DEFAULT_PANEL_PATH,
    school_total_path: Path = DEFAULT_SCHOOL_TOTAL_PATH,
    monthly_profile_path: Path = DEFAULT_MONTHLY_PROFILE_PATH,
    actual_weather_path: Path = DEFAULT_ACTUAL_WEATHER_PATH,
    forecast_weather_path: Path = DEFAULT_FORECAST_WEATHER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_factory: ModelFactory = default_model_factory,
) -> WeatherSchoolRunResult:
    predictions = build_weather_school_predictions(
        school_total=pd.read_csv(school_total_path),
        monthly_profiles=pd.read_csv(monthly_profile_path),
        actual_weather=load_actual_weather_table(actual_weather_path),
        forecast_weather=load_forecast_weather_table(forecast_weather_path),
        validation_panel=pd.read_csv(panel_path),
        model_factory=model_factory,
    )
    save_weather_outputs(predictions, output_dir)
    return WeatherSchoolRunResult(
        output_dir=output_dir,
        evaluation_rows=int(len(predictions)),
        evaluation_start=predictions["date"].min().strftime("%Y-%m-%d")
        if not predictions.empty
        else None,
        evaluation_end=predictions["date"].max().strftime("%Y-%m-%d")
        if not predictions.empty
        else None,
        building_count=int(predictions["building_name_recent"].nunique()),
    )


def load_actual_weather_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "temp_mean_c" in frame.columns and "weather_source" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
    return normalize_actual_weather(frame)


def load_forecast_weather_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "temp_mean_c" in frame.columns and "weather_source" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        if "issued_at" in frame.columns:
            frame["issued_at"] = pd.to_datetime(frame["issued_at"])
        return frame
    selected = select_forecast_snapshots(frame)
    return aggregate_forecast_weather(selected)


def _prepare_school_total_with_multipliers(
    school_total: pd.DataFrame,
    monthly_profiles: pd.DataFrame,
) -> pd.DataFrame:
    school = school_total.copy()
    school["date"] = pd.to_datetime(school["date"])
    school["school_total_daily_kwh"] = pd.to_numeric(
        school["school_total_daily_kwh"],
        errors="coerce",
    )
    if "calendar_days_in_month" not in school.columns:
        school["calendar_days_in_month"] = school["date"].dt.days_in_month
    school["calendar_days_in_month"] = pd.to_numeric(
        school["calendar_days_in_month"],
        errors="coerce",
    ).fillna(school["date"].dt.days_in_month)
    if "school_total_is_usable" not in school.columns:
        school["school_total_is_usable"] = True

    building_base = build_daily_building_base(monthly_profiles, school["date"].unique())
    school_base = _calculate_school_base_totals(building_base)
    school = school.merge(school_base, on="date", how="left")
    if "school_total_day_weight_observed_month" in school.columns:
        school["school_total_day_weight_observed_month"] = pd.to_numeric(
            school["school_total_day_weight_observed_month"],
            errors="coerce",
        )
    else:
        month_sum = school.groupby("report_month")["school_total_daily_kwh"].transform("sum")
        school["school_total_day_weight_observed_month"] = (
            school["school_total_daily_kwh"] / month_sum
        )
    school["school_multiplier"] = (
        school["school_total_day_weight_observed_month"]
        * school["calendar_days_in_month"]
    )
    school = school.sort_values("date").reset_index(drop=True)
    school["school_multiplier_lag_1d"] = school["school_multiplier"].shift(1)
    school["school_multiplier_lag_7d_mean"] = (
        school["school_multiplier"].shift(1).rolling(7, min_periods=1).mean()
    )
    return school


def _calculate_school_base_totals(building_base: pd.DataFrame) -> pd.DataFrame:
    return (
        building_base.groupby("date", as_index=False)["building_base_kwh"]
        .sum()
        .rename(columns={"building_base_kwh": "school_base_total_kwh"})
    )


def _prepare_actual_weather(actual_weather: pd.DataFrame) -> pd.DataFrame:
    weather = actual_weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    weather["weather_source"] = "actual"
    return weather


def _prepare_forecast_weather(forecast_weather: pd.DataFrame) -> pd.DataFrame:
    weather = forecast_weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    if "issued_at" not in weather.columns:
        weather["issued_at"] = pd.NaT
    weather["issued_at"] = pd.to_datetime(weather["issued_at"])
    weather["weather_source"] = "forecast"
    return weather


def _prepare_validation_panel(
    validation_panel: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    validation = validation_panel.copy()
    validation["date"] = pd.to_datetime(validation["date"])
    validation["usage_kwh_clean"] = pd.to_numeric(validation["usage_kwh_clean"], errors="coerce")
    mask = (
        validation["is_validation_target_clean"].map(_is_truthy)
        & (validation["date"] >= pd.Timestamp(validation_start))
        & (validation["date"] <= pd.Timestamp(validation_end))
    )
    return validation.loc[mask].copy()


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["day_of_week"] = result["date"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["day_of_year"] = result["date"].dt.dayofyear
    return result


def _fill_model_features(
    frame: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if reference is None:
        reference = result
    for column in MODEL_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
        fill_value = pd.to_numeric(reference.get(column), errors="coerce").median()
        if pd.isna(fill_value):
            fill_value = 0.0
        result[column] = result[column].fillna(fill_value)
    return result


def _school_lag_columns() -> list[str]:
    return ["date", "school_multiplier_lag_1d", "school_multiplier_lag_7d_mean"]


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe CEMS weather school-shape model.",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--school-total", type=Path, default=DEFAULT_SCHOOL_TOTAL_PATH)
    parser.add_argument("--monthly-profile", type=Path, default=DEFAULT_MONTHLY_PROFILE_PATH)
    parser.add_argument("--actual-weather", type=Path, default=DEFAULT_ACTUAL_WEATHER_PATH)
    parser.add_argument("--forecast-weather", type=Path, default=DEFAULT_FORECAST_WEATHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_weather_school_model(
        panel_path=args.panel,
        school_total_path=args.school_total,
        monthly_profile_path=args.monthly_profile,
        actual_weather_path=args.actual_weather,
        forecast_weather_path=args.forecast_weather,
        output_dir=args.output_dir,
    )
    print(f"output_dir={result.output_dir}")
    print(f"evaluation_rows={result.evaluation_rows}")
    print(f"building_count={result.building_count}")
    print(f"evaluation_range={result.evaluation_start}..{result.evaluation_end}")


if __name__ == "__main__":
    main()
