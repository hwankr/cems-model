from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("10_baseline_ready_panel_primary_reliable.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")

REALTIME_OBSERVATION_START_DATE = pd.Timestamp("2026-04-01")
WEEKDAY_RECENT_PREDICTION_COLUMN = "pred_weekday_recent_kwh"

PREDICTION_COLUMNS = [
    "pred_realtime_uniform_daily_kwh",
    WEEKDAY_RECENT_PREDICTION_COLUMN,
]

REALTIME_TRAINING_CUTOFF_MONTH = "2026-04"

OBSERVED_SCHOOL_FLOW_COLUMNS = [
    "school_total_daily_kwh",
    "school_total_is_usable",
    "school_total_observed_month_sum_kwh",
    "school_total_observed_day_count",
    "school_total_month_observed_completeness",
    "school_total_day_weight_observed_month",
    "baseline_school_shape_kwh_observed_norm",
]

WEEKDAY_RECENT_FEATURE_COLUMNS = [
    "weekday_recent_prev_day_kwh",
    "weekday_recent_recent_3d_mean_kwh",
    "weekday_recent_recent_7d_mean_kwh",
    "weekday_recent_prev_week_same_weekday_kwh",
    "weekday_recent_weekday_profile_kwh",
]

REQUIRED_COLUMNS = {
    "date",
    "report_month",
    "building_name_recent",
    "usage_kwh_clean",
    "is_validation_target_clean",
    "profile_monthly_kwh_mean",
    "calendar_days_in_month",
}


@dataclass(frozen=True)
class BaselineRunResult:
    data_path: Path
    output_dir: Path
    total_rows: int
    validation_rows: int
    building_count: int
    validation_start: str | None
    validation_end: str | None


def load_panel(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    panel = pd.read_csv(path)
    panel["date"] = pd.to_datetime(panel["date"])
    validate_panel_schema(panel)
    return panel


def validate_panel_schema(panel: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if (pd.to_numeric(panel["calendar_days_in_month"], errors="coerce") <= 0).any():
        raise ValueError("calendar_days_in_month must be positive.")

    if "profile_source_months" in panel.columns:
        future_sources = sorted(
            {
                month
                for value in panel["profile_source_months"].dropna()
                for month in re.findall(r"\d{4}-\d{2}", str(value))
                if month >= REALTIME_TRAINING_CUTOFF_MONTH
            }
        )
        if future_sources:
            raise ValueError(
                "future profile source months are not allowed for realtime baseline: "
                f"{future_sources}"
            )


def add_baseline_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"])

    monthly_profile = pd.to_numeric(result["profile_monthly_kwh_mean"], errors="coerce")
    calendar_days = pd.to_numeric(result["calendar_days_in_month"], errors="coerce")

    result["pred_realtime_uniform_daily_kwh"] = monthly_profile / calendar_days
    result = _add_date_features(result)
    result = _add_weekday_profile_features(result)
    result = _add_recent_usage_features(result)
    result = _add_weekday_recent_prediction(result)

    return result


def _add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["weekday_recent_day_of_week"] = result["date"].dt.dayofweek
    result["weekday_recent_is_weekend"] = result["weekday_recent_day_of_week"] >= 5
    result["weekday_recent_month"] = result["date"].dt.month
    result["weekday_recent_day"] = result["date"].dt.day
    return result


def _add_weekday_profile_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["weekday_recent_weekday_multiplier"] = 1.0

    history = result.loc[result["date"] < REALTIME_OBSERVATION_START_DATE].copy()
    if not history.empty:
        history_usage = pd.to_numeric(history["usage_kwh_clean"], errors="coerce")
        history_baseline = pd.to_numeric(
            history["pred_realtime_uniform_daily_kwh"],
            errors="coerce",
        )
        history["weekday_recent_weekday_multiplier"] = np.where(
            history_baseline > 0,
            history_usage / history_baseline,
            np.nan,
        )
        history = history.dropna(subset=["weekday_recent_weekday_multiplier"])

    if not history.empty:
        same_month_weekday = (
            history.groupby(
                [
                    "building_name_recent",
                    "weekday_recent_month",
                    "weekday_recent_day_of_week",
                ],
                dropna=False,
            )["weekday_recent_weekday_multiplier"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "weekday_recent_weekday_multiplier": "_same_month_weekday_multiplier",
                },
            )
        )
        same_month = (
            history.groupby(
                ["building_name_recent", "weekday_recent_month"],
                dropna=False,
            )["weekday_recent_weekday_multiplier"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "weekday_recent_weekday_multiplier": "_same_month_multiplier",
                },
            )
        )

        result["_original_row_order"] = np.arange(len(result))
        result = result.merge(
            same_month_weekday,
            how="left",
            on=[
                "building_name_recent",
                "weekday_recent_month",
                "weekday_recent_day_of_week",
            ],
        )
        result = result.merge(
            same_month,
            how="left",
            on=["building_name_recent", "weekday_recent_month"],
        )
        result["weekday_recent_weekday_multiplier"] = result[
            "_same_month_weekday_multiplier"
        ].fillna(result["_same_month_multiplier"])
        result["weekday_recent_weekday_multiplier"] = result[
            "weekday_recent_weekday_multiplier"
        ].fillna(1.0)
        result = result.sort_values("_original_row_order").drop(
            columns=[
                "_original_row_order",
                "_same_month_weekday_multiplier",
                "_same_month_multiplier",
            ],
        )

    result["weekday_recent_weekday_profile_kwh"] = (
        pd.to_numeric(result["pred_realtime_uniform_daily_kwh"], errors="coerce")
        * result["weekday_recent_weekday_multiplier"]
    )
    return result


def _add_recent_usage_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["building_name_recent", "date"]).copy()
    result["_weekday_recent_usage_kwh"] = pd.to_numeric(
        result["usage_kwh_clean"],
        errors="coerce",
    )
    grouped_usage = result.groupby("building_name_recent", sort=False)[
        "_weekday_recent_usage_kwh"
    ]

    result["weekday_recent_prev_day_kwh"] = grouped_usage.shift(1)
    result["weekday_recent_recent_3d_mean_kwh"] = grouped_usage.transform(
        lambda values: values.shift(1).rolling(window=3, min_periods=3).mean(),
    )
    result["weekday_recent_recent_7d_mean_kwh"] = grouped_usage.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).mean(),
    )
    result["weekday_recent_prev_week_same_weekday_kwh"] = grouped_usage.shift(7)

    monthly_fallback = pd.to_numeric(
        result["pred_realtime_uniform_daily_kwh"],
        errors="coerce",
    )
    for column in WEEKDAY_RECENT_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(
            monthly_fallback,
        )

    result = result.drop(columns=["_weekday_recent_usage_kwh"])
    return result.sort_index()


def _add_weekday_recent_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    uniform = pd.to_numeric(result["pred_realtime_uniform_daily_kwh"], errors="coerce")
    weekday_profile = pd.to_numeric(
        result["weekday_recent_weekday_profile_kwh"],
        errors="coerce",
    ).fillna(uniform)
    prev_day = pd.to_numeric(result["weekday_recent_prev_day_kwh"], errors="coerce")
    recent_3 = pd.to_numeric(result["weekday_recent_recent_3d_mean_kwh"], errors="coerce")
    recent_7 = pd.to_numeric(result["weekday_recent_recent_7d_mean_kwh"], errors="coerce")
    prev_week = pd.to_numeric(
        result["weekday_recent_prev_week_same_weekday_kwh"],
        errors="coerce",
    )

    weekday_prediction = (
        (0.10 * uniform)
        + (0.10 * weekday_profile)
        + (0.30 * prev_day)
        + (0.25 * recent_3)
        + (0.15 * recent_7)
        + (0.10 * prev_week)
    )
    weekend_prediction = (
        (0.15 * uniform)
        + (0.15 * weekday_profile)
        + (0.15 * prev_day)
        + (0.10 * recent_3)
        + (0.10 * recent_7)
        + (0.35 * prev_week)
    )

    result[WEEKDAY_RECENT_PREDICTION_COLUMN] = np.where(
        result["weekday_recent_is_weekend"],
        weekend_prediction,
        weekday_prediction,
    )
    result[WEEKDAY_RECENT_PREDICTION_COLUMN] = result[
        WEEKDAY_RECENT_PREDICTION_COLUMN
    ].clip(lower=0)
    return result


def build_validation_frame(panel: pd.DataFrame) -> pd.DataFrame:
    validation_mask = panel["is_validation_target_clean"].map(_is_truthy)
    validation = panel.loc[validation_mask].copy()
    return validation.sort_values(["building_name_recent", "date"]).reset_index(drop=True)


def calculate_metrics(
    frame: pd.DataFrame,
    pred_col: str,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    group_cols = list(group_cols)
    scored = frame.dropna(subset=["usage_kwh_clean", pred_col]).copy()
    scored["usage_kwh_clean"] = pd.to_numeric(scored["usage_kwh_clean"], errors="coerce")
    scored[pred_col] = pd.to_numeric(scored[pred_col], errors="coerce")
    scored = scored.dropna(subset=["usage_kwh_clean", pred_col])

    scored["error"] = scored[pred_col] - scored["usage_kwh_clean"]
    scored["abs_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    scored["ape"] = np.where(
        scored["usage_kwh_clean"] > 0,
        scored["abs_error"] / scored["usage_kwh_clean"],
        np.nan,
    )

    rows: list[dict[str, object]] = []
    grouped = scored.groupby(group_cols, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        actual_sum = group["usage_kwh_clean"].sum()
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "prediction_method": pred_col,
                "n_rows": int(len(group)),
                "actual_sum_kwh_clean": float(actual_sum),
                "pred_sum_kwh": float(group[pred_col].sum()),
                "mae_kwh": float(group["abs_error"].mean()),
                "rmse_kwh": float(np.sqrt(group["squared_error"].mean())),
                "mape": float(group["ape"].mean()),
                "bias_kwh_mean": float(group["error"].mean()),
                "bias_pct_sum": float(group["error"].sum() / actual_sum)
                if actual_sum
                else np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def create_error_rankings(
    validation: pd.DataFrame,
    pred_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if pred_cols is None:
        pred_cols = PREDICTION_COLUMNS

    ranking_frames: list[pd.DataFrame] = []
    for pred_col in pred_cols:
        if pred_col not in validation.columns:
            continue

        ranked = validation.copy()
        ranked["prediction_method"] = pred_col
        ranked["pred_kwh"] = pd.to_numeric(ranked[pred_col], errors="coerce")
        ranked["abs_error_kwh"] = (
            ranked["pred_kwh"] - pd.to_numeric(ranked["usage_kwh_clean"], errors="coerce")
        ).abs()
        ranked["abs_pct_error"] = np.where(
            ranked["usage_kwh_clean"] > 0,
            ranked["abs_error_kwh"] / ranked["usage_kwh_clean"],
            np.nan,
        )

        ranking_columns = [
            "prediction_method",
            "date",
            "report_month",
            "building_name_recent",
            "usage_kwh_clean",
            "pred_kwh",
            "abs_error_kwh",
            "abs_pct_error",
            "validation_period",
        ]
        existing_columns = [column for column in ranking_columns if column in ranked.columns]
        ranking_frames.append(
            ranked.sort_values("abs_error_kwh", ascending=False)[existing_columns].head(100),
        )

    if not ranking_frames:
        return pd.DataFrame()

    return pd.concat(ranking_frames, ignore_index=True)


def save_outputs(
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    metrics_by_building: pd.DataFrame,
    metrics_by_month: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_output = panel.drop(columns=OBSERVED_SCHOOL_FLOW_COLUMNS, errors="ignore")
    prediction_output.to_csv(
        output_dir / "baseline_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_building.to_csv(
        output_dir / "baseline_metrics_by_building.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_month.to_csv(
        output_dir / "baseline_metrics_by_month.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_error_rankings(validation).to_csv(
        output_dir / "baseline_error_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_baseline(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> BaselineRunResult:
    panel = load_panel(data_path)
    panel = add_baseline_predictions(panel)
    validation = build_validation_frame(panel)

    metric_frames = [
        calculate_metrics(validation, pred_col, ["building_name_recent"])
        for pred_col in PREDICTION_COLUMNS
    ]
    metrics_by_building = pd.concat(metric_frames, ignore_index=True)

    metric_frames = [
        calculate_metrics(validation, pred_col, ["report_month"])
        for pred_col in PREDICTION_COLUMNS
    ]
    metrics_by_month = pd.concat(metric_frames, ignore_index=True)

    save_outputs(panel, validation, metrics_by_building, metrics_by_month, output_dir)

    validation_start = None
    validation_end = None
    if not validation.empty:
        validation_start = validation["date"].min().strftime("%Y-%m-%d")
        validation_end = validation["date"].max().strftime("%Y-%m-%d")

    return BaselineRunResult(
        data_path=data_path,
        output_dir=output_dir,
        total_rows=int(len(panel)),
        validation_rows=int(len(validation)),
        building_count=int(panel["building_name_recent"].nunique()),
        validation_start=validation_start,
        validation_end=validation_end,
    )


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CEMS electricity-usage baseline validation model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the baseline-ready panel CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated baseline CSV outputs.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_baseline(args.data, args.output_dir)
    print(f"data_path={result.data_path}")
    print(f"output_dir={result.output_dir}")
    print(f"total_rows={result.total_rows}")
    print(f"validation_rows={result.validation_rows}")
    print(f"building_count={result.building_count}")
    print(f"validation_range={result.validation_start}..{result.validation_end}")


if __name__ == "__main__":
    main()
