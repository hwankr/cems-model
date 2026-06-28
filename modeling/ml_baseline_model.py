from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from modeling.baseline_model import add_baseline_predictions, load_panel


DEFAULT_DATA_PATH = Path("10_baseline_ready_panel_primary_reliable.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")

TARGET_COLUMN = "usage_kwh_clean"
LIGHTGBM_Q50_COLUMN = "pred_lightgbm_q50_kwh"
LIGHTGBM_Q_COLUMNS = {
    0.05: "pred_lightgbm_q05_kwh",
    0.10: "pred_lightgbm_q10_kwh",
    0.50: LIGHTGBM_Q50_COLUMN,
    0.90: "pred_lightgbm_q90_kwh",
    0.95: "pred_lightgbm_q95_kwh",
}

COMPARISON_COLUMNS = {
    "naive_seasonal_month_profile": "pred_naive_seasonal_kwh",
    "naive_last_week_same_weekday": "pred_naive_last_week_same_weekday_kwh",
    "current_realtime_uniform_daily": "pred_realtime_uniform_daily_kwh",
    "current_weekday_recent_heuristic": "pred_weekday_recent_kwh",
    "lightgbm_quantile_q50_walk_forward": LIGHTGBM_Q50_COLUMN,
}

NUMERIC_FEATURE_COLUMNS = [
    "month",
    "day",
    "ml_day_of_week",
    "ml_is_weekend",
    "ml_day_of_year",
    "profile_monthly_kwh_mean",
    "profile_monthly_kwh_median",
    "profile_monthly_kwh_std",
    "profile_monthly_kwh_min",
    "profile_monthly_kwh_max",
    "profile_source_month_count",
    "overall_avg_monthly_kwh",
    "month_factor_vs_overall",
    "calendar_days_in_month",
    "pred_realtime_uniform_daily_kwh",
    "pred_weekday_recent_kwh",
    "ml_lag_1d_kwh",
    "ml_lag_3d_mean_kwh",
    "ml_lag_7d_mean_kwh",
    "ml_lag_7d_same_weekday_kwh",
]

RESIDUAL_CORRELATION_FEATURES = [
    "month",
    "day",
    "ml_day_of_week",
    "ml_is_weekend",
    "ml_day_of_year",
    "pred_realtime_uniform_daily_kwh",
    "pred_weekday_recent_kwh",
    "profile_monthly_kwh_mean",
    "ml_lag_1d_kwh",
    "ml_lag_3d_mean_kwh",
    "ml_lag_7d_mean_kwh",
    "ml_lag_7d_same_weekday_kwh",
]


@dataclass(frozen=True)
class MLPredictionSpec:
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95)
    min_train_days: int = 14
    random_state: int = 20260628


@dataclass(frozen=True)
class MLBaselineRunResult:
    data_path: Path
    output_dir: Path
    evaluation_rows: int
    evaluation_start: str | None
    evaluation_end: str | None
    building_count: int


ModelFactory = Callable[[float], object]


def default_model_factory(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=20260628,
        verbosity=-1,
    )


def add_ml_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = add_baseline_predictions(panel)
    result["date"] = pd.to_datetime(result["date"])
    result["ml_day_of_week"] = result["date"].dt.dayofweek
    result["ml_is_weekend"] = result["ml_day_of_week"].isin([5, 6]).astype(int)
    result["ml_day_of_year"] = result["date"].dt.dayofyear

    result = result.sort_values(["building_name_recent", "date"]).copy()
    grouped_usage = result.groupby("building_name_recent", sort=False)[TARGET_COLUMN]
    result["ml_lag_1d_kwh"] = grouped_usage.shift(1)
    result["ml_lag_3d_mean_kwh"] = grouped_usage.transform(
        lambda values: values.shift(1).rolling(3, min_periods=3).mean(),
    )
    result["ml_lag_7d_mean_kwh"] = grouped_usage.transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).mean(),
    )
    result["ml_lag_7d_same_weekday_kwh"] = grouped_usage.shift(7)

    fallback = pd.to_numeric(result["pred_realtime_uniform_daily_kwh"], errors="coerce")
    for column in [
        "ml_lag_1d_kwh",
        "ml_lag_3d_mean_kwh",
        "ml_lag_7d_mean_kwh",
        "ml_lag_7d_same_weekday_kwh",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(fallback)

    result["pred_naive_seasonal_kwh"] = fallback
    result["pred_naive_last_week_same_weekday_kwh"] = result[
        "ml_lag_7d_same_weekday_kwh"
    ].fillna(fallback)

    return result.sort_values(["date", "building_name_recent"]).reset_index(drop=True)


def build_validation_frame(panel: pd.DataFrame) -> pd.DataFrame:
    validation = panel[
        panel["is_validation_target_clean"].map(lambda value: str(value).lower() == "true")
    ].copy()
    validation = validation.dropna(subset=[TARGET_COLUMN])
    return validation.sort_values(["date", "building_name_recent"]).reset_index(drop=True)


def build_feature_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = frame[NUMERIC_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    building_dummies = pd.get_dummies(
        frame["building_name_recent"],
        prefix="building",
        dtype=float,
    )
    features = pd.concat(
        [numeric.reset_index(drop=True), building_dummies.reset_index(drop=True)],
        axis=1,
    )
    if columns is not None:
        features = features.reindex(columns=columns, fill_value=0)
    return features


def build_walk_forward_predictions(
    panel: pd.DataFrame,
    spec: MLPredictionSpec = MLPredictionSpec(),
    model_factory: ModelFactory = default_model_factory,
) -> pd.DataFrame:
    validation = build_validation_frame(panel)
    dates = sorted(validation["date"].unique())
    rows: list[pd.DataFrame] = []

    for test_date in dates[spec.min_train_days :]:
        train = validation[validation["date"] < test_date].copy()
        test = validation[validation["date"] == test_date].copy()
        if train.empty or test.empty:
            continue

        x_train = build_feature_frame(train)
        x_test = build_feature_frame(test, list(x_train.columns))
        y_train = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")

        fold = test.copy()
        for alpha in spec.quantiles:
            if alpha not in LIGHTGBM_Q_COLUMNS:
                continue
            model = model_factory(alpha)
            model.fit(x_train, y_train)
            fold[LIGHTGBM_Q_COLUMNS[alpha]] = np.maximum(model.predict(x_test), 0)

        fold["walk_forward_train_start"] = train["date"].min()
        fold["walk_forward_train_end"] = train["date"].max()
        fold["walk_forward_train_rows"] = len(train)
        rows.append(fold)

    if not rows:
        raise RuntimeError("No walk-forward predictions were produced.")

    predictions = pd.concat(rows, ignore_index=True)
    return predictions.sort_values(["date", "building_name_recent"]).reset_index(drop=True)


def calculate_comparison_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, prediction_column in COMPARISON_COLUMNS.items():
        if prediction_column not in predictions.columns:
            continue
        rows.append(calculate_metric_row(predictions, model_name, prediction_column))
    return pd.DataFrame(rows)


def calculate_metric_row(
    frame: pd.DataFrame,
    model_name: str,
    prediction_column: str,
) -> dict[str, float | int | str]:
    scored = frame.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    y = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
    pred = pd.to_numeric(scored[prediction_column], errors="coerce")
    residual = pred - y
    abs_error = residual.abs()
    squared_error = residual**2
    actual_sum = y.sum()

    return {
        "model": model_name,
        "prediction_method": prediction_column,
        "prediction_column": prediction_column,
        "n_rows": int(len(scored)),
        "actual_sum_kwh": float(actual_sum),
        "pred_sum_kwh": float(pred.sum()),
        "mean_actual_kwh": float(y.mean()),
        "mae_kwh": float(abs_error.mean()),
        "rmse_kwh": float(np.sqrt(squared_error.mean())),
        "wape": float(abs_error.sum() / actual_sum) if actual_sum else np.nan,
        "nmae": float(abs_error.mean() / y.mean()) if y.mean() else np.nan,
        "bias_kwh_mean_pred_minus_actual": float(residual.mean()),
        "bias_pct_sum_pred_minus_actual": float(residual.sum() / actual_sum)
        if actual_sum
        else np.nan,
    }


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


def pinball_loss(y: pd.Series, pred: pd.Series, quantile: float) -> float:
    diff = y - pred
    return float(np.maximum(quantile * diff, (quantile - 1) * diff).mean())


def calculate_pinball_loss(predictions: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(predictions[TARGET_COLUMN], errors="coerce")
    rows = []
    for quantile, column in LIGHTGBM_Q_COLUMNS.items():
        if column not in predictions.columns:
            continue
        rows.append(
            {
                "quantile": quantile,
                "prediction_column": column,
                "pinball_loss_kwh": pinball_loss(y, predictions[column], quantile),
            }
        )
    return pd.DataFrame(rows)


def calculate_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(predictions[TARGET_COLUMN], errors="coerce")
    intervals = [
        ("90", 0.90, "pred_lightgbm_q05_kwh", "pred_lightgbm_q95_kwh"),
        ("80", 0.80, "pred_lightgbm_q10_kwh", "pred_lightgbm_q90_kwh"),
    ]
    rows = []
    for label, nominal, lower_col, upper_col in intervals:
        if lower_col not in predictions.columns or upper_col not in predictions.columns:
            continue
        lower = predictions[lower_col]
        upper = predictions[upper_col]
        covered = (y >= lower) & (y <= upper)
        rows.append(
            {
                "interval": label,
                "nominal_coverage": nominal,
                "actual_coverage": float(covered.mean()),
                "coverage_gap_actual_minus_nominal": float(covered.mean() - nominal),
                "mean_interval_width_kwh": float((upper - lower).mean()),
                "median_interval_width_kwh": float((upper - lower).median()),
            }
        )
    return pd.DataFrame(rows)


def calculate_residual_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    diagnostics = predictions.dropna(subset=[TARGET_COLUMN, LIGHTGBM_Q50_COLUMN]).copy()
    diagnostics["residual_pred_minus_actual"] = (
        diagnostics[LIGHTGBM_Q50_COLUMN] - diagnostics[TARGET_COLUMN]
    )
    rows = []
    for feature in RESIDUAL_CORRELATION_FEATURES:
        x = pd.to_numeric(diagnostics[feature], errors="coerce")
        y = pd.to_numeric(diagnostics["residual_pred_minus_actual"], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3 or x[valid].nunique() < 2:
            continue
        pearson = stats.pearsonr(x[valid], y[valid])
        spearman = stats.spearmanr(x[valid], y[valid])
        rows.append(
            {
                "feature": feature,
                "n_rows": int(valid.sum()),
                "pearson_r": float(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
            }
        )
    return pd.DataFrame(rows)


def calculate_lightgbm_group_bias(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for building, group in predictions.groupby("building_name_recent", sort=True):
        row = calculate_metric_row(
            group,
            str(building),
            LIGHTGBM_Q50_COLUMN,
        )
        row["building_name_recent"] = building
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bias_pct_sum_pred_minus_actual")


def save_outputs(
    predictions: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_dir / "ml_baseline_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_comparison_metrics(predictions).to_csv(
        output_dir / "ml_baseline_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_group_metrics(predictions, ["report_month"]).to_csv(
        output_dir / "ml_baseline_metrics_by_month.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_group_metrics(predictions, ["building_name_recent"]).to_csv(
        output_dir / "ml_baseline_metrics_by_building.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_coverage(predictions).to_csv(
        output_dir / "ml_baseline_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_pinball_loss(predictions).to_csv(
        output_dir / "ml_baseline_pinball_loss.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_residual_correlations(predictions).to_csv(
        output_dir / "ml_baseline_residual_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_lightgbm_group_bias(predictions).to_csv(
        output_dir / "ml_baseline_group_bias_by_building.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_ml_baseline(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    spec: MLPredictionSpec = MLPredictionSpec(),
    model_factory: ModelFactory = default_model_factory,
) -> MLBaselineRunResult:
    panel = load_panel(data_path)
    panel = add_ml_features(panel)
    predictions = build_walk_forward_predictions(panel, spec, model_factory)
    save_outputs(predictions, output_dir)

    evaluation_start = None
    evaluation_end = None
    if not predictions.empty:
        evaluation_start = predictions["date"].min().strftime("%Y-%m-%d")
        evaluation_end = predictions["date"].max().strftime("%Y-%m-%d")

    return MLBaselineRunResult(
        data_path=data_path,
        output_dir=output_dir,
        evaluation_rows=int(len(predictions)),
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        building_count=int(predictions["building_name_recent"].nunique()),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CEMS LightGBM quantile baseline benchmark.",
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
        help="Directory for generated ML baseline CSV outputs.",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=14,
        help="Number of past validation days required before first prediction.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_ml_baseline(
        data_path=args.data,
        output_dir=args.output_dir,
        spec=MLPredictionSpec(min_train_days=args.min_train_days),
    )
    print(f"data_path={result.data_path}")
    print(f"output_dir={result.output_dir}")
    print(f"evaluation_rows={result.evaluation_rows}")
    print(f"building_count={result.building_count}")
    print(f"evaluation_range={result.evaluation_start}..{result.evaluation_end}")


if __name__ == "__main__":
    main()
