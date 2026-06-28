from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.baseline_model import add_baseline_predictions, load_panel


DATA_PATH = ROOT / "10_baseline_ready_panel_primary_reliable.csv"
OUTPUT_DIR = ROOT / "outputs" / "fairness_diagnostics"
MIN_TRAIN_DAYS = 14
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]

OBSERVED_SCHOOL_FLOW_COLUMNS = {
    "school_total_daily_kwh",
    "school_total_is_usable",
    "school_total_observed_month_sum_kwh",
    "school_total_observed_day_count",
    "school_total_month_observed_completeness",
    "school_total_day_weight_observed_month",
    "baseline_school_shape_kwh_observed_norm",
}

NUMERIC_FEATURES = [
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "day_of_year",
    "profile_monthly_kwh_mean",
    "profile_monthly_kwh_median",
    "profile_monthly_kwh_std",
    "profile_monthly_kwh_min",
    "profile_monthly_kwh_max",
    "profile_source_month_count",
    "overall_avg_monthly_kwh",
    "month_factor_vs_overall",
    "calendar_days_in_month",
    "baseline_uniform_daily_kwh",
    "lag_1d_kwh",
    "lag_3d_mean_kwh",
    "lag_7d_mean_kwh",
    "lag_7d_same_weekday_kwh",
]


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(DATA_PATH)
    panel = add_baseline_predictions(panel)
    panel = add_features(panel)
    validation = build_validation(panel)
    predictions = run_walk_forward_quantiles(validation)

    save_split_summary(panel, validation, predictions)
    save_model_comparison(predictions)
    save_pinball_loss(predictions)
    save_coverage(predictions)
    save_residual_diagnostics(predictions)
    save_shap_summary(validation, predictions)
    save_plots(predictions)

    print(f"wrote={OUTPUT_DIR}")
    print(f"eval_rows={len(predictions)}")
    print(
        "eval_range="
        f"{predictions['date'].min().date()}..{predictions['date'].max().date()}"
    )


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["day_of_week"] = result["date"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["day_of_year"] = result["date"].dt.dayofyear

    result = result.sort_values(["building_name_recent", "date"]).copy()
    grouped = result.groupby("building_name_recent", sort=False)["usage_kwh_clean"]
    result["lag_1d_kwh"] = grouped.shift(1)
    result["lag_3d_mean_kwh"] = grouped.transform(
        lambda values: values.shift(1).rolling(3, min_periods=3).mean()
    )
    result["lag_7d_mean_kwh"] = grouped.transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).mean()
    )
    result["lag_7d_same_weekday_kwh"] = grouped.shift(7)

    fallback = pd.to_numeric(result["baseline_uniform_daily_kwh"], errors="coerce")
    for column in [
        "lag_1d_kwh",
        "lag_3d_mean_kwh",
        "lag_7d_mean_kwh",
        "lag_7d_same_weekday_kwh",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(fallback)

    result["pred_naive_seasonal_kwh"] = fallback
    result["pred_naive_last_week_same_weekday_kwh"] = result[
        "lag_7d_same_weekday_kwh"
    ].fillna(fallback)

    return result.sort_values(["date", "building_name_recent"]).reset_index(drop=True)


def build_validation(panel: pd.DataFrame) -> pd.DataFrame:
    validation = panel[
        panel["is_validation_target_clean"].map(lambda value: str(value).lower() == "true")
    ].copy()
    validation = validation.dropna(subset=["usage_kwh_clean"])
    return validation.sort_values(["date", "building_name_recent"]).reset_index(drop=True)


def feature_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    building_dummies = pd.get_dummies(
        frame["building_name_recent"],
        prefix="building",
        dtype=float,
    )
    features = pd.concat([numeric.reset_index(drop=True), building_dummies.reset_index(drop=True)], axis=1)
    if columns is not None:
        features = features.reindex(columns=columns, fill_value=0)
    return features


def make_model(alpha: float) -> lgb.LGBMRegressor:
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


def run_walk_forward_quantiles(validation: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(validation["date"].unique())
    start_index = MIN_TRAIN_DAYS
    rows: list[pd.DataFrame] = []

    for test_date in dates[start_index:]:
        train = validation[validation["date"] < test_date].copy()
        test = validation[validation["date"] == test_date].copy()
        if train.empty or test.empty:
            continue

        x_train = feature_frame(train)
        x_test = feature_frame(test, list(x_train.columns))
        y_train = pd.to_numeric(train["usage_kwh_clean"], errors="coerce")

        fold = test.copy()
        for alpha in QUANTILES:
            model = make_model(alpha)
            model.fit(x_train, y_train)
            fold[f"pred_lgbm_q{int(alpha * 100):02d}_kwh"] = model.predict(x_test)

        fold["walk_forward_train_start"] = train["date"].min()
        fold["walk_forward_train_end"] = train["date"].max()
        fold["walk_forward_train_rows"] = len(train)
        rows.append(fold)

    if not rows:
        raise RuntimeError("No walk-forward folds were produced.")

    predictions = pd.concat(rows, ignore_index=True)
    quantile_columns = [f"pred_lgbm_q{int(alpha * 100):02d}_kwh" for alpha in QUANTILES]
    predictions[quantile_columns] = predictions[quantile_columns].clip(lower=0)
    predictions["quantile_crossing_raw"] = (
        (predictions["pred_lgbm_q05_kwh"] > predictions["pred_lgbm_q10_kwh"])
        | (predictions["pred_lgbm_q10_kwh"] > predictions["pred_lgbm_q50_kwh"])
        | (predictions["pred_lgbm_q50_kwh"] > predictions["pred_lgbm_q90_kwh"])
        | (predictions["pred_lgbm_q90_kwh"] > predictions["pred_lgbm_q95_kwh"])
    )

    sorted_quantiles = np.sort(predictions[quantile_columns].to_numpy(), axis=1)
    for idx, column in enumerate(quantile_columns):
        predictions[f"{column}_monotone"] = sorted_quantiles[:, idx]

    predictions.to_csv(
        OUTPUT_DIR / "walk_forward_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return predictions


def save_split_summary(
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    summary = {
        "current_repository_model": {
            "type": "deterministic heuristic baseline, not LightGBM",
            "train_test_split": "no random split; current script scores rows where is_validation_target_clean is true",
            "validation_start": str(validation["date"].min().date()),
            "validation_end": str(validation["date"].max().date()),
            "validation_rows": int(len(validation)),
            "building_count": int(validation["building_name_recent"].nunique()),
        },
        "diagnostic_lightgbm_quantile": {
            "split": "expanding walk-forward by date",
            "first_prediction_date": str(predictions["date"].min().date()),
            "last_prediction_date": str(predictions["date"].max().date()),
            "min_train_days_before_first_prediction": MIN_TRAIN_DAYS,
            "eval_rows": int(len(predictions)),
            "train_rows_first_fold": int(predictions["walk_forward_train_rows"].min()),
            "train_rows_last_fold": int(predictions["walk_forward_train_rows"].max()),
            "features_excluded_as_leaky": sorted(OBSERVED_SCHOOL_FLOW_COLUMNS),
            "weather_features_available": False,
            "hourly_time_of_day_available": False,
        },
        "panel": {
            "rows": int(len(panel)),
            "date_start": str(panel["date"].min().date()),
            "date_end": str(panel["date"].max().date()),
            "building_count": int(panel["building_name_recent"].nunique()),
        },
    }
    (OUTPUT_DIR / "validation_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def metric_row(frame: pd.DataFrame, model_name: str, pred_col: str) -> dict[str, float | str | int]:
    scored = frame.dropna(subset=["usage_kwh_clean", pred_col]).copy()
    y = pd.to_numeric(scored["usage_kwh_clean"], errors="coerce")
    pred = pd.to_numeric(scored[pred_col], errors="coerce")
    residual = pred - y
    abs_error = residual.abs()
    return {
        "model": model_name,
        "prediction_column": pred_col,
        "n_rows": int(len(scored)),
        "actual_sum_kwh": float(y.sum()),
        "pred_sum_kwh": float(pred.sum()),
        "mean_actual_kwh": float(y.mean()),
        "mae_kwh": float(mean_absolute_error(y, pred)),
        "rmse_kwh": float(mean_squared_error(y, pred) ** 0.5),
        "wape": float(abs_error.sum() / y.sum()),
        "nmae": float(abs_error.mean() / y.mean()),
        "bias_kwh_mean_pred_minus_actual": float(residual.mean()),
        "bias_pct_sum_pred_minus_actual": float(residual.sum() / y.sum()),
    }


def save_model_comparison(predictions: pd.DataFrame) -> None:
    rows = [
        metric_row(predictions, "naive_seasonal_month_profile", "pred_naive_seasonal_kwh"),
        metric_row(
            predictions,
            "naive_last_week_same_weekday",
            "pred_naive_last_week_same_weekday_kwh",
        ),
        metric_row(predictions, "current_weekday_recent_heuristic", "pred_weekday_recent_kwh"),
        metric_row(predictions, "lightgbm_quantile_q50_walk_forward", "pred_lgbm_q50_kwh"),
    ]
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )


def pinball_loss(y: pd.Series, pred: pd.Series, quantile: float) -> float:
    diff = y - pred
    return float(np.maximum(quantile * diff, (quantile - 1) * diff).mean())


def save_pinball_loss(predictions: pd.DataFrame) -> None:
    y = pd.to_numeric(predictions["usage_kwh_clean"], errors="coerce")
    rows = []
    for quantile in QUANTILES:
        suffix = int(quantile * 100)
        for monotone in [False, True]:
            column = f"pred_lgbm_q{suffix:02d}_kwh"
            if monotone:
                column = f"{column}_monotone"
            rows.append(
                {
                    "quantile": quantile,
                    "prediction_column": column,
                    "monotone_repaired": monotone,
                    "pinball_loss_kwh": pinball_loss(y, predictions[column], quantile),
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "pinball_loss.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_coverage(predictions: pd.DataFrame) -> None:
    y = pd.to_numeric(predictions["usage_kwh_clean"], errors="coerce")
    rows = []
    intervals = [
        ("raw_90", 0.90, "pred_lgbm_q05_kwh", "pred_lgbm_q95_kwh"),
        ("raw_80", 0.80, "pred_lgbm_q10_kwh", "pred_lgbm_q90_kwh"),
        (
            "monotone_90",
            0.90,
            "pred_lgbm_q05_kwh_monotone",
            "pred_lgbm_q95_kwh_monotone",
        ),
        (
            "monotone_80",
            0.80,
            "pred_lgbm_q10_kwh_monotone",
            "pred_lgbm_q90_kwh_monotone",
        ),
    ]
    for label, nominal, lower_col, upper_col in intervals:
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

    rows.append(
        {
            "interval": "raw_quantile_crossing_rate",
            "nominal_coverage": np.nan,
            "actual_coverage": float(predictions["quantile_crossing_raw"].mean()),
            "coverage_gap_actual_minus_nominal": np.nan,
            "mean_interval_width_kwh": np.nan,
            "median_interval_width_kwh": np.nan,
        }
    )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_residual_diagnostics(predictions: pd.DataFrame) -> None:
    diagnostics = predictions.copy()
    diagnostics["residual_pred_minus_actual"] = (
        diagnostics["pred_lgbm_q50_kwh"] - diagnostics["usage_kwh_clean"]
    )

    correlation_features = [
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "baseline_uniform_daily_kwh",
        "profile_monthly_kwh_mean",
        "lag_1d_kwh",
        "lag_3d_mean_kwh",
        "lag_7d_mean_kwh",
        "lag_7d_same_weekday_kwh",
    ]
    rows = []
    for feature in correlation_features:
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
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "residual_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_rows = []
    for building, group in diagnostics.groupby("building_name_recent"):
        group_rows.append(metric_row(group, str(building), "pred_lgbm_q50_kwh"))
    group_bias = pd.DataFrame(group_rows).rename(columns={"model": "building_name_recent"})
    group_bias = group_bias.sort_values("bias_pct_sum_pred_minus_actual")
    group_bias.to_csv(
        OUTPUT_DIR / "group_bias_by_building.csv",
        index=False,
        encoding="utf-8-sig",
    )

    day_rows = []
    for day, group in diagnostics.groupby("day_of_week"):
        row = metric_row(group, str(day), "pred_lgbm_q50_kwh")
        row["day_of_week"] = int(day)
        day_rows.append(row)
    pd.DataFrame(day_rows).sort_values("day_of_week").to_csv(
        OUTPUT_DIR / "residual_by_day_of_week.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_shap_summary(validation: pd.DataFrame, predictions: pd.DataFrame) -> None:
    final_train = validation[validation["date"] < predictions["date"].max()].copy()
    x_train = feature_frame(final_train)
    y_train = pd.to_numeric(final_train["usage_kwh_clean"], errors="coerce")
    model = make_model(0.50)
    model.fit(x_train, y_train)

    sample = predictions.sample(min(700, len(predictions)), random_state=20260628)
    x_sample = feature_frame(sample, list(x_train.columns))
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(x_sample))

    rows = []
    for idx, feature in enumerate(x_sample.columns):
        values = pd.to_numeric(x_sample[feature], errors="coerce")
        contributions = pd.Series(shap_values[:, idx])
        if values.nunique(dropna=True) > 1:
            corr = stats.spearmanr(values, contributions).statistic
        else:
            corr = np.nan
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap_kwh": float(np.abs(contributions).mean()),
                "mean_shap_kwh": float(contributions.mean()),
                "spearman_feature_value_vs_shap": float(corr)
                if not pd.isna(corr)
                else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_abs_shap_kwh", ascending=False)
    summary.to_csv(
        OUTPUT_DIR / "shap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shap_payload = {
        "weekend_mean_shap_kwh": mean_shap_for_mask(
            x_sample,
            shap_values,
            "is_weekend",
            x_sample["is_weekend"] == 1,
        ),
        "weekday_mean_shap_kwh": mean_shap_for_mask(
            x_sample,
            shap_values,
            "is_weekend",
            x_sample["is_weekend"] == 0,
        ),
        "lag_1d_spearman_feature_value_vs_shap": float(
            summary.loc[
                summary["feature"] == "lag_1d_kwh",
                "spearman_feature_value_vs_shap",
            ].iloc[0]
        ),
        "baseline_uniform_spearman_feature_value_vs_shap": float(
            summary.loc[
                summary["feature"] == "baseline_uniform_daily_kwh",
                "spearman_feature_value_vs_shap",
            ].iloc[0]
        ),
    }
    (OUTPUT_DIR / "shap_direction_checks.json").write_text(
        json.dumps(shap_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mean_shap_for_mask(
    x_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature: str,
    mask: pd.Series,
) -> float:
    feature_index = list(x_sample.columns).index(feature)
    return float(np.asarray(shap_values)[mask.to_numpy(), feature_index].mean())


def save_plots(predictions: pd.DataFrame) -> None:
    plot_frame = predictions.copy()
    plot_frame["residual_pred_minus_actual"] = (
        plot_frame["pred_lgbm_q50_kwh"] - plot_frame["usage_kwh_clean"]
    )

    scatter_specs = [
        ("day_of_week", "residual_vs_day_of_week.png"),
        ("day_of_year", "residual_vs_day_of_year.png"),
        ("baseline_uniform_daily_kwh", "residual_vs_seasonal_baseline.png"),
        ("lag_1d_kwh", "residual_vs_lag_1d.png"),
    ]
    for x_col, file_name in scatter_specs:
        plt.figure(figsize=(8, 4.8))
        plt.scatter(
            plot_frame[x_col],
            plot_frame["residual_pred_minus_actual"],
            s=12,
            alpha=0.45,
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel(x_col)
        plt.ylabel("residual pred-actual kWh")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / file_name, dpi=150)
        plt.close()

    building_bias = pd.read_csv(OUTPUT_DIR / "group_bias_by_building.csv")
    building_bias = building_bias.sort_values("bias_pct_sum_pred_minus_actual")
    plt.figure(figsize=(9, 5.5))
    plt.barh(
        building_bias["building_name_recent"],
        building_bias["bias_pct_sum_pred_minus_actual"],
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("sum bias pred-actual / actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "building_bias_pct.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
