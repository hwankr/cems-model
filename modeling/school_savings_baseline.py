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

    # Naive WAPE: frozen_profile_kwh vs actual on the held-out calib set
    naive_wape: float = float("nan")
    if "frozen_profile_kwh" in calib.columns:
        naive_pred = pd.to_numeric(calib["frozen_profile_kwh"], errors="coerce").to_numpy()
        naive_abs = np.abs(naive_pred - y_calib)
        if actual_sum > 0:
            naive_wape = float(naive_abs.sum() / float(y_calib.sum()))

    accuracy["naive_wape"] = round(naive_wape, 6) if np.isfinite(naive_wape) else None

    return FrozenBandResult(
        predictions=predictions,
        p50_model=p50_model,
        x_reporting=x_reporting.reset_index(drop=True),
        feature_columns=list(x_proper.columns),
        calibration=calibration,
        accuracy=accuracy,
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

GLOSSARY: list[dict] = [
    {
        "term": "동결 베이스라인",
        "desc": "기준 기간(대회 전) 데이터만으로 만들어 고정(동결)한 예상 사용량. 리포팅 기간이 시작되면 업데이트하지 않으므로, 실제 절감이 있어도 베이스라인이 따라 내려가지 않습니다.",
    },
    {
        "term": "baseline chasing",
        "desc": "절감 후 실제 사용량을 학습해 베이스라인이 낮아지는 현상. 동결 베이스라인은 이를 원천 차단합니다.",
    },
    {
        "term": "절감률",
        "desc": "avoided_kwh / baseline_kwh × 100(%). 양수 = 절감, 음수 = 초과 사용. 면적·계절 보정 없이 단순 비율이므로 비교 시 기간·날씨 조건 동일 여부를 확인하세요.",
    },
    {
        "term": "면적당(kWh/㎡)",
        "desc": "총 절감량(kWh) ÷ 건물 연면적(㎡). 규모가 다른 학교·기간을 공정하게 비교할 수 있는 단위.",
    },
    {
        "term": "M&V",
        "desc": "Measurement & Verification(측정·검증). 에너지 절감을 계약상 증명하기 위한 방법론 총칭. 이 모델은 M&V 원칙에 따라 베이스라인을 동결합니다.",
    },
    {
        "term": "P10/P50/P90",
        "desc": "이 시간에 '정상이라면' 사용량이 P10 이상일 확률 90%, P50(중앙값)이 가장 그럴듯한 값, P90 이하일 확률 90%. [P10, P90] 안에 정상의 약 80%가 들어옵니다.",
    },
    {
        "term": "WAPE",
        "desc": "가중 절대 백분율 오차 = Σ|예측−실제| / Σ실제. 낮을수록 정확. 본 모델은 held-out 28일 데이터로 계산합니다.",
    },
    {
        "term": "적중률(coverage)",
        "desc": "실제값이 [P10, P90] 밴드 안에 든 비율. 잘 보정(calibration)된 모델은 ≈80%.",
    },
    {
        "term": "확실한 절감(actual<P10)",
        "desc": "실제 사용량이 P10(하한)보다 낮은 시간. 통계적으로 90% 이상 확률로 절감이 일어난 것으로 볼 수 있습니다.",
    },
]


# ---------------------------------------------------------------------------
# Task 2: Savings computation
# ---------------------------------------------------------------------------

def compute_savings(predictions: pd.DataFrame, total_area_sqm: float) -> pd.DataFrame:
    """Add savings columns to predictions DataFrame.

    Columns added:
    - actual_kwh: usage_kwh cast to numeric
    - avoided_kwh: p50_kwh - actual_kwh (POSITIVE = saved energy)
    - avoided_pct: avoided_kwh / p50_kwh (guarded against /0)
    - avoided_kwh_per_sqm: avoided_kwh / total_area_sqm
    - is_confirmed_saving: actual_kwh < p10_kwh
    - is_overuse: actual_kwh > p90_kwh
    """
    result = predictions.copy()
    actual = pd.to_numeric(result["usage_kwh"], errors="coerce")
    result["actual_kwh"] = actual
    p50 = pd.to_numeric(result["p50_kwh"], errors="coerce")
    p10 = pd.to_numeric(result["p10_kwh"], errors="coerce")
    p90 = pd.to_numeric(result["p90_kwh"], errors="coerce")

    avoided = p50 - actual
    result["avoided_kwh"] = avoided
    # Guard /0 on p50
    result["avoided_pct"] = avoided.where(p50.abs() > 1e-9, other=float("nan")) / p50.where(p50.abs() > 1e-9, other=float("nan"))
    result["avoided_kwh_per_sqm"] = avoided / total_area_sqm
    result["is_confirmed_saving"] = actual < p10
    result["is_overuse"] = actual > p90
    return result


# ---------------------------------------------------------------------------
# Task 2: Scorecard
# ---------------------------------------------------------------------------

def build_scorecard(savings: pd.DataFrame, accuracy: dict, total_area_sqm: float) -> dict:
    """Build summary scorecard dict."""
    baseline_sum = float(savings["p50_kwh"].sum())
    actual_sum = float(savings["actual_kwh"].sum())
    avoided_sum = float(savings["avoided_kwh"].sum())
    avoided_pct = avoided_sum / baseline_sum if abs(baseline_sum) > 1e-9 else float("nan")
    avoided_per_sqm = avoided_sum / total_area_sqm
    confirmed_saving_hours = int(savings["is_confirmed_saving"].sum())
    overuse_hours = int(savings["is_overuse"].sum())
    n_rows = int(len(savings))

    return {
        "baseline_sum_kwh": round(baseline_sum, 2),
        "actual_sum_kwh": round(actual_sum, 2),
        "avoided_sum_kwh": round(avoided_sum, 2),
        "avoided_pct": round(avoided_pct, 6) if np.isfinite(avoided_pct) else None,
        "avoided_per_sqm_kwh": round(avoided_per_sqm, 4),
        "confirmed_saving_hours": confirmed_saving_hours,
        "overuse_hours": overuse_hours,
        "n_rows": n_rows,
        "heldout_wape": accuracy.get("wape"),
        "heldout_coverage": accuracy.get("coverage"),
        "area_sqm": round(total_area_sqm, 2),
        "avoided_pct_display": round(avoided_pct * 100, 2) if (avoided_pct is not None and np.isfinite(avoided_pct)) else None,
    }


# ---------------------------------------------------------------------------
# Task 2: Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(
    savings: pd.DataFrame,
    total_area_sqm: float,
    round_days: int = 7,
) -> pd.DataFrame:
    """Split reporting dates into consecutive round_days-day rounds and rank by avoided_pct desc.

    Label format: "N주차 (MM-DD~MM-DD)"
    """
    df = savings.copy()
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date

    all_dates = sorted(df["report_date"].unique())
    if not all_dates:
        return pd.DataFrame(columns=["round", "days", "baseline_kwh", "actual_kwh",
                                      "avoided_kwh", "avoided_pct", "avoided_per_sqm_kwh", "rank"])

    start_date = all_dates[0]
    import datetime
    rounds_list = []
    round_num = 1

    cursor = start_date
    while cursor <= all_dates[-1]:
        round_end = cursor + datetime.timedelta(days=round_days - 1)
        mask = (df["report_date"] >= cursor) & (df["report_date"] <= round_end)
        chunk = df[mask]
        if chunk.empty:
            cursor = round_end + datetime.timedelta(days=1)
            round_num += 1
            continue
        days_in_round = len(chunk["report_date"].unique())
        label = f"{round_num}주차 ({cursor.strftime('%m-%d')}~{min(round_end, all_dates[-1]).strftime('%m-%d')})"
        baseline_kwh = float(chunk["p50_kwh"].sum())
        actual_kwh = float(chunk["actual_kwh"].sum())
        avoided_kwh = float(chunk["avoided_kwh"].sum())
        avoided_pct = avoided_kwh / baseline_kwh if abs(baseline_kwh) > 1e-9 else float("nan")
        avoided_per_sqm_kwh = avoided_kwh / total_area_sqm

        rounds_list.append({
            "round": label,
            "days": days_in_round,
            "baseline_kwh": round(baseline_kwh, 2),
            "actual_kwh": round(actual_kwh, 2),
            "avoided_kwh": round(avoided_kwh, 2),
            "avoided_pct": round(avoided_pct, 6) if np.isfinite(avoided_pct) else None,
            "avoided_per_sqm_kwh": round(avoided_per_sqm_kwh, 4),
        })
        cursor = round_end + datetime.timedelta(days=1)
        round_num += 1

    lb = pd.DataFrame(rounds_list)
    # Rank by avoided_pct desc (NaN last)
    lb = lb.sort_values("avoided_pct", ascending=False, na_position="last").reset_index(drop=True)
    lb.insert(0, "rank", range(1, len(lb) + 1))
    return lb


# ---------------------------------------------------------------------------
# Task 2: SavingsRunResult dataclass + run_savings_demo
# ---------------------------------------------------------------------------

@dataclass
class SavingsRunResult:
    """Result of run_savings_demo."""
    output_dir: Path
    reporting_rows: int
    avoided_pct: float
    avoided_sum_kwh: float
    avoided_per_sqm_kwh: float
    heldout_wape: float
    heldout_coverage: float
    used_fallback: bool


DEFAULT_AREA_PATH = Path("inputs/yu_building_area.xlsx")


def run_savings_demo(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = Path("outputs"),
    web_dir: Path = Path("school_savings_web"),
    area_path: Path = DEFAULT_AREA_PATH,
) -> SavingsRunResult:
    """Full savings baseline pipeline.

    1. build_savings_frame -> add_frozen_profile -> train_frozen_band
    2. compute_savings -> build_scorecard -> build_leaderboard -> daily agg
    3. Write outputs/ CSVs + JSON and school_savings_web/data/savings.json
    """
    import json

    # --- area ---
    total_area_sqm = load_total_area_sqm(area_path)

    # --- model ---
    frame = build_savings_frame(data_path)
    frame = add_frozen_profile(frame)
    band_result = train_frozen_band(frame)

    predictions = band_result.predictions
    accuracy = band_result.accuracy
    calibration = band_result.calibration

    # --- savings ---
    savings = compute_savings(predictions, total_area_sqm)

    # --- scorecard ---
    scorecard = build_scorecard(savings, accuracy, total_area_sqm)

    # --- leaderboard ---
    leaderboard = build_leaderboard(savings, total_area_sqm)

    # --- daily aggregation ---
    savings["report_date_str"] = pd.to_datetime(savings["report_date"]).dt.strftime("%Y-%m-%d")
    daily_agg = (
        savings.groupby("report_date_str")
        .agg(
            baseline_kwh=("p50_kwh", "sum"),
            actual_kwh=("actual_kwh", "sum"),
            avoided_kwh=("avoided_kwh", "sum"),
        )
        .reset_index()
        .rename(columns={"report_date_str": "report_date"})
        .sort_values("report_date")
        .reset_index(drop=True)
    )
    daily_agg["avoided_pct"] = (
        daily_agg["avoided_kwh"] / daily_agg["baseline_kwh"].replace(0, float("nan"))
    )
    daily_agg = daily_agg.round({"baseline_kwh": 2, "actual_kwh": 2, "avoided_kwh": 2, "avoided_pct": 6})

    # --- output paths ---
    output_dir = Path(output_dir)
    web_dir = Path(web_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "data").mkdir(parents=True, exist_ok=True)

    # --- write CSVs ---
    savings_out = savings.copy()
    # ensure boolean columns are true Python bool for CSV
    for col in ["is_confirmed_saving", "is_overuse"]:
        if col in savings_out.columns:
            savings_out[col] = savings_out[col].astype(bool)

    savings_out.to_csv(output_dir / "school_savings_predictions.csv", index=False, encoding="utf-8-sig")
    daily_agg.to_csv(output_dir / "school_savings_daily.csv", index=False, encoding="utf-8-sig")
    leaderboard.to_csv(output_dir / "school_savings_leaderboard.csv", index=False, encoding="utf-8-sig")

    # --- write scorecard JSON ---
    with open(output_dir / "school_savings_scorecard.json", "w", encoding="utf-8") as f:
        json.dump(scorecard, f, ensure_ascii=False, indent=2)

    # --- build run summary ---
    run_summary = {
        "run_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reporting_rows": len(savings),
        "used_fallback": band_result.used_fallback,
        "accuracy": accuracy,
        "calibration": calibration,
        "scorecard": scorecard,
    }
    with open(output_dir / "school_savings_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # --- build savings.json bundle ---
    # series (per-hour)
    def _safe(v):
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), 4)
        return v

    series_rows = []
    for _, row in savings.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        series_rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "report_date": pd.Timestamp(row["report_date"]).strftime("%Y-%m-%d") if pd.notna(row.get("report_date")) else ts.strftime("%Y-%m-%d"),
            "hour": int(row["hour"]),
            "actual": round(float(row["actual_kwh"]), 2) if np.isfinite(float(row["actual_kwh"])) else None,
            "p10": round(float(row["p10_kwh"]), 2),
            "p50": round(float(row["p50_kwh"]), 2),
            "p90": round(float(row["p90_kwh"]), 2),
            "avoided_kwh": round(float(row["avoided_kwh"]), 2) if np.isfinite(float(row["avoided_kwh"])) else None,
            "avoided_pct": round(float(row["avoided_pct"]), 4) if (pd.notna(row["avoided_pct"]) and np.isfinite(float(row["avoided_pct"]))) else None,
            "is_confirmed_saving": bool(row["is_confirmed_saving"]),
            "is_overuse": bool(row["is_overuse"]),
        })

    # daily
    daily_list = []
    for _, row in daily_agg.iterrows():
        daily_list.append({
            "report_date": row["report_date"],
            "baseline_kwh": round(float(row["baseline_kwh"]), 2),
            "actual_kwh": round(float(row["actual_kwh"]), 2),
            "avoided_kwh": round(float(row["avoided_kwh"]), 2),
            "avoided_pct": round(float(row["avoided_pct"]), 4) if np.isfinite(float(row["avoided_pct"])) else None,
        })

    # leaderboard
    lb_list = []
    for _, row in leaderboard.iterrows():
        lb_list.append({
            "round": row["round"],
            "days": int(row["days"]),
            "baseline_kwh": round(float(row["baseline_kwh"]), 2),
            "actual_kwh": round(float(row["actual_kwh"]), 2),
            "avoided_kwh": round(float(row["avoided_kwh"]), 2),
            "avoided_pct": round(float(row["avoided_pct"]), 4) if (row["avoided_pct"] is not None and np.isfinite(float(row["avoided_pct"]))) else None,
            "avoided_per_sqm_kwh": round(float(row["avoided_per_sqm_kwh"]), 4),
            "rank": int(row["rank"]),
        })

    # naive wape from accuracy
    naive_wape = accuracy.get("naive_wape")

    bundle = {
        "meta": {
            "reference_start": "2025-07-01 00:00:00",
            "reference_end": REFERENCE_END,
            "reporting_start": REPORTING_START,
            "reporting_end": REPORTING_END,
            "reporting_rows": len(savings),
            "reporting_days": 20,
            "total_area_sqm": total_area_sqm,
            "baseline_kind": "frozen day-of-week x hour profile + weather + academic (no usage lags)",
            "heldout_days": HELDOUT_DAYS,
            "used_fallback": band_result.used_fallback,
            "calibration": calibration,
            "note": "베이스라인은 대회 전 데이터로 동결 → 절감해도 베이스라인이 따라 내려가지 않음",
        },
        "accuracy": {
            "wape": accuracy.get("wape"),
            "mae_kwh": accuracy.get("mae_kwh"),
            "rmse_kwh": accuracy.get("rmse_kwh"),
            "coverage": accuracy.get("coverage"),
            "baselines": [
                {"model": "frozen P50", "wape": accuracy.get("wape")},
                {"model": "naive_same_dow_hour_profile", "wape": naive_wape},
            ],
        },
        "scorecard": scorecard,
        "series": series_rows,
        "daily": daily_list,
        "leaderboard": lb_list,
        "glossary": GLOSSARY,
    }

    bundle_path = web_dir / "data" / "savings.json"
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    avoided_pct_val = scorecard.get("avoided_pct") or float("nan")
    avoided_sum = scorecard.get("avoided_sum_kwh") or float("nan")
    avoided_per_sqm = scorecard.get("avoided_per_sqm_kwh") or float("nan")

    return SavingsRunResult(
        output_dir=output_dir,
        reporting_rows=len(savings),
        avoided_pct=float(avoided_pct_val) if avoided_pct_val is not None else float("nan"),
        avoided_sum_kwh=float(avoided_sum) if avoided_sum is not None else float("nan"),
        avoided_per_sqm_kwh=float(avoided_per_sqm) if avoided_per_sqm is not None else float("nan"),
        heldout_wape=float(accuracy.get("wape") or float("nan")),
        heldout_coverage=float(accuracy.get("coverage") or float("nan")),
        used_fallback=band_result.used_fallback,
    )
