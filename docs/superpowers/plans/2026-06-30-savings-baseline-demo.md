# Savings-vs-Baseline Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a *frozen* (M&V-style) baseline model that estimates "how much a school would have used if it hadn't changed behavior," score the reporting period as savings-vs-baseline in **% and per-㎡**, prove the baseline is accurate on held-out data, and ship a static demo website making the case that modest, already-available data (hourly meter + public weather + academic calendar) enables fair savings comparison without expensive equipment.

**Architecture:** New `modeling/school_savings_baseline.py` reuses feature functions from `modeling/school_hourly_model.py` and area parsing from `modeling/building_area_features.py`. It builds a **frozen (day-of-week × hour) usage profile** from the baseline period only, trains LightGBM quantile baselines (P10/P50/P90) on **calendar + frozen profile + weather + academic** (NO usage lags → no baseline chasing), conformal-calibrates the band, computes savings/scorecard/leaderboard, and writes `outputs/school_savings_*` + `school_savings_web/data/savings.json`. A new `school_savings_web/` renders it. Nothing existing is modified.

**Tech Stack:** Python 3.12 (`.venv\Scripts\python.exe`), pandas 3.0, numpy, LightGBM 4.6 quantile, openpyxl (area xlsx), pytest, static HTML/CSS/JS + inline SVG.

## Global Constraints

- Python only via `.venv\Scripts\python.exe` (no system Python). Tests: `.venv\Scripts\python.exe -m pytest <path> -q`.
- Baseline (reference) period: `2025-07-01 00:00:00 .. 2026-05-31 23:00:00`. Reporting period: `2026-06-01 00:00:00 .. 2026-06-20 23:00:00` (480h/20d).
- Held-out = last 28 days of the baseline period (used for BOTH CQR calibration and accuracy proof). Models fit on the PROPER subset (baseline minus held-out).
- **Frozen-baseline leakage/chasing guard:** `FROZEN_FEATURES` MUST NOT contain any of `school_lag_1h_kwh`, `school_lag_24h_kwh`, `school_lag_168h_kwh`, `school_rolling_24h_mean_kwh`, `school_rolling_168h_mean_kwh`, `school_same_hour_7d_mean_kwh`. It MUST contain `frozen_profile_kwh`. The frozen profile is computed using ONLY rows with timestamp ≤ baseline reference end (never reporting-period rows).
- Quantiles (0.1, 0.5, 0.9); enforce row-wise p10≤p50≤p90; predictions clipped ≥0. CQR symmetric widening on the held-out set (level `ceil((n+1)*0.8)/n`, method "higher").
- Total floor area constant `TOTAL_AREA_SQM = 447761.8` (㎡), or compute via `building_area_features` (header=2, sum `total_area_sqm`).
- savings convention: `avoided_kwh = p50_baseline − actual` (POSITIVE = saved, NEGATIVE = over-used).
- Do NOT modify any existing file except `requirements.txt` (already has `openpyxl` added). Only ADD `school_savings_*` artifacts + `school_savings_web/`. Do not touch `web/`, `school_hourly_web/`, `school_overuse_web/`, or existing `modeling/*`/`outputs/*`.
- Korean-first user-facing copy. CSV `utf-8-sig`, JSON UTF-8. Each task `git add`s only its own files.

### Reused interfaces (import)

```python
from modeling.school_hourly_model import (
    load_hourly_usage, add_hourly_features, add_weather_features, add_academic_features,
    split_train_validation, build_feature_frame, load_optional_table,
    DEFAULT_DATA_PATH, DEFAULT_ACTUAL_WEATHER_PATH, DEFAULT_FORECAST_WEATHER_PATH,
    DEFAULT_ACADEMIC_FEATURES_PATH, TARGET_COLUMN, CALENDAR_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
)
from modeling.academic_calendar_features import ACADEMIC_FEATURE_COLUMNS
```

`add_hourly_features` also creates the lag columns — that is fine; we simply DO NOT include them in `FROZEN_FEATURES`. The leakage guard test enforces their absence from the feature list.

---

### Task 1: Frozen baseline model + held-out accuracy (TDD)

**Files:** Create `modeling/school_savings_baseline.py`, `tests/test_school_savings_baseline.py`.

**Interfaces produced:**
- `REFERENCE_END = "2026-05-31 23:00:00"`, `REPORTING_START = "2026-06-01 00:00:00"`, `REPORTING_END = "2026-06-20 23:00:00"`, `HELDOUT_DAYS = 28`, `TOTAL_AREA_SQM = 447761.8`, `QUANTILES = (0.1, 0.5, 0.9)`.
- `FROZEN_FEATURES: list[str]` = CALENDAR_FEATURE_COLUMNS + ["frozen_profile_kwh"] + WEATHER_FEATURE_COLUMNS + ACADEMIC_FEATURE_COLUMNS.
- `load_total_area_sqm(path=Path("inputs/yu_building_area.xlsx")) -> float` (header=2, sum total_area_sqm; fallback to TOTAL_AREA_SQM on error).
- `build_savings_frame(...) -> pd.DataFrame` (load usage + hourly features + weather + academic; same shape as overuse pipeline).
- `add_frozen_profile(frame, reference_end=REFERENCE_END) -> pd.DataFrame` adds `frozen_profile_kwh` from (day_of_week,hour) mean over rows with timestamp ≤ reference_end.
- `FrozenBandResult` dataclass: predictions (reporting rows + p10/p50/p90), p50_model, x_reporting, feature_columns, calibration: dict, accuracy: dict (held-out wape/mae/rmse/coverage), used_fallback: bool.
- `train_frozen_band(frame, reference_end=REFERENCE_END, reporting_start=REPORTING_START, reporting_end=REPORTING_END, heldout_days=HELDOUT_DAYS, quantiles=QUANTILES) -> FrozenBandResult`.

- [ ] **Step 1: Write failing tests** `tests/test_school_savings_baseline.py`:

```python
from pathlib import Path
import numpy as np, pandas as pd, pytest
from modeling import school_savings_baseline as ssb

DATA = Path("school_power_usage_split/ml_ready/power_usage_1hour_ml.csv")

def test_frozen_features_have_no_usage_lags():
    banned = {"school_lag_1h_kwh","school_lag_24h_kwh","school_lag_168h_kwh",
              "school_rolling_24h_mean_kwh","school_rolling_168h_mean_kwh","school_same_hour_7d_mean_kwh"}
    assert banned.isdisjoint(set(ssb.FROZEN_FEATURES))
    assert "frozen_profile_kwh" in ssb.FROZEN_FEATURES

def test_frozen_profile_ignores_reporting_period():
    frame = ssb.build_savings_frame(DATA)
    out = ssb.add_frozen_profile(frame, reference_end=ssb.REFERENCE_END)
    # frozen_profile for a (dow,hour) equals the baseline-only mean, not the all-data mean
    ref = out[out["timestamp"] <= pd.Timestamp(ssb.REFERENCE_END)]
    g = ref.groupby(["day_of_week","hour"])["usage_kwh"].mean()
    sample = out.dropna(subset=["frozen_profile_kwh"]).iloc[0]
    expected = g.loc[(sample["day_of_week"], sample["hour"])]
    assert abs(sample["frozen_profile_kwh"] - expected) < 1e-6

def test_train_frozen_band_shapes_and_accuracy():
    frame = ssb.build_savings_frame(DATA)
    frame = ssb.add_frozen_profile(frame)
    result = ssb.train_frozen_band(frame)
    p = result.predictions
    assert (p["p10_kwh"] <= p["p50_kwh"] + 1e-6).all()
    assert (p["p50_kwh"] <= p["p90_kwh"] + 1e-6).all()
    assert (p["p10_kwh"] >= 0).all()
    # reporting period length = 480 hours
    assert len(p) == 480
    # held-out accuracy reported and sane
    assert 0 < result.accuracy["wape"] < 0.5
    assert 0.0 <= result.accuracy["coverage"] <= 1.0
    assert result.calibration["applied"] is True

def test_load_total_area_positive():
    assert ssb.load_total_area_sqm() > 100000  # campus is hundreds of thousands of m^2
```

- [ ] **Step 2: Run tests, confirm fail.**
- [ ] **Step 3: Implement `modeling/school_savings_baseline.py`.** Key logic:
  - `build_savings_frame`: mirror `school_overuse_model.build_modeling_frame` (load_hourly_usage → add_hourly_features → add_weather_features → add_academic_features).
  - `add_frozen_profile`: `ref = frame[frame.timestamp <= reference_end]; prof = ref.groupby(["day_of_week","hour"])["usage_kwh"].mean().rename("frozen_profile_kwh"); merge on [day_of_week,hour]`.
  - `train_frozen_band`: split frame into reference (ts ≤ reference_end) and reporting (reporting_start..end). Within reference, calib = last `heldout_days` days, proper = earlier. Fit 3 quantile LGBM (objective="quantile", alpha=q, seed, verbosity=-1) on `build_feature_frame(proper, feature_columns=FROZEN_FEATURES)`. Predict raw band on calib + reporting. CQR: `s=max(p10raw-y, y-p90raw)` on calib; `Q=quantile(s, ceil((n+1)*0.8)/n, method="higher")`; reporting band `p10=max(p10raw-Q,0)`, `p90=p90raw+Q`, re-sort monotone. Held-out accuracy = WAPE/MAE/RMSE of p50 on calib + coverage (calibrated band) on calib. LightGBM-free fallback like overuse model. Reuse `build_feature_frame` for imputation with `reference=proper`.
- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Commit** `git add modeling/school_savings_baseline.py tests/test_school_savings_baseline.py` → `feat: add frozen M&V baseline model with held-out accuracy`.

---

### Task 2: Savings, scorecard, leaderboard, pipeline, outputs + savings.json (TDD)

**Files:** Modify `modeling/school_savings_baseline.py`; create `scripts/run_school_savings_demo.py`; test `tests/test_school_savings_baseline.py`.

**Interfaces produced:**
- `compute_savings(predictions, total_area_sqm) -> pd.DataFrame` adds `actual_kwh, avoided_kwh (=p50-actual), avoided_pct (=avoided/p50), avoided_kwh_per_sqm (=avoided/total_area), is_confirmed_saving (actual<p10), is_overuse (actual>p90)`.
- `build_scorecard(savings, accuracy, total_area_sqm) -> dict`: keys `baseline_sum_kwh, actual_sum_kwh, avoided_sum_kwh, avoided_pct (sum avoided / sum p50), avoided_per_sqm_kwh, confirmed_saving_hours, overuse_hours, n_rows, heldout_wape, heldout_coverage`.
- `build_leaderboard(savings, total_area_sqm, round_days=7) -> pd.DataFrame`: split reporting dates into 7-day rounds (label e.g. "1주차 (06-01~06-07)"); per round `baseline_kwh, actual_kwh, avoided_kwh, avoided_pct, avoided_per_sqm_kwh`; rank by `avoided_pct` desc (rank 1 = best). Include all rounds even partial last one.
- `run_savings_demo(data_path=..., output_dir=Path("outputs"), web_dir=Path("school_savings_web"), area_path=...) -> SavingsRunResult` (dataclass: output_dir, reporting_rows, avoided_pct, avoided_sum_kwh, avoided_per_sqm_kwh, heldout_wape, heldout_coverage, used_fallback). Writes the outputs + `school_savings_web/data/savings.json`.

**`savings.json` schema (model↔web contract — EXACT):**

```json
{
  "meta": {"reference_start":"2025-07-01 00:00:00","reference_end":"2026-05-31 23:00:00",
           "reporting_start":"2026-06-01 00:00:00","reporting_end":"2026-06-20 23:00:00",
           "reporting_rows":480,"reporting_days":20,"total_area_sqm":447761.8,
           "baseline_kind":"frozen day-of-week x hour profile + weather + academic (no usage lags)",
           "heldout_days":28,"used_fallback":false,
           "calibration":{"applied":true,"q_kwh":..,"n_proper":..,"n_calib":..},
           "note":"베이스라인은 대회 전 데이터로 동결 → 절감해도 베이스라인이 따라 내려가지 않음"},
  "accuracy": {"wape":..,"mae_kwh":..,"rmse_kwh":..,"coverage":..,
               "baselines":[{"model":"frozen P50","wape":..},{"model":"naive_same_dow_hour_profile","wape":..}]},
  "scorecard": { ...build_scorecard keys... , "avoided_pct_display":.., "area_sqm":447761.8 },
  "series": [ {"timestamp":"..","report_date":"..","hour":0,"actual":..,"p10":..,"p50":..,"p90":..,
               "avoided_kwh":..,"avoided_pct":..,"is_confirmed_saving":false,"is_overuse":false} ],
  "daily": [ {"report_date":"2026-06-01","baseline_kwh":..,"actual_kwh":..,"avoided_kwh":..,"avoided_pct":..} ],
  "leaderboard": [ {"round":"1주차 (06-01~06-07)","days":7,"baseline_kwh":..,"actual_kwh":..,
                    "avoided_kwh":..,"avoided_pct":..,"avoided_per_sqm_kwh":..,"rank":1} ],
  "glossary": [ {"term":"동결 베이스라인","desc":".."}, ... ]
}
```

Numbers rounded to 2 decimals (per_sqm to 4). series sorted by timestamp; leaderboard sorted by rank. The naive accuracy comparator = predicting the frozen (dow,hour) profile directly (WAPE of frozen_profile_kwh vs actual on the held-out calib set).

- [ ] **Step 1: Write failing tests** (append):

```python
def test_compute_savings_math():
    df = pd.DataFrame({"usage_kwh":[100.0,200.0], "p10_kwh":[90.0,150.0],
                       "p50_kwh":[120.0,180.0], "p90_kwh":[140.0,210.0]})
    out = ssb.compute_savings(df, total_area_sqm=1000.0)
    assert out.loc[0,"avoided_kwh"] == pytest.approx(20.0)   # 120-100 saved
    assert out.loc[0,"is_confirmed_saving"] == True          # 100<90? no -> False actually
    # row0 actual 100 not < p10 90 -> not confirmed; row1 actual 200 > p90 210? no
    assert out.loc[1,"avoided_kwh"] == pytest.approx(-20.0)  # over-used
    assert out.loc[0,"avoided_kwh_per_sqm"] == pytest.approx(20.0/1000.0)

def test_scorecard_and_leaderboard_and_bundle(tmp_path):
    res = ssb.run_savings_demo(output_dir=tmp_path/"outputs", web_dir=tmp_path/"web")
    import json
    bundle = json.loads((tmp_path/"web"/"data"/"savings.json").read_text(encoding="utf-8"))
    for k in ["meta","accuracy","scorecard","series","daily","leaderboard","glossary"]:
        assert k in bundle
    assert bundle["meta"]["reporting_rows"] == len(bundle["series"]) == 480
    assert len(bundle["daily"]) == 20
    assert len(bundle["leaderboard"]) >= 3
    assert all("rank" in r for r in bundle["leaderboard"])
    assert bundle["scorecard"]["area_sqm"] > 100000
    assert res.reporting_rows == 480
```

(Fix the inline comment logic when implementing; assert the true expected values.)

- [ ] **Step 2: Run, confirm fail.** **Step 3: Implement** compute_savings/build_scorecard/build_leaderboard/run_savings_demo + JSON bundle + `scripts/run_school_savings_demo.py` (mirror `scripts/run_school_overuse_model.py`). **Step 4:** Run tests + `.venv\Scripts\python.exe scripts/run_school_savings_demo.py` for real; report avoided_pct, avoided_per_sqm, heldout_wape/coverage. **Step 5: Commit** module+tests+script+outputs+savings.json → `feat: add savings scorecard, leaderboard, pipeline and web bundle`.

---

### Task 3: Demo website (TDD on content)

**Files:** Create `school_savings_web/index.html`, `styles.css`, `app.js`; `tests/test_school_savings_web_content.py`.

Reads ONLY `school_savings_web/data/savings.json`. Korean-first. Inline SVG. No external CDN.

Sections/tabs:
1. **주제(히어로)**: "큰 설비 없이, 학교에 이미 있는 데이터(시간별 검침 + 공공 날씨 + 학사일정)만으로 공정한 절감 비교." + 한 줄 설명.
2. **정확도 증명**: KPI 카드 — held-out WAPE(%), coverage(%), naive 대비. "이 데이터로 베이스라인이 이만큼 정확."
3. **베이스라인 대비 절감(차트)**: 리포팅 기간 동결 베이스라인 밴드 + 실제선 SVG; 실제<베이스라인 구간 초록 음영(절감), 실제>P90 빨강 점(초과). 날짜 선택 또는 일별 보기.
4. **스코어카드**: 총 절감 kWh, 절감률 %, 면적당 절감(kWh/㎡), 확실한 절감 시간 수.
5. **리더보드**: leaderboard[] 막대/표 — %절감 + 면적당, 순위. 주석 "여기선 같은 학교의 주차를 참가자로 시연 — 학교 A·B·C로 바꿔도 동일하게 동작."
6. **공정성·확장·용어 설명**: 동결 베이스라인/baseline chasing/절감률/면적당/M&V 설명(glossary[]).

- [ ] **Step 1: Failing web-content test** `tests/test_school_savings_web_content.py`: assert 3 files exist; index.html contains "절감","베이스라인","면적당","리더보드","정확도","용어"; app.js references 'data/savings.json' and fields scorecard/series/leaderboard/accuracy/avoided/p50/p90/glossary; a JSON-contract test loading savings.json asserting top-level keys + series[0] keys + leaderboard[0] keys + all series numerics finite.
- [ ] **Step 2:** Run, confirm fail. **Step 3:** Build the 3 files. **Step 4:** pytest pass + serve `.venv\Scripts\python.exe -m http.server 8770 --directory school_savings_web` and confirm 200 for page + savings.json (controller drives browser visual check). **Step 5: Commit** → `feat: add savings demo static web viewer`.

---

### Task 4: Run record + full verification

**Files:** Create `docs/runs/2026-06-30-savings-baseline-v1.md`.

- [ ] **Step 1:** Read real numbers from `outputs/school_savings_scorecard.json` + `savings.json`. **Step 2:** Write Korean run record: 목적(데이터 충분성·공정성 증명), 동결 베이스라인 방법(no-lag, M&V, baseline chasing 회피), 기간/held-out, 정확도(WAPE/coverage + naive 비교), 스코어카드(절감 kWh/%/면적당), 리더보드 요약, 한계(동결 베이스라인은 행동 외 구조 변화 미반영; 단일 학교 시연), 웹 사용법. **Step 3:** Full suite `.venv\Scripts\python.exe -m pytest tests/ -q` → record summary, confirm no regressions. **Step 4: Commit** → `docs: add savings baseline run record v1`.

---

## Self-Review (author)
- Frozen leakage/chasing guard tested (T1). Accuracy proof produced + documented (T1/T2/T4). Savings %/per-㎡ + leaderboard (T2). Web tells the data-sufficiency + fairness story with glossary (T3). Run doc from real numbers (T4). ✔
- JSON contract pinned (T2) and consumed with matching field names (T3). ✔
- No existing file modified except requirements.txt (openpyxl). ✔
