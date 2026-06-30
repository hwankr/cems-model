# 학교 전력 과다사용 모니터 설계 (School Over-Consumption Monitor)

## 목표

학교 전체 1시간 단위 전력 사용량에 대해, 매 시간마다 **정상 범위 `[P10, P90]`** 와 **기대값 `P50`** 를 LightGBM 분위수 회귀로 산출한다. `실제값 > P90` 인 시간을 **과다사용**으로 표시·정량화하고, 그 시간의 예측 근거를 **SHAP(트리 모델 기여도)** 로 설명한다. 검증 구간은 **2026-06-01 ~ 2026-06-20** 이다.

핵심 메시지: "과거 패턴·날씨·학사일정으로 볼 때 이 시간엔 `x~y kWh` 가 정상인데, 지금 `z kWh` 를 쓰고 있고 정상 상한보다 `(z−y) kWh` 더 쓰는 중이다. 그 이유는 트리 모델이 보기에 …"

## 배경 / 기존 자산

- 입력: `school_power_usage_split/ml_ready/power_usage_1hour_ml.csv` (8,696행, 2025-07-01 ~ 2026-06-30, target `usage_kwh`).
- 기존 점예측 모델 `modeling/school_hourly_model.py` 에 검증된 피처 엔지니어링이 이미 있다: `load_hourly_usage`, `add_hourly_features`, `add_weather_features`, `add_academic_features`, `split_train_validation`, `build_feature_frame`, day-ahead/operational 피처 컬럼 정의.
- 보조 피처 소스: `outputs/weather_actual_daily.csv`, `outputs/weather_forecast_daily.csv`(KMA API 산출), `outputs/yu_academic_calendar_daily_features.csv`(학사일정).
- **이 작업은 기존 17개 건물 모델, `web/`, `school_hourly_web/`, 그리고 `school_hourly_model.py` 의 동작을 일절 변경하지 않는다.** 새 모듈/새 산출물 prefix/새 웹 폴더로만 추가한다.

## 범위 (YAGNI)

- 포함: 분위수 밴드 모델, 과다사용 플래그/정량화, SHAP 설명, 산출물, 새 정적 웹, 용어 설명, TDD, 실행기록.
- 제외: 실시간 스트리밍/배포, DB, 인증, 15분/30분 해상도(향후), 건물별 분해(향후), 자동 재학습 스케줄러.

## 설계 결정 (확정)

1. **정상 범위 정의** = LightGBM 분위수 회귀 P10/P50/P90 밴드. 과다사용 = `actual > P90`.
2. **기준(피처) 계약** = **day-ahead**. 당일 직전 1시간 lag·롤링 피처를 쓰지 않아 "지금 평소보다 많이 쓰는 중"을 탐지할 수 있다.
3. **웹** = `school_overuse_web/` 완전 신규 페이지(한국어 우선). 기존 웹은 보존.
4. **용어 설명** = 웹과 실행기록에 핵심 전문용어 설명을 포함.

## 컴포넌트와 인터페이스

### 1. `modeling/school_overuse_model.py` (신규)

기존 `school_hourly_model.py` 의 피처 함수를 import 재사용한다(중복 구현 금지).

- `DAY_AHEAD_FEATURES` = `school_hourly_model.DAY_AHEAD_WEATHER_ACADEMIC_FEATURE_COLUMNS` 재사용.
  - 달력: `month, day, day_of_week, is_weekend, hour, hour_sin, hour_cos, dow_sin, dow_cos`
  - 과거 부하: `school_lag_24h_kwh, school_lag_168h_kwh, school_same_hour_7d_mean_kwh`
  - 예보 날씨: `weather_temp_mean_c/min/max, weather_humidity_mean_pct, weather_rainfall_mm, weather_wind_mean_mps, weather_cdd_18, weather_hdd_18, weather_is_rainy`
  - 학사일정: `academic_is_instruction_period, academic_is_vacation, academic_is_midterm, academic_is_final, academic_is_makeup_class_day, academic_is_course_eval, academic_is_education_practice, academic_days_since_semester_start, academic_days_to_final, academic_semester_week`
  - **제외(누수 방지)**: `school_lag_1h_kwh`, `school_rolling_24h_mean_kwh`, `school_rolling_168h_mean_kwh`.

- `QUANTILES = (0.1, 0.5, 0.9)`
- `train_quantile_band(train, validation, quantiles=QUANTILES) -> QuantileBandResult`
  - 분위수마다 `LGBMRegressor(objective="quantile", alpha=q, ...)` 학습. 시드 고정.
  - 반환: 검증행별 `p10/p50/p90`, 적합된 **P50 모델**과 검증 피처행렬(SHAP용), 학습 피처 목록.
  - **분위수 교차 보정**: 행별로 세 값을 정렬해 `p10 ≤ p50 ≤ p90` 강제.
  - **LightGBM 미설치 폴백**: P50 = 학습 같은시간 중앙값, 밴드 = 학습 잔차의 10/90 분위수로 대칭 근사. (테스트가 lightgbm 없이도 의미 있는 밴드를 받도록.)
- `flag_overuse(predictions) -> DataFrame`
  - `in_normal_band = p10 ≤ actual ≤ p90`
  - `is_overuse = actual > p90`, `is_underuse = actual < p10`
  - `exceedance_kwh = max(actual − p90, 0)`, `exceedance_pct = exceedance_kwh / p90`
  - `band_position = (actual − p10) / (p90 − p10)` (1 초과 시 밴드 위로 벗어남)
- `explain_band(p50_model, x_validation, predictions, top_k=5) -> (explanations_long, global_importance)`
  - `shap.TreeExplainer(p50_model)` 로 검증행 SHAP 값 계산.
  - 과다사용 시간별 상위 `top_k` 기여 피처(부호 포함)를 long-format으로: `timestamp, report_date, hour, rank, feature, feature_label_ko, shap_value_kwh, feature_value, direction`.
  - 전역 중요도: 피처별 평균 `|SHAP|` 내림차순.
  - SHAP 미설치 폴백: P50 모델 `feature_importances_` 기반 근사(부호 없음)로 대체하고 메타에 표시.
- 지표 `compute_band_metrics(predictions) -> dict`
  - `coverage` = 밴드 안 비율(보정되면 ≈0.8), `pinball_loss`(q별), P50 점예측 `wape/mae/rmse/bias`, `overuse_hours`, `overuse_total_exceedance_kwh`, `mean_band_width_kwh`.
- `run_school_overuse_model(...) -> SchoolOveruseRunResult` 가 전체 파이프라인을 돌리고 산출물 저장.

### 2. 산출물 (`outputs/`, prefix `school_overuse_`)

- `school_overuse_predictions.csv`: `timestamp, report_date, report_month, hour, day_of_week, is_weekend, actual_kwh, p10/p50/p90, in_normal_band, is_overuse, is_underuse, exceedance_kwh, exceedance_pct, band_position, pred_last_day_same_hour_kwh, pred_last_week_same_hour_kwh, pred_same_hour_7d_mean_kwh`.
- `school_overuse_explanations.csv`: 과다사용 시간 SHAP long-format.
- `school_overuse_feature_importance.csv`: 전역 평균 |SHAP| 순위.
- `school_overuse_daily_summary.csv`: 일별 과다사용 시간 수·총 초과 kWh·최대 초과시간 등.
- `school_overuse_metrics.json`: coverage, pinball, 점예측 지표, 과다사용 요약, 검증창.
- `school_overuse_run_summary.json`: 실행 메타.
- `school_overuse_web/data/monitor.json`: 웹 전용 컴팩트 번들(시계열 밴드 + 과다사용 + 설명 + 지표 + 용어집).

### 3. 새 웹 `school_overuse_web/` (정적 HTML/CSS/JS, 한국어 우선)

`fetch` 로 `data/monitor.json` 을 읽어 렌더. 탭:

- **개요**: KPI 카드 — 검증기간, 검증 시간 수, 정상밴드 적중률(coverage), 과다사용 시간 수, 총 초과 kWh, P50 정확도(WAPE).
- **시계열 모니터**: 실제선 + P50선 + `[P10,P90]` 음영 밴드(SVG/Canvas), 과다사용 시간 빨강 강조. 날짜 선택(6/1~6/20), 호버 시 actual/p10/p50/p90/초과량 툴팁.
- **과다사용 분석**: 초과량 내림차순 표 → 행 클릭 시 SHAP 가로 막대(상승=빨강, 하락=파랑)와 한글 피처 라벨로 근거 표시. "정상 기대 `x~y`, 실제 `z`, 초과 `z−y`" 요약 문구 포함.
- **모델 정확도(보조)**: P50 vs naive 베이스라인 WAPE/MAE/RMSE, coverage 보정 설명.
- **용어 설명**: 아래 용어집을 카드로 표시.

### 4. 용어 설명 (웹 + 실행기록에 포함, 핵심 위주)

- **분위수(Quantile)**: 데이터를 크기순으로 줄 세웠을 때 특정 비율 위치의 값. P90 = 하위 90% 지점.
- **P10 / P50 / P90**: 이 시간에 "정상이라면" 사용량이 P10 이상일 확률 90%, P50(중앙값) 근처가 가장 그럴듯, P90 이하일 확률 90%. `[P10,P90]` 안에 정상의 약 80%가 들어옴.
- **정상 밴드 / 과다사용**: `[P10,P90]` 가 정상 범위. `실제 > P90` 이면 과다사용, 초과량 `= 실제 − P90`.
- **day-ahead(하루 전 기준)**: 당일 직전 관측을 안 쓰고 날짜·요일·시간·예보날씨·학사일정·과거 부하만으로 기대치를 만든 것. 그래서 "지금 평소보다 많이 쓰는 중"을 잡아낼 수 있음.
- **SHAP**: 트리 모델의 예측을 각 피처가 얼마나 끌어올리고/내렸는지 kWh 단위로 분해한 기여도. "기온 +120, 시험기간 +80 …" 식.
- **WAPE**: 가중 절대 백분율 오차 = Σ|예측−실제| / Σ실제. 낮을수록 정확.
- **Pinball loss**: 분위수 예측 품질 지표. 낮을수록 분위수가 잘 맞음.
- **Coverage(적중률)**: 실제값이 `[P10,P90]` 안에 든 비율. 잘 보정되면 ≈80%.
- **CDD/HDD**: 냉방·난방 도일(degree-day). 기준온도 대비 더운/추운 정도의 누적량 → 냉난방 전력 수요 대리지표.

### 5. SHAP 해석 프레이밍 (정확성 주의)

SHAP는 *모델의 기대값(P50)이 왜 그 수준인지*를 설명한다. 따라서 과다사용 시간에 대해 두 가지를 함께 보여준다.

1. **맥락**: 어떤 피처가 이 시간의 기대 부하를 끌어올렸나(폭염·시험기간·평일 등).
2. **초과분**: 실제값이 밴드 상한(P90)을 넘은 양 = 모델이 예상하지 못한 추가 사용량.

"폭염·시험기간이라 원래 높게 기대됐는데 그보다도 더 썼다" vs "주말 야간이라 낮게 기대됐는데 실제는 크게 초과(강한 이상)" 를 구분해 해석할 수 있게 한다. 웹/문서에 이 의미를 명시한다.

## 검증 분리

- 학습: `2025-07-01 00:00` ~ `2026-05-31 23:00` (검증창 시작 이전 전부).
- 검증·채점: `2026-06-01 00:00` ~ `2026-06-20 23:00` (480시간, 20일).
- 제외: `2026-06-21` 이후 및 불완전한 `2026-06-30` 은 채점에서 제외.
- 누수 방지: 모든 lag/rolling은 `shift` 이후 계산(기존 모듈 규칙 그대로). day-ahead 계약은 동시각 직전 1시간 관측을 쓰지 않는다.

## 테스트 (TDD, 구현 전 작성)

`tests/test_school_overuse_model.py`:
- 밴드 단조성: `p10 ≤ p50 ≤ p90` (교차 보정 후).
- coverage 계산식 정확성(합성 데이터).
- 과다사용 플래그·`exceedance_kwh`·`exceedance_pct`·`band_position` 계산 검증.
- SHAP 설명 표의 스키마/정렬/상위 top_k 검증(폴백 경로 포함).
- day-ahead 피처 집합에 `school_lag_1h_kwh`/rolling이 **없음**(누수 가드).
- 산출물 파일 생성/스키마 검증.

`tests/test_overuse_web_content.py`(또는 기존 web content 테스트 확장): 새 웹의 필수 텍스트/요소(탭, 용어집 용어, data fetch 경로) 존재 확인.

## 산출 문서

- 본 설계 spec(이 파일).
- 구현 계획 `docs/superpowers/plans/2026-06-30-school-overuse-model-web.md` (writing-plans 단계에서 생성).
- 실행기록 `docs/runs/2026-06-30-school-overuse-v1.md` (한국어 우선, 실제 지표 기반).

## 승인 기준

- 분위수 밴드·과다사용·SHAP 설명이 day-ahead 계약으로 2026-06-01~06-20에서 산출된다.
- coverage가 합리적 범위(대략 0.7~0.9)이고 보정 상태가 문서화된다.
- 새 웹이 밴드·과다사용·시간별 SHAP 근거·용어 설명을 보여준다.
- 모든 신규 테스트 통과, 기존 테스트 영향 없음.
- 기존 모델/웹 산출물 미변경.
