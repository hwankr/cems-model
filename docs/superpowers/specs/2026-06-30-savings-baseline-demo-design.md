# 베이스라인 대비 절감 비교 데모 설계 (Savings-vs-Baseline Demo)

## 목적 / 메시지

**"큰 설비(서브미터·IoT 센서) 없이, 거의 모든 학교에 이미 있는 데이터 — 시간별 전력 검침 + 공공 기상(기상청) + 학사일정 — 만으로, 각 학교를 *자기 베이스라인 대비* 공정하게 절감 비교할 수 있다"** 를 실제 한 학교 데이터로 증명하는 데모.

학교 간 대결은 가정이다. 핵심 증명 2가지:
1. **데이터 충분성**: 동결(frozen) 베이스라인이 보유 데이터만으로 충분히 정확하다(held-out 정확도로 입증).
2. **공정성**: 절대 kWh가 아니라 *자기 베이스라인 대비 %* 또는 *면적당 kWh* 로 채점하면 규모가 달라도 공정하다.

## 핵심 방법론 (왜 "동결 베이스라인"인가)

기존 `school_overuse_model`은 베이스라인 피처로 학교 자신의 최근 사용량 lag(24h/168h/7일평균)을 쓴다. 이는 단기 이상탐지엔 좋지만 절감 대회엔 **치명적**이다: 학교가 꾸준히 절약하면 최근 사용량이 낮아져 베이스라인도 따라 내려가고("baseline chasing"), 절감 점수가 사라진다.

**해결 = M&V(IPMVP Option C) 스타일 동결 베이스라인:**
- 베이스라인 모델은 **리포팅(대회) 기간 *이전* 데이터로만 학습**한다.
- 리포팅 기간 예측에 **리포팅 기간의 실제 사용량을 입력으로 쓰지 않는다.**
- 피처 = 달력(hour/dow/weekend/month/sin/cos) + **동결 프로필**(`frozen_profile_kwh` = 기준기간의 (요일×시간) 평균 사용량, 한 번 계산해 고정) + 기상(예보/실측) + 학사일정. **사용량 lag·rolling 미사용.**
- 동결 프로필이 부하 형상(요일·시간 패턴)을 고정 앵커로 제공하므로, lag 없이도 정확도를 상당 부분 회복하면서 chasing을 차단한다.

## 데이터 / 상수

- 입력: `school_power_usage_split/ml_ready/power_usage_1hour_ml.csv` (학교 전체 시간별, target `usage_kwh`).
- 기상: `outputs/weather_actual_daily.csv`(~2026-06-23), `outputs/weather_forecast_daily.csv`(2026-04~06). 학사일정: `outputs/yu_academic_calendar_daily_features.csv`.
- **총 연면적 = 447,761.8 ㎡** (179개 건물, `inputs/yu_building_area.xlsx`, `modeling/building_area_features.py`로 파싱: header=2, 합계 `total_area_sqm`). 면적당 정규화 분모.
- `requirements.txt`에 `openpyxl` 추가(xlsx 읽기).

## 기간 분리

- **기준(베이스라인) 기간**: `2025-07-01 ~ 2026-05-31`.
  - 그 중 **마지막 28일(≈5월)을 held-out**으로 떼어 (a) CQR 보정과 (b) **정확도 증명**(WAPE/coverage)에 사용. 모델은 그 이전(proper) 데이터로 학습.
- **리포팅(데모) 기간**: `2026-06-01 ~ 2026-06-20` (480시간/20일). 여기서 절감을 채점.
- 누수 차단: 동결 프로필·모델 모두 리포팅 기간 실측을 보지 않는다.

## 컴포넌트

### 1. `modeling/school_savings_baseline.py` (신규)

기존 피처 함수(`school_hourly_model`의 load/feature/split/build_feature_frame, `building_area_features`)를 import 재사용.

- `TOTAL_AREA_SQM = 447761.8` (또는 `building_area_features`로 계산하는 헬퍼 `load_total_area_sqm(path)`).
- `add_frozen_profile(frame, reference_end) -> frame`: 기준기간(timestamp ≤ reference_end) 행으로 `(day_of_week, hour)` 평균 `usage_kwh` 계산 → 전 행에 `frozen_profile_kwh`로 매핑.
- `FROZEN_FEATURES` = 달력 + `frozen_profile_kwh` + 기상 + 학사일정. (사용량 lag/rolling 제외.)
- `train_frozen_band(proper, calib, reporting, quantiles=(0.1,0.5,0.9)) -> FrozenBandResult`: LightGBM 분위수 회귀 P10/P50/P90 (frozen 피처). CQR 보정(held-out calib로 q 계산, 양쪽 확장, 단조 정렬, ≥0). 반환: 리포팅 예측 p10/p50/p90, p50 모델, 피처행렬, calibration dict, held-out 정확도(WAPE/MAE/RMSE/coverage on calib).
- `compute_savings(predictions, total_area_sqm) -> df`: `avoided_kwh = p50 − actual`(양수=절감), `is_confirmed_saving = actual < p10`, `is_overuse = actual > p90`, `saving_pct_row`, `avoided_kwh_per_sqm`.
- `build_scorecard(savings, accuracy) -> dict`: 합계 baseline/actual/avoided kWh, 절감률 %(Σavoided/Σp50), 면적당 절감, 확실한 절감 시간 수, 초과 시간 수, held-out 정확도, coverage.
- `build_leaderboard(savings) -> df`: 리포팅 기간을 **주차(round)**로 분할(Jun1–7 / 8–14 / 15–20 등 7일 단위) → 각 라운드 %절감·면적당 절감으로 순위. (학교 A·B·C 대신 실데이터 라운드로 랭킹 시연.)
- `run_savings_demo(...) -> SavingsRunResult`: 전체 파이프라인 + 산출물 저장 + `school_savings_web/data/savings.json` 생성.

### 2. 산출물 (`outputs/school_savings_*`, 기존 파일 불변)

`school_savings_predictions.csv`(시간별 baseline+actual+avoided), `school_savings_daily.csv`, `school_savings_leaderboard.csv`, `school_savings_scorecard.json`, `school_savings_run_summary.json`, `school_savings_web/data/savings.json`.

`savings.json` 스키마: `meta`(기간/면적/frozen 설명/calibration), `accuracy`(held-out wape/mae/rmse/coverage + naive 비교), `scorecard`(상기), `series`(시간별 timestamp/report_date/hour/actual/p10/p50/p90/avoided_kwh/is_confirmed_saving/is_overuse), `daily`(일별 baseline/actual/avoided/%), `leaderboard`(라운드별 label/days/baseline/actual/avoided/pct/per_sqm/rank), `glossary`(용어).

### 3. 새 웹 `school_savings_web/` (정적, 한국어 우선)

- **히어로/주제**: "큰 설비 없이, 학교에 이미 있는 데이터만으로 공정한 절감 비교."
- **정확도 증명 카드**: held-out WAPE/coverage + naive 대비 — "이 데이터로 베이스라인이 이만큼 정확."
- **베이스라인 vs 실제 차트**(리포팅 기간): 동결 베이스라인 밴드 + 실제선, 절감분(실제<베이스라인) 초록 음영 / 초과분 빨강.
- **스코어카드 KPI**: 총 절감 kWh, 절감률 %, 면적당 절감, 확실한 절감 시간.
- **리더보드(주차별)**: %절감 + 면적당 막대/표 순위. "학교 A·B·C로 바꿔도 동일하게 동작" 주석.
- **공정성·확장·용어 설명** 탭/섹션. (동결 베이스라인, 절감률, 면적당, M&V, baseline chasing 등 용어.)

### 4. 테스트 / 문서

- `tests/test_school_savings_baseline.py`(TDD): 동결 프로필이 리포팅 실측 비참조(누수 가드), FROZEN_FEATURES에 lag/rolling 없음, 절감 계산식, 스코어카드/리더보드 집계, 밴드 단조성, 산출물 스키마.
- `tests/test_school_savings_web_content.py`: 새 웹 필수 텍스트/필드/데이터 계약(+ savings.json 스키마·유한성 검증).
- `docs/runs/2026-06-30-savings-baseline-v1.md`(한국어 실행기록, 실수치 기반).
- 기존 모델·웹(`web/`, `school_hourly_web/`, `school_overuse_web/`, 기존 모듈)은 **불변**. 신규 `school_savings_*`만 추가.

## 승인 기준

- 동결 베이스라인이 lag/rolling 없이(누수·chasing 차단) 리포팅 기간 절감을 산출한다.
- held-out 정확도(WAPE/coverage)가 문서화되어 "데이터 충분성"을 입증한다.
- 절감을 **% 및 면적당**으로 채점하고 주차 리더보드로 공정 랭킹을 시연한다.
- 새 웹이 정확도 증명·베이스라인 대비 절감·리더보드·용어를 보여준다.
- 신규 테스트 통과, 기존 테스트·파일 무영향.
