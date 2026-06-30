# 2026-06-30 school-overuse-v1 실행 기록

## 목적

학교 전체 hourly 전력의 '정상 범위(P10~P90) + 과다사용 탐지 + SHAP 근거' 모델이다. 기존 점예측 비교 모델(`school_hourly`)과 분리된 별도 모델로, 예측 자체가 목적이 아니라 "언제, 얼마나 정상을 벗어났는가"를 실시간으로 판단하고 그 근거를 SHAP로 제공하는 모니터링 시스템이다.

입력 데이터는 `school_power_usage_split/ml_ready/power_usage_1hour_ml.csv`이고 target은 `usage_kwh`이다.

## 검증 기준

- 학습 구간: `2025-07-01 00:00:00`부터 `2026-05-31 23:00:00`까지
- 공식 검증 구간: `2026-06-01 00:00:00`부터 `2026-06-20 23:00:00`까지
- 검증 행: **480시간**, **20일**
- 제외: `2026-06-21` 이후 및 불완전한 `2026-06-30` 제외

피처 계약은 **day-ahead**이다. 각 timestamp를 예측할 때 동시각 직전 1시간 lag만 허용하고 rolling을 사용하지 않아 누수를 방지한다. CQR calibration에는 최근 28일(`2026-05-04` ~ `2026-05-31`, n_calib=672행)을 사용했으며, 학습용 proper set은 n_proper=7317행이다.

## 모델

LightGBM 분위수 회귀 P10/P50/P90 (`objective=quantile`)를 세 개 독립 학습했다.

- 피처 그룹: 달력(월/요일/시간/휴일/공휴일) + 과거 부하(24h/168h/7일평균) + 예보 날씨 + 학사일정
- 모델 식별자: `lightgbm_quantile_day_ahead_weather_academic`
- SHAP 계산: P50 모델에 대해 TreeExplainer 적용, `shap_available=true`

## 정상밴드 보정(CQR)

Conformalized Quantile Regression(CQR)으로 정상 밴드 폭을 보정했다.

| 항목 | 값 |
| --- | --- |
| calibration_days | 28일 |
| n_proper | 7,317행 |
| n_calib | 672행 |
| q_kwh (양쪽 확장량) | 108.83 kWh |
| 보정 후 calibration coverage | 80.36% |
| 목표 coverage | 80.0% |

보정 전(`coverage_raw`) 검증 coverage는 58.96%였지만, CQR 적용 후 검증 coverage는 **78.33%**다. 목표 80%에 0.017 pt 못 미친다. 이는 2026년 6월이 CQR calibration 기준(2026년 5월)보다 높게 전력을 사용했다는 의미로, 밴드 안에 들어오지 못한 시간이 그만큼 많았음을 나타낸다.

## 결과

### 커버리지·과다사용

| 지표 | 값 |
| --- | --- |
| coverage (보정 후, 검증) | 78.33% |
| coverage_raw (보정 전, 검증) | 58.96% |
| coverage 목표 | 80.00% |
| overuse_hours (P90 초과 시간) | 72시간 |
| underuse_hours (P10 미달 시간) | 32시간 |
| overuse_total_exceedance_kwh | 17,287.46 kWh |
| mean_band_width_kwh (평균 밴드 폭) | 645.27 kWh |

### P50 점예측 정확도

| 지표 | 값 |
| --- | --- |
| WAPE | 7.57% |
| MAE | 237.80 kWh |
| RMSE | 419.57 kWh |
| Bias | -3.68% |

### Pinball Loss

| 분위수 | Pinball Loss |
| --- | ---: |
| P10 | 72.578 |
| P50 | 118.901 |
| P90 | 60.162 |

### P50 vs naive 베이스라인 WAPE 비교

| 모델 | WAPE | MAE (kWh) | RMSE (kWh) |
| --- | ---: | ---: | ---: |
| **P50 (quantile median)** | **7.57%** | **237.80** | **419.57** |
| naive_last_week_same_hour | 10.67% | 335.00 | 544.21 |
| naive_last_day_same_hour | 14.20% | 445.77 | 798.68 |
| naive_same_hour_7d_mean | 14.22% | 446.36 | 609.94 |

P50 점예측은 가장 강력한 naive 베이스라인(`naive_last_week_same_hour`) 대비 WAPE 3.1 pt 개선됐다.

## SHAP 근거 해석

SHAP는 P50 기대값을 피처별로 분해한 것이다. 빨강 막대는 해당 피처가 예측 kWh를 위로 올린 기여분이고, 파랑은 내린 기여분이다. SHAP 합계 = P50 예측값 − global mean이 된다.

실제 사용량이 P90 밴드를 초과한 시간의 초과분(exceedance)은 모델이 day-ahead 시점에서 예상하지 못한 추가 사용량이다. 즉, SHAP로 설명되는 P50 기대치에서 밴드 끝까지의 거리(불확실성)를 더한 것보다 실제 사용이 더 많았음을 의미한다. 따라서 과다사용 탐지 알림을 받았을 때 SHAP를 보면 "무엇이 높은 예측을 만들었는가"를 알 수 있고, 초과분은 그 외 설명되지 않은 실제 증가분이다.

## 산출물

| 파일 | 설명 |
| --- | --- |
| `outputs/school_overuse_predictions.csv` | 시간별 실제/P10/P50/P90/밴드·초과 여부 |
| `outputs/school_overuse_daily_summary.csv` | 일별 집계(coverage, 초과 시간, 초과량) |
| `outputs/school_overuse_explanations.csv` | 시간별 SHAP feature contribution |
| `outputs/school_overuse_feature_importance.csv` | 글로벌 feature importance |
| `outputs/school_overuse_metrics.json` | 검증 지표 전체 |
| `outputs/school_overuse_run_summary.json` | 실행 메타데이터 및 CQR 설정 |
| `school_overuse_web/data/monitor.json` | 웹 대시보드용 직렬화 데이터(meta/metrics/baselines/series) |

## 웹

`school_overuse_web/`는 5개 탭으로 구성된 정적 대시보드이다.

| 탭 | 내용 |
| --- | --- |
| 개요 | KPI 카드(coverage, 과다사용 시간, 총 초과량, P50 WAPE) |
| 시계열 모니터 | 실제값 + P10/P50/P90 밴드, 과다/과소 시간 하이라이트 |
| 과다사용 분석 | 시간대별·요일별 과다사용 분포, 초과량 히트맵 |
| 모델 정확도 | P50 점예측 vs naive 베이스라인 비교표, Pinball loss |
| 용어 설명 | CQR, SHAP, day-ahead 계약 등 용어 해설 |

로컬에서 여는 법:

```bash
.venv\Scripts\python.exe -m http.server 8000 --directory school_overuse_web
```

이후 브라우저에서 `http://localhost:8000` 접속.

## 검증 결과

`tests/` 전체: **85 passed, 3 warnings, 61 subtests passed** (회귀 없음, 2026-06-30 기준).

## 한계

- 밴드는 **day-ahead 기대치**이며, 직전 1시간 관측값을 추가로 활용하는 operational 모델과 다르다.
- CQR coverage 보장은 분포가 안정적(exchangeable)일 때 근사적으로 성립한다. 6월처럼 전월 대비 사용 패턴이 달라지면 calibration 구간과 검증 구간의 분포 차이로 coverage가 목표(80%)를 밑돌 수 있다.
- 학교 전체 단일 시계열 기반이며, 건물별 분해 및 층별 탐지는 향후 과제이다.
