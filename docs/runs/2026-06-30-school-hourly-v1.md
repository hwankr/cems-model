# 2026-06-30 school-hourly-v1 실행 기록

## 목적

기존 17개 건물별 일별 비교 모델과 분리해서, 학교 전체 전력 사용량을 1시간 단위로 예측하는 별도 모델을 만들었다. 입력 데이터는 `school_power_usage_split/ml_ready/power_usage_1hour_ml.csv`이고, target은 `usage_kwh`이다.

## 검증 기준

- 학습 구간: `2025-07-01 00:00:00`부터 `2026-05-31 23:00:00`까지
- 공식 검증 구간: `2026-06-01 00:00:00`부터 `2026-06-29 23:00:00`까지
- 검증 행: 696시간, 29일
- 제외: `2026-06-30`은 원본 hourly 데이터가 10시까지만 있어서 공식 점수에서 제외

이번 모델은 운영형 hourly forecast로 정의했다. 즉 각 timestamp를 예측할 때 그 시각 이전에 관측된 값은 쓸 수 있지만, 같은 timestamp의 실제 `usage_kwh`는 쓰지 않는다. 모든 lag/rolling feature는 `shift` 이후 계산한다.

## 모델 후보

| 모델 | 설명 |
| --- | --- |
| `naive_last_day_same_hour` | 24시간 전 같은 시간 사용량 |
| `naive_last_week_same_hour` | 168시간 전 같은 요일/시간 사용량 |
| `naive_same_hour_7d_mean` | 과거 7일 같은 시간 사용량 평균 |
| `lightgbm_school_hourly_day_ahead` | 날짜/요일/시간 주기 feature와 24시간 전, 168시간 전, 같은 시간 7일 평균만 쓰는 LightGBM 회귀 모델 |
| `lightgbm_school_hourly_day_ahead_weather` | 전기 과거 패턴에 일별 예보 날씨 feature를 추가한 모델 |
| `lightgbm_school_hourly_day_ahead_weather_academic` | 전기 과거 패턴, 일별 예보 날씨, 학사일정 feature를 함께 쓰는 모델 |
| `lightgbm_school_hourly_operational` | 직전 1시간 관측값과 rolling feature까지 쓰는 운영형 1시간 갱신 모델 |

## 결과

| 모델 | WAPE | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: |
| `lightgbm_school_hourly_operational` | 2.44% | 72.47 kWh | 104.45 kWh | +0.14% |
| `lightgbm_school_hourly_day_ahead_weather_academic` | 6.73% | 200.23 kWh | 347.38 kWh | -1.07% |
| `lightgbm_school_hourly_day_ahead_weather` | 7.26% | 215.95 kWh | 367.97 kWh | -1.84% |
| `lightgbm_school_hourly_day_ahead` | 7.36% | 219.02 kWh | 349.39 kWh | +0.03% |
| `naive_last_week_same_hour` | 12.94% | 384.94 kWh | 591.36 kWh | +1.76% |
| `naive_last_day_same_hour` | 13.24% | 393.71 kWh | 721.69 kWh | -0.67% |
| `naive_same_hour_7d_mean` | 14.22% | 423.08 kWh | 588.25 kWh | +0.86% |

공식 champion은 `lightgbm_school_hourly_day_ahead_weather_academic`이다. 2.44%는 직전 관측값까지 쓰는 운영형 1시간 갱신 모델의 점수라서, 하루 전 예측 성능으로 해석하면 안 된다.

## 생성 산출물

- `outputs/school_hourly_predictions.csv`
- `outputs/school_hourly_model_comparison.csv`
- `outputs/school_hourly_metrics_by_hour.csv`
- `outputs/school_hourly_metrics_by_day.csv`
- `outputs/school_hourly_top_errors.csv`
- `outputs/school_hourly_run_summary.json`
- `school_hourly_web/index.html`

## 웹 시각화

`school_hourly_web/`는 기존 `web/`과 분리된 별도 정적 웹이다. 모델 선택, 날짜 선택, 시간 선택이 가능하고 다음을 보여준다.

- 전체/필터 기준 KPI: 검증 행, 실제 합계, 예측 합계, MAE, RMSE, WAPE, Bias
- 시간별 실제값/예측값 선 그래프
- 모델별 비교표
- 시간대별 성능
- 일별 성능
- 오차 상위 timestamp

## 해석

학교 전체 hourly 데이터는 건물별 일별 모델보다 예측 단위가 명확하고 데이터 행 수가 많다. 1시간 전/하루 전/일주일 전의 부하 패턴이 강해서, 운영형 단기 예측 문제로 두면 모델 품질이 높게 나온다.

2.44% 결과는 "직전 관측값까지 사용할 수 있는 운영형 예측" 기준이다. 하루 전 예측 기준에서는 같은 검증 기간의 전기-only WAPE가 7.36%이고, 예보 날씨와 학사일정을 붙이면 6.73%로 내려간다. 한 달 전체를 미리 예측하는 장기 forecast로 바꾸면 lag 사용 가능 범위가 더 줄어드므로 별도 검증이 필요하다.
