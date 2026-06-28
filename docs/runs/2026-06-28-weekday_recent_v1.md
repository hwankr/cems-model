# 2026-06-28 weekday_recent_v1 실행 기록

## 실행 개요

| 항목 | 값 |
| --- | --- |
| 실행 ID | `weekday_recent_v1` |
| 기준 비교 모델 | `baseline-v2-realtime` / `pred_realtime_uniform_daily_kwh` |
| 새 예측 컬럼 | `pred_weekday_recent_kwh` |
| 실행 스크립트 | `scripts/run_baseline_model.py` |
| 입력 파일 | `10_baseline_ready_panel_primary_reliable.csv` |
| 출력 폴더 | `outputs/` |
| 검증 구간 | 2026-04-01 ~ 2026-06-23 |
| 검증 행 수 | 1,428 |
| 건물 수 | 17 |

## 데이터 누수 방지 조건

- 2026년 4월~6월은 실시간 운영 상황으로 간주했다.
- `pred_weekday_recent_kwh`는 예측일 이전의 같은 건물 실제 사용량만 최근 사용량 feature로 사용한다.
- 예: 2026-04-10 예측에는 2026-04-09까지의 값만 들어가며, 2026-04-10 당일값과 이후 값은 feature 계산에 들어가지 않는다.
- `school_total_daily_kwh`, `school_total_day_weight_observed_month`, `baseline_school_shape_kwh_observed_norm` 등 학교 전체 실측 일별 흐름 컬럼은 예측 입력과 산출 CSV에서 제외했다.
- 기상 변수는 사용하지 않았다.

## 사용 feature

| 구분 | feature |
| --- | --- |
| 월 기준 사용량 | `profile_monthly_kwh_mean`, `calendar_days_in_month`, `pred_realtime_uniform_daily_kwh` |
| 날짜 | 요일, 주말 여부, 월, 일자 |
| 과거 요일 패턴 | 2026-04-01 이전 같은 건물/월/요일 평균 비중. 현재 체크아웃에는 2026년 3월 이전 건물별 일별 행이 없어서 모든 검증 행은 건물 월 기준값으로 fallback됨 |
| 최근 사용량 | 전일, 최근 3일 평균, 최근 7일 평균, 전주 같은 요일 사용량. 각 항목은 해당 날짜 이전 값만 사용하고 부족하면 월 기준값으로 fallback |

## 산출물

- `outputs/baseline_predictions.csv`
- `outputs/baseline_metrics_by_building.csv`
- `outputs/baseline_metrics_by_month.csv`
- `outputs/baseline_error_rankings.csv`

`baseline_predictions.csv`에는 두 예측 컬럼이 함께 들어 있다.

```text
pred_realtime_uniform_daily_kwh
pred_weekday_recent_kwh
```

## 전체 성능 비교

| 모델 | MAE kWh | RMSE kWh | MAPE | 합계 Bias |
| --- | ---: | ---: | ---: | ---: |
| 실시간 기준선: 월 프로파일 균등 일할 | 207.87 | 290.33 | 23.25% | -2.79% |
| 요일+최근사용량 모델 | 160.36 | 239.80 | 17.90% | -2.21% |

개선율:

| 지표 | 개선 |
| --- | ---: |
| MAE | 22.9% 개선 |
| RMSE | 17.4% 개선 |
| MAPE | 23.0% 개선 |

## 월별 비교

| 월 | 모델 | MAE kWh | RMSE kWh | MAPE | 합계 Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026-04 | 실시간 기준선: 월 프로파일 균등 일할 | 166.51 | 233.07 | 19.16% | -1.64% |
| 2026-04 | 요일+최근사용량 모델 | 128.18 | 181.61 | 14.30% | -1.57% |
| 2026-05 | 실시간 기준선: 월 프로파일 균등 일할 | 222.15 | 295.29 | 24.31% | -6.95% |
| 2026-05 | 요일+최근사용량 모델 | 165.23 | 236.08 | 18.76% | -3.08% |
| 2026-06 | 실시간 기준선: 월 프로파일 균등 일할 | 242.60 | 345.64 | 27.16% | 0.62% |
| 2026-06 | 요일+최근사용량 모델 | 195.78 | 303.12 | 21.43% | -1.95% |

## 검증 메모

- Python 단위 테스트에서 2026-04-10 예측 행의 최근 사용량 feature가 2026-04-10 이후 실제값 변경에 영향을 받지 않는지 확인했다.
- `baseline_predictions.csv`에서 `school_total*` 실측 학교 전체 흐름 컬럼과 `baseline_school_shape_kwh_observed_norm`이 제거된 것을 확인했다.
- `baseline_metrics_by_building.csv`, `baseline_metrics_by_month.csv`, `baseline_error_rankings.csv` 모두 두 모델을 포함한다.
