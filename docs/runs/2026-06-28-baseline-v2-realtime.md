# 2026-06-28 baseline-v2-realtime 실행 기록

## 수정 배경

`baseline-v1`의 `학교 전체 일별 패턴` 방식은 2026년 4월 이후의 실제 학교 전체 일별 전력 흐름을 사용했다. 이 값은 검증 대상 기간에 이미 발생한 관측값이므로, "2026년 4월부터는 실시간 예측 상황"이라는 조건에서는 입력으로 사용할 수 없다.

따라서 `baseline-v2-realtime`에서는 2026년 3월 이전에 만들 수 있는 건물별 월 프로파일만 사용한다.

## 실행 정보

| 항목 | 값 |
| --- | --- |
| 실행 ID | baseline-v2-realtime |
| 모델 | 실시간 기준선: 월 프로파일 균등 일할 |
| 실행 파일 | `scripts/run_baseline_model.py` |
| 입력 파일 | `10_baseline_ready_panel_primary_reliable.csv` |
| 출력 폴더 | `outputs/` |

## 실시간 입력 조건

사용:

- 건물명
- 날짜와 월
- `profile_monthly_kwh_mean`
- `calendar_days_in_month`
- 2026년 3월 이전 월별 전력 프로파일 출처

예측 입력에서 제외:

- `school_total_daily_kwh`
- `school_total_day_weight_observed_month`
- `baseline_school_shape_kwh_observed_norm`
- 2026년 4월 이후 실제 학교 전체 일별 흐름

## 계산식

```text
건물 일별 예측값 = 건물 월 기준 사용량 / 해당 월 날짜 수
```

출력 컬럼:

```text
pred_realtime_uniform_daily_kwh
```

## 데이터 확인

| 항목 | 값 |
| --- | ---: |
| 전체 행 수 | 1,547 |
| 건물 수 | 17 |
| 검증 행 수 | 1,428 |
| 검증 시작일 | 2026-04-01 |
| 검증 종료일 | 2026-06-23 |

검증 행은 `is_validation_target_clean == True`인 행만 사용했다. 2026-06-24 부분 일자와 2026-06-25부터 2026-06-30까지의 placeholder 일자는 평가에서 제외된다.

## 주요 결과

| 월 | MAE kWh | RMSE kWh | MAPE | 합계 bias |
| --- | ---: | ---: | ---: | ---: |
| 2026-04 | 166.51 | 233.07 | 0.1916 | -0.0164 |
| 2026-05 | 222.15 | 295.29 | 0.2431 | -0.0695 |
| 2026-06 | 242.60 | 345.64 | 0.2716 | 0.0062 |

이 결과가 현재의 정직한 실시간 기준선이다. 이후 기상 예보, 요일, 최근 전력값을 추가한 모델은 이 기준선을 이겨야 한다.
