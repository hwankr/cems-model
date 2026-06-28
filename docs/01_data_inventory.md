# 데이터 인벤토리

## 원본 폴더

`C:\Users\fabro\Desktop\cems-model\cems_preprocessed_baseline`

이 폴더는 git 저장소가 아니다. 전처리 결과 CSV 파일을 모아 둔 데이터 작업 공간이다.

## 주요 파일

| 파일 | 행 수 | 역할 |
| --- | ---: | --- |
| `10_baseline_ready_panel_primary_reliable.csv` | 1547 | 신뢰 가능한 17개 건물의 기준 모델용 메인 패널 |
| `04_monthly_train_primary_reliable.csv` | 442 | 학습 기준 시점까지의 건물별 월별 사용량 |
| `06_monthly_profile_primary_reliable.csv` | 204 | 건물별 월별 사용량 프로필 요약 |
| `08_daily_clean_primary_reliable.csv` | 1547 | 최근 건물별 일별 전력 사용량 정제본 |
| `13_school_total_daily_clean.csv` | 365 | 학교 전체 일별 전력 사용 패턴 |
| `14_baseline_metrics_primary_reliable.csv` | 136 | 기존 기준 모델 성능 참고표 |
| `15_preprocessing_summary.csv` | 10 | 전처리 요약과 선택된 건물 수 |
| `preprocessing_rules.json` | 1 | 전처리 규칙을 기록한 JSON 파일 |

## 메인 모델링 패널

기준 모델의 기본 입력은 `10_baseline_ready_panel_primary_reliable.csv`다.

중요 컬럼:

| 컬럼 | 의미 |
| --- | --- |
| `date` | 일자 |
| `report_month` | `2026-04` 같은 월 단위 라벨 |
| `building_name_recent` | 최근 일별 데이터에서 사용하는 건물명 |
| `building_name_long_term` | 장기 월별 데이터에서 사용하는 건물명 |
| `usage_kwh_clean` | 검증 타깃으로 사용하는 정제된 실제 일일 사용량 |
| `usage_kwh_imputed_continuous` | 연속성 확보를 위해 대체한 사용량. 기본 검증 타깃은 아님 |
| `is_validation_target_clean` | 정제 기준에서 검증에 사용할 수 있는 행인지 여부 |
| `validation_period` | 검증 행의 구분값 |
| `profile_monthly_kwh_mean` | 해당 건물과 해당 월의 과거 월별 평균 사용량 |
| `calendar_days_in_month` | 해당 월의 일수 |
| `baseline_uniform_daily_kwh` | 기존 균등 일할 기준 모델 예측값 |
| `school_total_daily_kwh` | 같은 날짜의 학교 전체 일별 사용량 |
| `school_total_day_weight_observed_month` | 학교 전체 사용량 기준의 월내 일별 비중 |
| `baseline_school_shape_kwh_observed_norm` | 기존 학교 전체 일별 패턴 기반 기준 모델 예측값 |

## 현재 데이터 형태

| 항목 | 값 |
| --- | ---: |
| 건물 수 | 17 |
| 전체 행 수 | 1547 |
| 날짜 범위 | 2026-04-01부터 2026-06-30까지 |
| 검증 행 수 | 1428 |
| `usage_kwh_clean` 결측 행 수 | 102 |
| placeholder 행 수 | 102 |

## 검증 구간

기본 검증 구간:

`2026-04-01`부터 `2026-06-23`까지

검증에서 제외하는 구간:

- `2026-06-24`: 부분 일자이므로 연속성 확인용으로만 사용
- `2026-06-25`부터 `2026-06-30`까지: 실제값이 없는 placeholder 날짜
