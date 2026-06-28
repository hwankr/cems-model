# CEMS baseline preprocessing output

생성 기준: 원본 압축파일의 `cems_organized/processed` 정리본을 입력으로 사용했습니다. 장기 월별 파일의 누적 검침값은 이미 `building_monthly_usage_from_cumulative_long.csv`에서 월 사용량으로 차분된 값을 사용했습니다.

## 최종 권장 사용 세트

기본 분석에는 `primary_reliable` 건물 17개만 사용하도록 정리했습니다.

건축관, 기계관, 박물관, 사범대학, 상경관, 생활과학대, 섬유관, 외국어교육원, 음대, 인문계식당, 전기관, 제1공장형실습장, 제1과학관, 제2공장형실습장, 제2인문관, 제3과학관, 화공관

주요 관심 건물이지만 원본 데이터 품질상 바로 순위/평가에 쓰기 어려운 건물은 `review_optional_major`로 분리했습니다.

IT관, 과학도서관, 정보전산원, 제1인문관심야, 중앙도서관

## 주요 판단 규칙

1. 장기 월별 데이터는 누적 검침값이므로 `monthly_usage_kwh = 이번 달 누적값 - 전월 누적값`으로 계산된 processed 파일만 사용했습니다.
2. 건물명 없는 meter row는 기준선 산정에서 제외했습니다.
3. 유효하지 않은 헤더 날짜는 날짜가 아니라 `report_month=YYYY-MM`만 사용했습니다.
4. 최근 일별 `2026-06-24`는 부분일로 보고 검증 대상에서 제외했습니다. 다만 연속 시계열 확인용으로는 건물-월 평균값을 `usage_kwh_imputed_continuous`에 채웠습니다.
5. `2026-06-25~2026-06-30`은 미관측 placeholder로 보고 `usage_kwh_clean`에서는 결측으로 유지했습니다.
6. 원본 품질 파일에 기록된 극단값, 관측 0값 의심치, 자동 고점 스파이크는 같은 건물·같은 월의 정상값 평균으로 대체했습니다.
7. 동일한 일별 시퀀스를 갖는 건물 그룹은 기본 세트에서 제외했습니다.
8. 관측 0값 비율이 10%를 초과하는 건물은 기본 세트에서 제외했습니다.

## 바로 쓰는 파일

- `10_baseline_ready_panel_primary_reliable.csv`: 기본 모델 입력/검증용 메인 패널입니다.
- `08_daily_clean_primary_reliable.csv`: 최근 일별 실제값 정제본입니다.
- `04_monthly_train_primary_reliable.csv`: 2026-03까지의 장기 월별 학습 데이터입니다.
- `06_monthly_profile_primary_reliable.csv`: 건물별 월 패턴 프로파일입니다.
- `01_building_selection_audit.csv`: 모든 최근 건물의 선정/제외 사유입니다.
- `12_imputation_log_all_selected.csv`: 평균 대체 또는 결측 처리된 행 전체 로그입니다.

## 권장 컬럼

`10_baseline_ready_panel_primary_reliable.csv`에서 우선 사용할 컬럼은 다음입니다.

- 식별: `date`, `report_month`, `building_name_recent`, `building_name_long_term`
- 실제값: `usage_kwh_clean`
- 원본 보존: `usage_kwh_raw`, `usage_kwh`
- 검증 필터: `is_validation_target_clean`, `validation_period`
- 학습 프로파일: `profile_monthly_kwh_mean`, `month_factor_vs_overall`
- 단순 기준선: `baseline_uniform_daily_kwh`
- 학교 전체 일별 shape 기준선: `baseline_school_shape_kwh_observed_norm`
- 품질 확인: `cleaning_action`, `was_imputed_in_clean`, `is_original_value_reliable`

검증 기본 기간은 `2026-04-01~2026-06-23`입니다. `2026-06-24`와 `2026-06-25~06-30`은 기본 검증에서 제외하십시오.
