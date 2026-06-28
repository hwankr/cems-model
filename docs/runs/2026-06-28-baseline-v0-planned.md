# 2026-06-28 기준 모델 v0 실행 계획

## 실행 메타데이터

| 항목 | 값 |
| --- | --- |
| 실행 ID | baseline-v0-planned |
| 날짜 | 2026-06-28 |
| 작성자 | Fabro / Codex |
| 노트북 또는 스크립트 | `cems_baseline_model_colab.ipynb` |
| 입력 파일 | `10_baseline_ready_panel_primary_reliable.csv` |
| 출력 폴더 | `outputs/` |

## 목적

기상, 일정, 머신러닝 모델을 추가하기 전에 전력 데이터만 사용한 첫 기준 모델을 만들고, 현재 데이터만으로 가능한 예측 정확도를 확인한다.

## 사용 데이터

예정 입력:

- `10_baseline_ready_panel_primary_reliable.csv`
- 참고 가능 파일: `14_baseline_metrics_primary_reliable.csv`

기본 타깃:

- `usage_kwh_clean`

기본 검증 필터:

- `is_validation_target_clean == True`

## 모델 방식

예정 방식:

- 균등 일할 기준 모델
- 학교 전체 일별 패턴 기준 모델

## 검증 설정

| 항목 | 값 |
| --- | --- |
| 학습 기준 시점 | 2026-03 |
| 검증 시작일 | 2026-04-01 |
| 검증 종료일 | 2026-06-23 |
| 제외할 부분 일자 | 2026-06-24 |
| 제외할 placeholder 날짜 | 2026-06-25부터 2026-06-30까지 |

## 결과

아직 새 실행 결과는 없다. 이 파일은 첫 실행 계획을 기록한다.

## 다음 단계

Colab 노트북을 만들고 예측값과 성능 지표 산출물을 생성한다.
