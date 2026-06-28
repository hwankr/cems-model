# CEMS 기준 모델 문서

이 폴더는 1차 기준 모델 작업을 데이터 확인부터 실제 프로젝트 적용 준비까지 기록하기 위한 문서 공간입니다.

## 현재 목표

전력 사용량 데이터만 사용해 첫 번째 기준 모델을 만든다. 이 모델은 다음 정보를 바탕으로 건물별 일일 전력 사용량을 추정한다.

- 학교 전체 일별 전력 사용 패턴
- 건물별 2년치 월별 전력 사용량
- 최근 3개월 건물별 일별 전력 사용량은 검증에만 사용

1차 모델에서는 의도적으로 기상 데이터와 학사일정 데이터를 제외한다. 해당 변수들은 이후 2차 모델에서 추가해 성능 차이를 비교한다.

## 작업 흐름

1. 원본 데이터와 사용 가능한 컬럼 확인: [01_data_inventory.md](01_data_inventory.md)
2. 기준 모델과 검증 구간 정의: [02_baseline_model_design.md](02_baseline_model_design.md)
3. 노트북 또는 스크립트 실행 기록: [03_experiment_log.md](03_experiment_log.md)
4. 모델 성능과 오차 패턴 평가: [04_evaluation_checklist.md](04_evaluation_checklist.md)
5. 노트북 결과를 실제 프로젝트로 옮기는 방법 정리: [05_project_integration_notes.md](05_project_integration_notes.md)

## 예상 산출물

- `cems_baseline_model_colab.ipynb`: 데이터 확인, 성능표, 그래프를 포함한 Colab용 기준 모델 노트북
- `outputs/baseline_predictions.csv`: 검증 구간과 placeholder 구간의 행 단위 예측값
- `outputs/baseline_metrics_by_building.csv`: 건물별 오차 지표
- `outputs/baseline_metrics_by_month.csv`: 월별 오차 지표
- `outputs/baseline_error_rankings.csv`: 오차가 큰 건물과 날짜 목록

## 기록 원칙

보고서나 프로젝트 코드에 결과를 사용하기 전에 의미 있는 실행 결과는 반드시 기록한다.

새 실험을 실행할 때는 [templates/experiment_entry.md](templates/experiment_entry.md)를 복사해 `docs/runs/` 아래에 저장한다.
