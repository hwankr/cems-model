# CEMS 기준 모델 설계

## 목표

전력 사용량 데이터만 사용해 CEMS 건물별 일일 전력 사용량 예측을 위한 첫 기준 모델을 만든다.

## 배경

현재 폴더에는 전처리된 CSV 파일이 들어 있다. 이 폴더는 git 저장소가 아니다. 메인 모델링 파일은 `10_baseline_ready_panel_primary_reliable.csv`이며, `primary_reliable` 건물 17개와 `2026-04-01`부터 `2026-06-30`까지의 일별 행을 포함한다.

## 범위

나중에 프로젝트 스크립트로 옮길 수 있는 Colab 우선 기준 모델 노트북을 만든다. 첫 모델에서는 기상, 학사일정, LightGBM을 제외한다.

## 입력

- `10_baseline_ready_panel_primary_reliable.csv`
- 참고 가능 파일: `14_baseline_metrics_primary_reliable.csv`

## 타깃과 검증 분리

- 타깃: `usage_kwh_clean`
- 학습 기준 개념: `2026-03`까지의 월별 프로필 사용
- 검증: `is_validation_target_clean == True`, 즉 `2026-04-01`부터 `2026-06-23`까지
- 제외: `2026-06-24` 부분 일자, `2026-06-25`부터 `2026-06-30`까지의 placeholder 날짜

## 기준 모델 방식

1. 균등 일할 기준 모델: 건물별 월별 프로필을 해당 월의 일수로 나눈다.
2. 학교 전체 일별 패턴 기준 모델: 건물별 월별 프로필에 학교 전체 월내 일별 비중을 곱한다.

## 산출물

- 노트북: `cems_baseline_model_colab.ipynb`
- 예측값: `outputs/baseline_predictions.csv`
- 건물별 성능표: `outputs/baseline_metrics_by_building.csv`
- 월별 성능표: `outputs/baseline_metrics_by_month.csv`
- 오차 순위표: `outputs/baseline_error_rankings.csv`
- 문서: `docs/` 아래의 추적 문서

## 프로젝트 적용 방향

노트북은 실험과 보고서 작성을 위한 도구다. 실제 프로젝트는 내보낸 예측 파일, Python 스크립트, 또는 백엔드에서 생성한 예측 행을 사용해야 한다. 운영 환경에서 노트북을 직접 실행하지 않는다.

## 승인 기준

- 문서 구조가 존재하고 모델링 흐름을 설명한다.
- 첫 실행 계획이 기록되어 있다.
- 이후 만들 노트북의 로드, 예측, 평가, 분석, 저장 섹션이 명확하다.
- 구현 전에 출력 파일명과 검증 규칙이 문서화되어 있다.
