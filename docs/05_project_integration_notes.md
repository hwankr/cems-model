# 프로젝트 적용 메모

## 핵심 정리

실제 CEMS 프로젝트에서는 Colab 노트북을 운영 환경에서 직접 실행하지 않는다.

노트북은 다음 용도로 사용한다.

- 실험
- 그래프 생성
- 보고서용 성능표 작성
- 어떤 모델링 방식이 유효한지 확인

실제 프로젝트에서는 다음 중 하나를 사용한다.

- 노트북에서 생성한 예측 CSV 파일
- Python 예측 스크립트
- 예측 로직을 실행하는 백엔드 API
- 예약 작업으로 채워지는 데이터베이스 테이블

## 권장 적용 흐름

### 1단계: 노트북 결과 내보내기

`cems_baseline_model_colab.ipynb`를 실행해 다음 파일을 생성한다.

- `baseline_predictions.csv`
- `baseline_metrics_by_building.csv`
- `baseline_metrics_by_month.csv`

웹 프로젝트는 이 파일을 직접 읽거나 Supabase에 적재해 사용할 수 있다.

### 2단계: 스크립트로 분리

노트북에서 안정화된 로직을 Python 파일로 옮긴다.

```text
modeling/
  baseline_model.py
  train_baseline.py
  predict_baseline.py
```

각 파일의 역할:

- `baseline_model.py`: 예측 함수와 성능 지표 계산 함수
- `train_baseline.py`: 모델 파라미터 준비와 설정 저장
- `predict_baseline.py`: 새 입력 데이터에서 예측 CSV 생성

### 3단계: 백엔드 또는 배치 작업 연동

백엔드 라우트, 예약 작업, 관리자용 수동 실행 프로세스에서 예측 스크립트를 사용한다.

```text
새 전력 데이터
  -> predict_baseline.py
  -> 예측 테이블 또는 CSV
  -> CEMS 대시보드
```

## 권장 데이터베이스 형태

최종 예측 행은 다음 형태가 적합하다.

| 컬럼 | 의미 |
| --- | --- |
| `date` | 예측 날짜 |
| `building_name` | 건물명 |
| `model_version` | `baseline-v1` 같은 모델 버전 |
| `predicted_kwh` | 예측 일일 전력 사용량 |
| `actual_kwh` | 실제값이 있는 경우의 실제 전력 사용량 |
| `absolute_error_kwh` | 실제값이 있는 경우의 절대오차 |
| `created_at` | 예측값 생성 시각 |

## 모델 버전 관리

단순한 버전명을 사용한다.

- `baseline-v1`: 전력 데이터만 사용한 기준 모델
- `weather-v1`: 전력 데이터와 기상 데이터 결합 모델
- `calendar-v1`: 전력, 기상, 일정 데이터 결합 모델
- `lightgbm-v1`: 첫 LightGBM 모델
