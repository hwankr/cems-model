from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebContentTest(unittest.TestCase):
    def test_model_explanation_is_self_contained(self):
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

        required_phrases = [
            "모델 작동 방식",
            "실시간 기준 월 프로파일 균등 일할",
            "요일+최근사용량 모델",
            "LightGBM q50 walk-forward",
            "LightGBM + weather + area-v2",
            "2026-04-10 예측에는 2026-04-09까지",
            "당일 실제값과 이후 값은 사용하지 않습니다",
            "전일 사용량",
            "최근 3일 평균",
            "최근 7일 평균",
            "전주 같은 요일",
            "KMA APIHub 단기예보",
            "17개 건물 1,190행",
            "131.46 kWh",
            "11.82%",
            "냉방도일, 난방도일, 강수량, 강수 여부, 건물 면적",
            "학사일정 후보",
            "성능이 악화되면 champion에서 자동 제외합니다",
            "공정성 baseline 판정",
            "아직 보류",
            "90% 예측구간",
            "calibration",
            "오차 무편향성",
        ]

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, html)

    def test_dashboard_loads_champion_model(self):
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        required_phrases = [
            'champion: "../outputs/ml_baseline_champion.json"',
            "loadOptionalJson(files.champion)",
            "state.champion?.prediction_column",
            "pred_lightgbm_weather_area_q50_kwh",
            "LightGBM + weather + area 모델",
            "pred_lightgbm_weather_academic_q50_kwh",
            "pred_lightgbm_weather_area_academic_q50_kwh",
        ]

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, script)

    def test_school_hourly_web_is_separate_and_self_contained(self):
        html = (ROOT / "school_hourly_web" / "index.html").read_text(encoding="utf-8")

        required_phrases = [
            "학교 전체 시간 단위 예측",
            "기존 17개 건물 비교 모델과 분리된 별도 웹",
            "2026-06-01 00:00 ~ 2026-06-29 23:00",
            "2026-06-30 제외",
            "day-ahead + operational 비교",
            "직전 관측값까지 사용",
            "LightGBM day-ahead + weather + academic",
            "2.44% 숫자의 의미",
            "공식 day-ahead WAPE는 6.73%",
            "예보 날씨와 학사일정",
            "last day same hour",
            "last week same hour",
            "시간대별 성능",
            "오차 상위 시간",
            "MAE",
            "RMSE",
            "WAPE",
            "Bias",
        ]

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, html)

    def test_school_hourly_web_loads_generated_outputs(self):
        script = (ROOT / "school_hourly_web" / "app.js").read_text(encoding="utf-8")

        required_phrases = [
            "../outputs/school_hourly_predictions.csv",
            "../outputs/school_hourly_model_comparison.csv",
            "../outputs/school_hourly_metrics_by_hour.csv",
            "../outputs/school_hourly_metrics_by_day.csv",
            "../outputs/school_hourly_top_errors.csv",
            "../outputs/school_hourly_run_summary.json",
            "pred_lightgbm_school_hourly_day_ahead_kwh",
            "pred_lightgbm_school_hourly_day_ahead_weather_kwh",
            "pred_lightgbm_school_hourly_day_ahead_weather_academic_kwh",
            "pred_lightgbm_school_hourly_operational_kwh",
            "renderHourlyChart",
            "renderModelComparison",
            "renderTopErrors",
        ]

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, script)


if __name__ == "__main__":
    unittest.main()
