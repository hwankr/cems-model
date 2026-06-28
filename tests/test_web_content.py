from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebContentTest(unittest.TestCase):
    def test_model_explanation_is_self_contained(self):
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

        required_phrases = [
            "모델 작동 방식",
            "월 프로파일 균등 일할",
            "요일+최근사용량 모델",
            "2026-04-10 예측에는 2026-04-09까지",
            "당일 실제값과 이후 값은 사용하지 않습니다",
            "전일 사용량",
            "최근 3일 평균",
            "최근 7일 평균",
            "전주 같은 요일",
            "초반 날짜처럼 최근값이 부족하면 월 기준값으로 대신 채웁니다",
            "학교 전체 일별 실측 흐름",
            "평일 가중치",
            "주말 가중치",
            "공정성 진단",
            "공정성 baseline 판정",
            "아직 신뢰 불가",
            "MAPE는 주력 지표로 쓰지 않습니다",
            "90% 예측구간",
            "calibration",
            "잔차 무편향성",
            "naive baseline 대비",
            "비교용 머신러닝 모델",
            "LightGBM q50 walk-forward",
            "기존 규칙 기반 baseline보다 지표가 얼마나 나아지는지",
        ]

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, html)


if __name__ == "__main__":
    unittest.main()
