"""
TDD: school_overuse_web content verification.
Checks that index.html, styles.css, app.js exist and contain required phrases
that demonstrate correct tab structure, data bindings, and key explanations.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "school_overuse_web"


class SchoolOveruseWebFilesExistTest(unittest.TestCase):
    def test_index_html_exists(self):
        self.assertTrue((WEB / "index.html").exists(), "school_overuse_web/index.html not found")

    def test_styles_css_exists(self):
        self.assertTrue((WEB / "styles.css").exists(), "school_overuse_web/styles.css not found")

    def test_app_js_exists(self):
        self.assertTrue((WEB / "app.js").exists(), "school_overuse_web/app.js not found")


class SchoolOveruseWebHtmlContentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")

    def _assert_contains(self, phrase):
        self.assertIn(phrase, self.html, f"HTML missing: {phrase!r}")

    # Tab labels
    def test_tab_overview(self):
        self._assert_contains("개요")

    def test_tab_timeseries(self):
        self._assert_contains("시계열")

    def test_tab_overuse_analysis(self):
        self._assert_contains("과다사용 분석")

    def test_tab_model_accuracy(self):
        self._assert_contains("모델 정확도")

    def test_tab_glossary(self):
        self._assert_contains("용어 설명")

    # Core story keywords
    def test_phrase_overuse(self):
        self._assert_contains("과다사용")

    def test_phrase_normal_range(self):
        self._assert_contains("정상 범위")

    def test_phrase_band(self):
        # Band/정상밴드concept
        self._assert_contains("정상밴드")

    def test_phrase_p10_p90(self):
        self._assert_contains("P10")
        self._assert_contains("P90")

    def test_phrase_shap(self):
        self._assert_contains("SHAP")

    def test_phrase_day_ahead(self):
        self._assert_contains("day-ahead")

    def test_phrase_cqr_calibration(self):
        self._assert_contains("CQR")

    def test_phrase_cdd(self):
        # CDD appears in glossary and/or explanation
        self._assert_contains("CDD")

    def test_phrase_wape(self):
        self._assert_contains("WAPE")

    def test_phrase_quantile(self):
        self._assert_contains("분위수")


class SchoolOveruseWebAppJsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = (WEB / "app.js").read_text(encoding="utf-8")

    def _assert_contains(self, phrase):
        self.assertIn(phrase, self.js, f"app.js missing: {phrase!r}")

    def test_data_path(self):
        self._assert_contains("data/monitor.json")

    def test_references_series(self):
        self._assert_contains("series")

    def test_references_overuse(self):
        self._assert_contains("overuse")

    def test_references_feature_importance(self):
        self._assert_contains("feature_importance")

    def test_references_glossary(self):
        self._assert_contains("glossary")

    def test_references_p10(self):
        self._assert_contains("p10")

    def test_references_p90(self):
        self._assert_contains("p90")

    def test_references_exceedance(self):
        self._assert_contains("exceedance")

    def test_fetch_call(self):
        self._assert_contains("fetch(")


class SchoolOveruseWebGlossaryTermsTest(unittest.TestCase):
    """Key glossary terms must appear in at least one of html or js."""

    @classmethod
    def setUpClass(cls):
        cls.combined = (
            (WEB / "index.html").read_text(encoding="utf-8")
            + (WEB / "app.js").read_text(encoding="utf-8")
        )

    def _assert_contains(self, phrase):
        self.assertIn(phrase, self.combined, f"Combined html+js missing: {phrase!r}")

    def test_term_quantile_ko(self):
        self._assert_contains("분위수")

    def test_term_shap(self):
        self._assert_contains("SHAP")

    def test_term_wape(self):
        self._assert_contains("WAPE")

    def test_term_day_ahead(self):
        self._assert_contains("day-ahead")

    def test_term_cdd(self):
        self._assert_contains("CDD")


if __name__ == "__main__":
    unittest.main()
