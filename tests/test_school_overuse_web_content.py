"""
TDD: school_overuse_web content verification.

Phrase-presence tests (SchoolOveruseWebHtml/AppJs/GlossaryTermsTest) are STATIC
checks against the source text of index.html and app.js only.  Runtime SVG
rendering and interactive click behaviour are verified manually; headless-browser
integration is deferred.

The MonitorJsonDataContractTest and MonitorJsonNanSafetyTest classes validate
the generated school_overuse_web/data/monitor.json file for schema completeness
and numeric safety.
"""
from pathlib import Path
import json
import math
import unittest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "school_overuse_web"
MONITOR_JSON = WEB / "data" / "monitor.json"


class MonitorJsonDataContractTest(unittest.TestCase):
    """Validate that monitor.json contains required top-level keys and row schemas."""

    @classmethod
    def setUpClass(cls):
        if not MONITOR_JSON.exists():
            raise unittest.SkipTest(f"monitor.json not found: {MONITOR_JSON}")
        cls.bundle = json.loads(MONITOR_JSON.read_text(encoding="utf-8"))

    def test_top_level_keys(self):
        required = {"meta", "metrics", "baselines", "series", "overuse", "feature_importance", "glossary"}
        missing = required - set(self.bundle.keys())
        self.assertFalse(missing, f"monitor.json missing top-level keys: {missing}")

    def test_series_row_schema(self):
        series = self.bundle["series"]
        self.assertGreater(len(series), 0, "series must not be empty")
        required = {"timestamp", "report_date", "hour", "actual", "p10", "p50", "p90",
                    "in_band", "is_overuse", "exceedance_kwh", "exceedance_pct", "band_position"}
        missing = required - set(series[0].keys())
        self.assertFalse(missing, f"series[0] missing keys: {missing}")

    def test_overuse_explanation_schema(self):
        overuse = self.bundle["overuse"]
        self.assertGreater(len(overuse), 0, "overuse list must not be empty")
        first = overuse[0]
        self.assertIn("explanations", first, "overuse[0] must have 'explanations' key")
        explanations = first["explanations"]
        self.assertGreater(len(explanations), 0, "overuse[0] explanations must not be empty")
        required = {"feature", "label_ko", "shap_kwh", "feature_value", "direction"}
        missing = required - set(explanations[0].keys())
        self.assertFalse(missing, f"overuse[0].explanations[0] missing keys: {missing}")

    def test_feature_importance_schema(self):
        fi = self.bundle["feature_importance"]
        self.assertGreater(len(fi), 0, "feature_importance must not be empty")
        required = {"feature", "label_ko", "mean_abs_shap"}
        missing = required - set(fi[0].keys())
        self.assertFalse(missing, f"feature_importance[0] missing keys: {missing}")

    def test_baselines_schema(self):
        baselines = self.bundle["baselines"]
        self.assertGreater(len(baselines), 0, "baselines must not be empty")
        required = {"model", "wape", "mae_kwh", "rmse_kwh"}
        missing = required - set(baselines[0].keys())
        self.assertFalse(missing, f"baselines[0] missing keys: {missing}")


class MonitorJsonNanSafetyTest(unittest.TestCase):
    """Validate that monitor.json contains no NaN/Inf values in numeric fields."""

    @classmethod
    def setUpClass(cls):
        if not MONITOR_JSON.exists():
            raise unittest.SkipTest(f"monitor.json not found: {MONITOR_JSON}")
        cls.bundle = json.loads(MONITOR_JSON.read_text(encoding="utf-8"))

    def test_series_band_values_finite(self):
        for i, row in enumerate(self.bundle["series"]):
            for field in ("p10", "p50", "p90"):
                val = row[field]
                self.assertIsNotNone(val, f"series[{i}].{field} is None")
                self.assertTrue(math.isfinite(val), f"series[{i}].{field}={val} is not finite")
            actual = row.get("actual")
            if actual is not None:
                self.assertTrue(math.isfinite(actual), f"series[{i}].actual={actual} is not finite")

    def test_series_band_width_positive(self):
        """p90 - p10 > 0 for every row so SVG yScale/band-width denominator is never zero."""
        for i, row in enumerate(self.bundle["series"]):
            width = row["p90"] - row["p10"]
            self.assertGreater(width, 0, f"series[{i}] has zero-width band: p10={row['p10']} p90={row['p90']}")

    def test_metrics_numeric_values_finite(self):
        metrics = self.bundle["metrics"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.assertTrue(math.isfinite(v), f"metrics.{k}={v} is not finite")


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

    def test_feature_importance_container(self):
        self._assert_contains("feature-importance-list")


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

    def test_references_mean_abs_shap(self):
        self._assert_contains("mean_abs_shap")

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
