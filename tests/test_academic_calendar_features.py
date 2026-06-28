import unittest

import pandas as pd

from modeling.academic_calendar_features import (
    build_academic_daily_features,
    extract_calendar_events_from_html,
)


class AcademicCalendarFeaturesTest(unittest.TestCase):
    def test_extract_calendar_events_from_html_reads_caldata_json(self):
        html = """
        <script>
        var calData = [
          {"startDt":"2026-04-21","endDt":"2026-04-27","text":"중간시험","articleNo":1},
          {"startDt":"2026-06-22","endDt":"2026-06-22","text":"방학","articleNo":2}
        ];
        </script>
        """

        events = extract_calendar_events_from_html(html)

        self.assertEqual(len(events), 2)
        self.assertEqual(events.loc[0, "start_date"], pd.Timestamp("2026-04-21"))
        self.assertEqual(events.loc[0, "event_text"], "중간시험")
        self.assertIn("source_article_no", events.columns)

    def test_build_academic_daily_features_uses_coarse_operation_state(self):
        events = pd.DataFrame(
            [
                {"start_date": "2026-03-03", "end_date": "2026-03-03", "event_text": "개강"},
                {"start_date": "2026-04-21", "end_date": "2026-04-27", "event_text": "중간시험"},
                {"start_date": "2026-06-09", "end_date": "2026-06-09", "event_text": "공휴일수업대체지정일"},
                {"start_date": "2026-06-15", "end_date": "2026-06-19", "event_text": "기말시험"},
                {"start_date": "2026-06-22", "end_date": "2026-06-22", "event_text": "방학"},
            ]
        )

        features = build_academic_daily_features(
            events,
            start_date="2026-04-20",
            end_date="2026-06-23",
        )

        midterm = features[features["date"] == pd.Timestamp("2026-04-21")].iloc[0]
        makeup = features[features["date"] == pd.Timestamp("2026-06-09")].iloc[0]
        vacation = features[features["date"] == pd.Timestamp("2026-06-23")].iloc[0]

        self.assertEqual(midterm["academic_is_midterm"], 1)
        self.assertEqual(midterm["academic_is_instruction_period"], 1)
        self.assertEqual(makeup["academic_is_makeup_class_day"], 1)
        self.assertEqual(vacation["academic_is_vacation"], 1)
        self.assertEqual(vacation["academic_is_instruction_period"], 0)
        self.assertEqual(
            set(features.columns),
            {
                "date",
                "academic_is_instruction_period",
                "academic_is_vacation",
                "academic_is_midterm",
                "academic_is_final",
                "academic_is_makeup_class_day",
                "academic_is_course_eval",
                "academic_is_education_practice",
                "academic_days_since_semester_start",
                "academic_days_to_final",
                "academic_semester_week",
            },
        )


if __name__ == "__main__":
    unittest.main()
