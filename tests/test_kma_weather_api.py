import tempfile
import unittest
from pathlib import Path

from modeling.kma_weather_api import kma_grid_from_lonlat, load_env_file
from scripts.fetch_kma_weather import (
    build_actual_weather_input,
    extract_grid_value,
    parse_apihub_table,
)


class KmaWeatherApiTest(unittest.TestCase):
    def test_load_env_file_strips_optional_quotes_without_exposing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'KMA_APIHUB_AUTH_KEY="abc123=="\nOTHER=value\n',
                encoding="utf-8",
            )

            values = load_env_file(env_path)

            self.assertEqual(values["KMA_APIHUB_AUTH_KEY"], "abc123==")
            self.assertEqual(values["OTHER"], "value")

    def test_kma_grid_from_lonlat_returns_stable_yeungnam_campus_grid(self):
        grid = kma_grid_from_lonlat(lon=128.7546, lat=35.8338)

        self.assertEqual(grid.x, 92)
        self.assertEqual(grid.y, 90)

    def test_asos_daily_fixed_width_response_maps_weather_columns(self):
        text = "\n".join(
            [
                "#START7777",
                "# YYMMDD STN WS WR WD WS WS WD WS WS TA TA TA TA TA TD TS TG HM HM HM PV EV_S EV_L FG PA PS PS PS PS PS CA SS SS SS SI SI SI RN",
                "20260331 143 2.6 2260 9 5.1 1925 9 7.9 1922 12.9 15.8 1326 10.4 15 8.1 13.3 10.1 74.1 56.0 1151 10.8 2.4 1.7 -9.00 1002.9 1009.4 1014.7 2033 1002.4 356 9.4 0.4 12.5 -9.0 5.95 1.29 900 17.3 -9.0",
                "#7777END",
            ],
        )

        raw = parse_apihub_table(text)
        actual_input = build_actual_weather_input(raw)

        self.assertEqual(actual_input.iloc[0]["tm"], "20260331")
        self.assertEqual(actual_input.iloc[0]["avgTa"], "12.9")
        self.assertEqual(actual_input.iloc[0]["maxTa"], "15.8")
        self.assertEqual(actual_input.iloc[0]["minTa"], "10.4")
        self.assertEqual(actual_input.iloc[0]["avgRhm"], "74.1")
        self.assertEqual(actual_input.iloc[0]["sumRn"], "17.3")
        self.assertEqual(actual_input.iloc[0]["avgWs"], "2.6")

    def test_extract_grid_value_uses_kma_row_major_grid_order(self):
        values = [-99.0] * (149 * 253)
        values[(90 - 1) * 149 + (92 - 1)] = 10.0
        text = ",".join(str(value) for value in values)

        self.assertEqual(extract_grid_value(text, x=92, y=90), 10.0)


if __name__ == "__main__":
    unittest.main()
