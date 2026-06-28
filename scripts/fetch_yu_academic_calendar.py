from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.academic_calendar_features import save_academic_calendar_outputs


def main() -> None:
    years = [2024, 2025, 2026]
    events, daily = save_academic_calendar_outputs(
        years=years,
        event_output_path=Path("outputs/yu_academic_calendar_events.csv"),
        daily_output_path=Path("outputs/yu_academic_calendar_daily_features.csv"),
        start_date="2024-01-01",
        end_date="2026-12-31",
    )
    print(f"years={','.join(map(str, years))}")
    print(f"events={len(events)}")
    print(f"daily_rows={len(daily)}")


if __name__ == "__main__":
    main()
