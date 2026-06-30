from pathlib import Path
from modeling.school_savings_baseline import run_savings_demo

if __name__ == "__main__":
    result = run_savings_demo()
    print("output_dir       ", result.output_dir)
    print("reporting_rows   ", result.reporting_rows)
    print("avoided_pct      ", round(result.avoided_pct, 6))
    print("avoided_sum_kwh  ", round(result.avoided_sum_kwh, 2))
    print("avoided_per_sqm  ", round(result.avoided_per_sqm_kwh, 4))
    print("heldout_wape     ", round(result.heldout_wape, 4))
    print("heldout_coverage ", round(result.heldout_coverage, 4))
    print("used_fallback    ", result.used_fallback)
