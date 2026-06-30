from pathlib import Path
from modeling.school_overuse_model import run_school_overuse_model

if __name__ == "__main__":
    result = run_school_overuse_model()
    print("output_dir", result.output_dir)
    print("validation_rows", result.validation_rows)
    print("coverage", round(result.coverage, 4))
    print("overuse_hours", result.overuse_hours)
    print("p50_wape", round(result.p50_wape, 4))
    print("shap_available", result.shap_available)
