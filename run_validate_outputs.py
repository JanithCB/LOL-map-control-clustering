"""
Top-level script that validates project outputs and saves a summary.

PowerShell execution command:
python run_validate_outputs.py
"""

import json
import logging
from pathlib import Path
from src.data.validate_outputs import validate_pipeline_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

def main():
    LOGGER.info("Starting validation of pipeline outputs...")
    
    report = validate_pipeline_outputs(output_root=".")
    
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = reports_dir / "validation_report.json"
    md_path = reports_dir / "validation_report.md"
    
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    with open(md_path, "w") as f:
        f.write("# Validation Report\n\n")
        f.write(f"**Overall Status:** {report['status'].upper()}\n\n")
        f.write("## Checks\n")
        for check in report["checks"]:
            f.write(f"- **{check['name']}**: {check['status'].upper()}\n")
            if check['details']:
                f.write(f"  - Details: {check['details']}\n")
                
    LOGGER.info(f"Validation finished with status: {report['status'].upper()}")
    LOGGER.info(f"Report saved to {json_path} and {md_path}")
    
    # Print summary to console
    print("\n" + "="*40)
    print("           VALIDATION SUMMARY")
    print("="*40)
    print(f"OVERALL STATUS: {report['status'].upper()}")
    print("-" * 40)
    for check in report['checks']:
        print(f"[{check['status'].upper()}] {check['name']}")
        if check['details']:
            print(f"    -> {check['details']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
