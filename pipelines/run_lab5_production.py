import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: Production Data Pipeline")
    print("[INFO] World Cup Run Prediction")
    print("[INFO] =========================================")

    scripts = [
        PROJECT_ROOT / "src" / "preprocess.py",
        PROJECT_ROOT / "src" / "train.py",
        PROJECT_ROOT / "src" / "evaluate.py",
        PROJECT_ROOT / "src" / "validate_reproducibility.py"
    ]

    for script in scripts:

        print(f"\n[INFO] ---> Executing {script.relative_to(PROJECT_ROOT)}...")

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print(
                f"[ERROR] Pipeline halted. "
                f"{script.name} encountered an error."
            )
            sys.exit(1)

        print(f"[SUCCESS] {script.name} completed successfully.")

    print("\n[INFO] =========================================")
    print("[SUCCESS] Lab 5 Production Pipeline fully executed!")
    print("[INFO] =========================================")


if __name__ == "__main__":
    execute_pipeline()