from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

REQUIRED_OUTPUTS = [
    "model_comparison.csv",
    "sample_prediction.csv",
    "maecomp.png",
    "msecomp.png",
    "rmsecomp.png",
    "r2comp.png"
]

REQUIRED_MODELS = [
    "linear_regression.pkl",
    "decision_tree.pkl",
    "random_forest.pkl",
    "gradient_boosting.pkl",
    "extra_trees.pkl"
]

def validate_outputs():
    print("=" * 60)
    print("OUTPUT VALIDATION")
    print("World Cup Run Prediction")
    print("=" * 60)

    print("\n[INFO] Checking output files...")

    for filename in REQUIRED_OUTPUTS:
        path = OUTPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing output file: {filename}"
            )

        print(f"[OK] {filename}")

    print("\n[INFO] Checking model files...")

    for filename in REQUIRED_MODELS:
        path = MODEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing model file: {filename}"
            )

        print(f"[OK] {filename}")

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    df = pd.read_csv(comparison_path)

    if df.empty:
        raise ValueError("model_comparison.csv is empty.")

    required_metrics = [
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]

    missing = [
        column for column in required_metrics
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metric columns: {missing}"
        )

    print("\n[SUCCESS] All outputs are valid.")

if __name__ == "__main__":
    validate_outputs()