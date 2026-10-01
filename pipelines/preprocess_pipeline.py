from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess

if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("World Cup Run Prediction")
    print("=" * 60)

    preprocess()

    print("\n[SUCCESS] Preprocessing completed successfully.")