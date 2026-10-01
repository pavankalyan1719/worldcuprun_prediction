"""
Lab 3 baseline pipeline.

Runs preprocessing, model training and evaluation without MLflow tracking.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess
from src.train import train_models
from src.evaluate import evaluate


def main():
    print("=" * 60)
    print("LAB 3 - BASELINE PIPELINE")
    print("=" * 60)
    preprocess()
    train_models()
    evaluate()
    print("\nBaseline pipeline completed successfully.")


if __name__ == "__main__":
    main()
