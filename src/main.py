from pathlib import Path
import joblib

from data_loader import load_data
from preprocessing import create_target
from feature_engineering import create_features


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "world_cup_score.csv"
MODEL_PATH = BASE_DIR / "models" / "extra_trees_model.pkl"


def main():

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print("Creating target variable...")

    df = create_target(df)

    print("Creating features...")

    df = create_features(df)

    print("Dataset prepared successfully.")

    print(df.head())


if __name__ == "__main__":
    main()