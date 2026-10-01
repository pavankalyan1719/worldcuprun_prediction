from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "world_cup_score.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

RAW_REQUIRED_COLUMNS = [
    "Year",
    "Match Number"
]

PROCESSED_REQUIRED_COLUMNS = [
    "Year",
    "Match Number",
    "final_score"
]

def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    print(f"[INFO] File: {path}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Columns: {len(df.columns)}")

    return df

def validate_columns(df, required_columns):
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

def validate_data():
    print("=" * 60)
    print("DATA VALIDATION")
    print("World Cup Run Prediction")
    print("=" * 60)

    print("\n[INFO] Validating raw dataset...")
    raw_df = validate_file(RAW_PATH)
    validate_columns(raw_df, RAW_REQUIRED_COLUMNS)
    print("[SUCCESS] Raw dataset is valid.")

    print("\n[INFO] Validating processed dataset...")
    processed_df = validate_file(PROCESSED_PATH)
    validate_columns(processed_df, PROCESSED_REQUIRED_COLUMNS)
    print("[SUCCESS] Processed dataset is valid.")

    print("\n[SUCCESS] Data validation completed successfully.")

if __name__ == "__main__":
    validate_data()