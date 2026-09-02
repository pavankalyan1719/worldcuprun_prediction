import pandas as pd
import numpy as np


def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv("data/raw/world_cup_score.csv")

    # Identify first batting team for each match
    first_batting_team = df.groupby(
        ["Year", "Match Number"]
    )["Batting team"].transform("first")

    # Create target variable
    df["final_score"] = np.where(
        first_batting_team == df["Team 1"],
        df["Total Score for Team 1"],
        df["Total Score for Team 2"]
    )

    # Feature engineering
    df["Current Run Rate"] = (
        df["Runs Scored till that over"] / df["Over Number"]
    )

    df["Remaining Overs"] = 50 - df["Over Number"]

    # Select features
    features = [
        "Year",
        "City",
        "Team 1",
        "Team 2",
        "Batting team",
        "Bowling team",
        "Over Number",
        "Runs Scored in over",
        "Runs Scored till that over",
        "Wicket Taken in over",
        "Wickets Taken till that over",
        "Current Run Rate",
        "Remaining Overs"
    ]

    X = df[features].copy()
    y = df["final_score"].copy()

    # Convert categorical data to numerical data
    categorical_columns = [
        "City",
        "Team 1",
        "Team 2",
        "Batting team",
        "Bowling team"
    ]

    for column in categorical_columns:
        X[column] = X[column].astype("category").cat.codes

    print("Data preprocessing completed.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return df, X, y