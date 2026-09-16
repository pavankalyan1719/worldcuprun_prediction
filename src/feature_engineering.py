def create_features(df):

    df["Current Run Rate"] = (
        df["Runs Scored till that over"] /
        df["Over Number"].replace(0, 1)
    )

    df["Remaining Overs"] = 50 - df["Over Number"]

    return df