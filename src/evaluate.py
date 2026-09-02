import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import load_and_preprocess_data
from sklearn.model_selection import GroupShuffleSplit

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)


# Load data
df, X, y = load_and_preprocess_data()


# Create match groups
groups = (
    df["Year"].astype(str)
    + "_"
    + df["Match Number"].astype(str)
)


# Same train-test split
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# Create models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42
    )
}


# Store results
results = []


# Train and evaluate
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, pred)

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2
    ])

    print(name)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)
    print("-" * 40)


# Create results DataFrame
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]
)


# Save results
results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# --------------------------------
# R² GRAPH
# --------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title("R² Score Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("R² Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/r2_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------
# MAE GRAPH
# --------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("MAE Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("MAE")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/mae_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------
# RMSE GRAPH
# --------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.title("RMSE Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/rmse_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------
# ACTUAL VS PREDICTED
# --------------------------------

extra_pred = predictions["Extra Trees"]

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    extra_pred
)

minimum = min(
    y_test.min(),
    extra_pred.min()
)

maximum = max(
    y_test.max(),
    extra_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")

plt.title(
    "Actual vs Predicted Final Score - Extra Trees"
)

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# --------------------------------
# PREDICTION ERROR
# --------------------------------

errors = y_test.values - extra_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    extra_pred,
    errors
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Final Score")
plt.ylabel("Prediction Error")

plt.title(
    "Prediction Error - Extra Trees"
)

plt.tight_layout()

plt.savefig(
    "outputs/prediction_error.png",
    dpi=300
)

plt.show()


print("All evaluation graphs saved successfully.")