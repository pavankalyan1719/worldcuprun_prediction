from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from preprocess import load_and_preprocess_data
import joblib

df, X, y = load_and_preprocess_data()
groups = (
    df["Year"].astype(str)
    + "_"
    + df["Match Number"].astype(str)
)


# Split data by match
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


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Create models
linear_model = LinearRegression()

tree_model = DecisionTreeRegressor(
    random_state=42
)

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

gradient_model = GradientBoostingRegressor(
    random_state=42
)

extra_model = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42
)


# Train models
linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)
gradient_model.fit(X_train, y_train)
extra_model.fit(X_train, y_train)


print("All models trained successfully.")


# Select best model
best_model = extra_model


# Save model
joblib.dump(
    best_model,
    "models/extra_trees_model.pkl"
)

print("Best model saved successfully.")