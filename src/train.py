from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

def train_models(X_train, y_train):

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
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

    for model in models.values():
        model.fit(X_train, y_train)

    return models
