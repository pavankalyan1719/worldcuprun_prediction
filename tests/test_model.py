import joblib
import pandas as pd


# Load saved model
model = joblib.load(
    "models/extra_trees_model.pkl"
)

print("Model loaded successfully.")


# Check that model has prediction capability
assert hasattr(model, "predict")

print("Model prediction test passed.")