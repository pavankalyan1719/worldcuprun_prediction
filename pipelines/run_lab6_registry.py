"""
Lab 6: MLflow Model Registry
World Cup Run Prediction
"""

from pathlib import Path
import sys
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "World_Cup_Run_Prediction"
MODEL_NAME = "WorldCupRunPrediction"

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

client = MlflowClient()

def register_model():
    print("=" * 60)
    print("LAB 6: MLFLOW MODEL REGISTRY")
    print("World Cup Run Prediction")
    print("=" * 60)

    print("\n[INFO] Searching for MLflow experiment...")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("[ERROR] MLflow experiment not found.")
        print("\n[INFO] Run Lab 4 first:")
        print("       python pipelines/run_lab4_tracking.py")
        sys.exit(1)

    print("\n[INFO] Searching for latest MLflow run...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        print("[ERROR] No MLflow runs found.")
        print("\n[INFO] Run Lab 4 first:")
        print("       python pipelines/run_lab4_tracking.py")
        sys.exit(1)

    run = runs[0]
    run_id = run.info.run_id

    print(f"[INFO] Latest Run ID: {run_id}")

    print("\n[INFO] Run Metrics:")
    for key, value in run.data.metrics.items():
        print(f"       {key}: {value}")

    selected_model = run.data.params.get("selected_model")

    if selected_model:
        print(f"\n[INFO] Selected Model: {selected_model}")
    else:
        print("\n[WARNING] selected_model parameter not found.")

    print("\n[INFO] Searching for MLflow model output...")

    model_outputs = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'"
    )

    if not model_outputs:
        print("[ERROR] No MLflow model output found for the latest run.")
        print("\n[INFO] Make sure Lab 4 logs the model.")
        sys.exit(1)

    logged_model = model_outputs[0]
    model_id = logged_model.model_id

    print(f"\n[INFO] MLflow Model ID: {model_id}")

    model_uri = f"runs:/{run_id}/model"

    print(f"[INFO] Model URI: {model_uri}")

    print("\n[INFO] Registering model...")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\n[SUCCESS] Model registered successfully!")
    print(f"[INFO] Model Name: {registered_model.name}")
    print(f"[INFO] Model Version: {registered_model.version}")

    client.update_model_version(
        name=MODEL_NAME,
        version=registered_model.version,
        description=(
            "World Cup Run Prediction model "
            "registered from the MLflow Lab 4 "
            "tracking pipeline."
        )
    )

    print("\n" + "=" * 60)
    print("MODEL REGISTRY INFORMATION")
    print("=" * 60)
    print(f"Model Name    : {MODEL_NAME}")
    print(f"Model Version : {registered_model.version}")
    print(f"Run ID        : {run_id}")
    print(f"Model ID      : {model_id}")
    print(f"Model URI     : {model_uri}")
    print("\n[SUCCESS] Lab 6 Model Registry completed!")
    print("=" * 60)

if __name__ == "__main__":
    register_model()