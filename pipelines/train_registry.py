from pathlib import Path
import sys
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess
from src.train import train_models
from src.evaluate import evaluate

EXPERIMENT_NAME = "World_Cup_Run_Prediction"
MODEL_NAME = "WorldCupRunPrediction"

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

client = MlflowClient()

def train_and_register():
    print("=" * 60)
    print("TRAINING AND MODEL REGISTRY")
    print("World Cup Run Prediction")
    print("=" * 60)

    print("\n[INFO] Running preprocessing...")
    preprocess()

    print("\n[INFO] Training models...")
    trained_models, X_test, y_test = train_models()

    print("\n[INFO] Evaluating models...")
    results = evaluate()

    best_model_name = results.loc[
        results["RMSE"].idxmin(),
        "Model"
    ]

    best_model = trained_models[best_model_name]

    print(
        f"\n[INFO] Selected Model: {best_model_name}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="train_registry"
    ) as run:

        mlflow.log_param(
            "selected_model",
            best_model_name
        )

        mlflow.log_param(
            "models",
            5
        )

        for _, row in results.iterrows():
            prefix = row["Model"].lower().replace(" ", "_")

            mlflow.log_metric(
                f"{prefix}_mae",
                float(row["MAE"])
            )

            mlflow.log_metric(
                f"{prefix}_mse",
                float(row["MSE"])
            )

            mlflow.log_metric(
                f"{prefix}_rmse",
                float(row["RMSE"])
            )

            mlflow.log_metric(
                f"{prefix}_r2",
                float(row["R2"])
            )

        mlflow.sklearn.log_model(
            best_model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        mlflow.log_artifacts(
            str(PROJECT_ROOT / "outputs")
        )

        run_id = run.info.run_id

    print(f"\n[INFO] Run ID: {run_id}")

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    model_outputs = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'"
    )

    if not model_outputs:
        raise RuntimeError(
            "No logged model found."
        )

    model_id = model_outputs[0].model_id
    model_uri = f"runs:/{run_id}/model"

    print(f"[INFO] Model ID: {model_id}")
    print(f"[INFO] Model URI: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client.update_model_version(
        name=MODEL_NAME,
        version=registered_model.version,
        description=(
            "World Cup Run Prediction model "
            "trained and registered through "
            "the training registry pipeline."
        )
    )

    print("\n[SUCCESS] Model registered successfully.")
    print(f"[INFO] Model Name: {MODEL_NAME}")
    print(
        f"[INFO] Model Version: "
        f"{registered_model.version}"
    )

if __name__ == "__main__":
    train_and_register()