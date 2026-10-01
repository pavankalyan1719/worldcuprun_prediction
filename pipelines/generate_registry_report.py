from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    print("[INFO] World Cup Run Prediction")

    client = MlflowClient()
    model_name = "WorldCupRunPrediction"

    try:
        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not all_versions:
            print(
                "[ERROR] No registered model versions found!"
            )
            return

        # Find Production model
        champion = next(
            (
                mv for mv in all_versions
                if mv.current_stage == "Production"
            ),
            None
        )

        if not champion:
            print(
                "[ERROR] No model found in Production stage!"
            )
            return

        # Get run details
        run = client.get_run(champion.run_id)

        # Extract model-specific metrics
        model_metrics = {}

        for key, value in run.data.metrics.items():
            if key.endswith("_mae"):
                model_metrics["MAE"] = value

            elif key.endswith("_mse"):
                model_metrics["MSE"] = value

            elif key.endswith("_rmse"):
                model_metrics["RMSE"] = value

            elif key.endswith("_r2"):
                model_metrics["R2"] = value

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "model_uri": (
                    f"models:/{model_name}/"
                    f"{champion.version}"
                )
            },
            "performance_metrics": model_metrics,
            "hyperparameters": run.data.params,
            "preprocessing_dependency": (
                "src/preprocess.py"
            ),
            "dataset": (
                "data/raw/world_cup_score.csv"
            ),
            "processed_dataset": (
                "data/processed/processed_data.csv"
            )
        }

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        report_path = (
            artifacts_dir /
            "production_model_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                report,
                f,
                indent=4
            )

        print(
            f"[SUCCESS] Report generated for "
            f"Version {champion.version}"
        )

        print(
            f"[INFO] Saved to: {report_path}"
        )

    except Exception as e:
        print(
            f"[ERROR] Failed to generate report: {e}"
        )


if __name__ == "__main__":
    generate_registry_report()