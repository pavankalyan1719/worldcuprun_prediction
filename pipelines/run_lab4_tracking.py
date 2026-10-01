"""
Lab 4 MLflow Tracking Pipeline.

Runs the baseline workflow while logging:
- Parameters
- Metrics
- Selected trained model
- Output artifacts

World Cup Run Prediction
"""

from pathlib import Path
import sys

import mlflow
import mlflow.sklearn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT MODULES
# ============================================================

from src.preprocess import preprocess
from src.train import train_models
from src.evaluate import evaluate


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("LAB 4: MLFLOW TRACKING PIPELINE")
    print("WORLD CUP RUN PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "World_Cup_Run_Prediction"
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="lab4_tracking"
    ):

        # ====================================================
        # PARAMETERS
        # ====================================================

        print("\n[INFO] Logging parameters...")

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "group_split",
            "Year + Match Number"
        )

        mlflow.log_param(
            "models",
            5
        )

        # ====================================================
        # PREPROCESSING
        # ====================================================

        print("\n[INFO] Running preprocessing...")

        preprocess()

        # ====================================================
        # TRAINING
        # ====================================================

        print("\n[INFO] Training models...")

        trained_models, X_test, y_test = train_models()

        # ====================================================
        # EVALUATION
        # ====================================================

        print("\n[INFO] Evaluating models...")

        results = evaluate()

        # ====================================================
        # LOG METRICS
        # ====================================================

        print("\n[INFO] Logging model metrics...")

        for _, row in results.iterrows():

            prefix = (
                row["Model"]
                .lower()
                .replace(" ", "_")
            )

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

        # ====================================================
        # SELECT BEST MODEL
        # ====================================================

        best_model_name = results.loc[
            results["RMSE"].idxmin(),
            "Model"
        ]

        best_model = trained_models[
            best_model_name
        ]

        print(
            f"\n[INFO] Selected model: "
            f"{best_model_name}"
        )

        # ====================================================
        # LOG SELECTED MODEL NAME
        # ====================================================

        mlflow.log_param(
            "selected_model",
            best_model_name
        )

        # ====================================================
        # LOG SELECTED MODEL
        # ====================================================

        print(
            "\n[INFO] Logging selected model "
            "to MLflow..."
        )

        mlflow.sklearn.log_model(
            best_model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(
            "[SUCCESS] Model logged to MLflow."
        )

        # ====================================================
        # LOG OUTPUT ARTIFACTS
        # ====================================================

        print(
            "\n[INFO] Logging output artifacts..."
        )

        mlflow.log_artifacts(
            str(PROJECT_ROOT / "outputs")
        )

        print(
            "[SUCCESS] Output artifacts logged."
        )

    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 60)
    print(
        "[SUCCESS] MLflow tracking pipeline "
        "completed successfully."
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()