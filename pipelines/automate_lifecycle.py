from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    print("[INFO] World Cup Run Prediction")

    client = MlflowClient()
    model_name = "WorldCupRunPrediction"
    metric_to_optimize = "rmse"

    try:
        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not all_versions:
            print("[ERROR] No registered model versions found.")
            return

        # 1. Discover new models and move them to Staging
        new_models = [
            mv for mv in all_versions
            if mv.current_stage == "None"
        ]

        if new_models:
            print(
                f"\n[INFO] Found {len(new_models)} "
                "new model(s). Moving them to Staging..."
            )

            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Staging",
                    archive_existing_versions=False
                )

                print(
                    f"  -> Version {mv.version} "
                    "is now in Staging."
                )

        # Refresh registry data
        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        staging_models = [
            mv for mv in all_versions
            if mv.current_stage == "Staging"
        ]

        if not staging_models:
            print(
                "\n[INFO] No models in Staging to evaluate."
            )
            return

        # 2. Evaluate Staging models
        print("\n[INFO] Evaluating Staging candidates...")

        best_challenger = None
        best_challenger_score = float("inf")

        for mv in staging_models:
            run = client.get_run(mv.run_id)

            metric_keys = [
                key for key in run.data.metrics
                if key.endswith("_rmse")
            ]

            if not metric_keys:
                print(
                    f"  -> Version {mv.version}: "
                    "RMSE metric not found."
                )
                continue

            score = min(
                run.data.metrics[key]
                for key in metric_keys
            )

            print(
                f"  -> Staging Candidate: "
                f"Version {mv.version} | "
                f"RMSE: {score:.4f}"
            )

            if score < best_challenger_score:
                best_challenger_score = score
                best_challenger = mv

        if best_challenger is None:
            print(
                "\n[ERROR] No valid Staging model "
                "with RMSE found."
            )
            return

        print(
            f"\n[INFO] Best Challenger: "
            f"Version {best_challenger.version} "
            f"(RMSE: {best_challenger_score:.4f})"
        )

        # 3. Find current Production Champion
        production_models = [
            mv for mv in all_versions
            if mv.current_stage == "Production"
        ]

        current_champion = (
            production_models[0]
            if production_models
            else None
        )

        promote_challenger = False

        if not current_champion:
            print(
                "[INFO] No model currently in Production."
            )
            print(
                "[INFO] Challenger will become "
                "the Production model."
            )
            promote_challenger = True

        else:
            champ_run = client.get_run(
                current_champion.run_id
            )

            champion_metric_keys = [
                key for key in champ_run.data.metrics
                if key.endswith("_rmse")
            ]

            if not champion_metric_keys:
                print(
                    "[ERROR] Champion RMSE metric not found."
                )
                return

            champion_score = min(
                champ_run.data.metrics[key]
                for key in champion_metric_keys
            )

            print(
                f"[INFO] Current Production Champion: "
                f"Version {current_champion.version} | "
                f"RMSE: {champion_score:.4f}"
            )

            # Lower RMSE is better
            if best_challenger_score < champion_score:
                print(
                    "[SUCCESS] Challenger has lower RMSE "
                    "than the Champion!"
                )
                promote_challenger = True
            else:
                print(
                    "[INFO] Challenger did not improve "
                    "RMSE. Champion remains in Production."
                )

        # 4. Promote challenger
        if promote_challenger:
            print(
                f"\n[INFO] Promoting Version "
                f"{best_challenger.version} to Production..."
            )

            client.transition_model_version_stage(
                name=model_name,
                version=best_challenger.version,
                stage="Production",
                archive_existing_versions=False
            )

            if current_champion:
                print(
                    f"[INFO] Archiving previous Champion "
                    f"Version {current_champion.version}..."
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=current_champion.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

        # 5. Clean remaining Staging models
        final_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        remaining_staging = [
            mv for mv in final_versions
            if mv.current_stage == "Staging"
        ]

        if remaining_staging:
            print(
                "\n[INFO] Cleaning up remaining "
                "Staging models..."
            )

            for mv in remaining_staging:
                print(
                    f"  -> Archiving Version "
                    f"{mv.version}"
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

        print(
            "\n[SUCCESS] Automated Model Lifecycle "
            "execution complete!"
        )

    except Exception as e:
        print(
            f"[ERROR] Failed to automate lifecycle: {e}"
        )


if __name__ == "__main__":
    automate_champion_challenger()