import os
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="[ %(asctime)s ] %(levelname)s - %(message)s")

def promote_model():
    """Promotes the latest Staging model to Production and archives old ones."""
    logging.info("--- Model Promotion Process Started ---")

    # 1. Auth Setup (using our environment-aware pattern)
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri("https://dagshub.com/RedLordezh7Venom/uberfareMLOPs.mlflow")
        logging.info("Using token-based DagsHub auth (CI mode).")
    else:
        import dagshub
        dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', mlflow=True)
        logging.info("Using interactive DagsHub auth (local mode).")

    client = mlflow.MlflowClient()
    model_name = "UberFareRegressor"

    try:
        # 2. Get the latest version in staging
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            logging.error("No model versions found in 'Staging' stage. Promotion aborted.")
            return

        latest_version_staging = staging_versions[0].version
        logging.info(f"Found Staging version: {latest_version_staging}")

        # 3. Archive the current production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in prod_versions:
            logging.info(f"Archiving existing Production version: {version.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

        # 4. Promote the new model to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )
        logging.info(f"✅ SUCCESS: Model version {latest_version_staging} promoted to Production.")

    except Exception as e:
        logging.error(f"❌ ERROR during promotion: {e}")
        raise

if __name__ == "__main__":
    promote_model()
