import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def transition_to_staging(model_name: str, version: int):
    """Transition the already registered model version to Staging."""
    try:
        client = mlflow.tracking.MlflowClient()
        logging.info(f"Transitioning {model_name} version {version} to Staging...")
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        logging.info(f"✅ Model {model_name} version {version} is now in STAGING.")
    except Exception as e:
        logging.error(f"Error during stage transition: {e}")
        raise

def main():
    try:
        logging.info("--- Model Lifecycle Management Started ---")
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_tracking_uri("https://dagshub.com/RedLordezh7Venom/uberfareMLOPs.mlflow")
        else:
            dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', mlflow=True)
        
        model_info = load_model_info('reports/experiment_info.json')
        model_name = "UberFareRegressor"
        
        # Transition the version that was just created by evaluation script
        transition_to_staging(model_name, model_info['version'])
        
    except Exception as e:
        logging.error(f"❌ Lifecycle Transition Failed: {e}")

if __name__ == '__main__':
    main()
