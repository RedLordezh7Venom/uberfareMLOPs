import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# --- PRODUCTION USE (Commented out) ---
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# dagshub_url = "https://dagshub.com"
# repo_owner = "RedLordezh7Venom"
# repo_name = "uberfareMLOPs"
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# --- LOCAL USE ---
dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', mlflow=True)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment info file {file_path} not found. Run evaluation first.")
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f'Model info loaded from {file_path}')
        return model_info
    except Exception as e:
        logging.error(f'Error loading model info: {e}')
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Registering model from URI: {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.info(f'✅ Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error(f'Error during model registration: {e}')
        raise

def main():
    try:
        logging.info("--- Model Registration Process Started ---")
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        # You can change this name to 'Uber_Fare_Best_Model' or similar
        model_name = "UberFareRegressor"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error(f'Failed to complete the model registration process: {e}', exc_info=True)

if __name__ == '__main__':
    main()
