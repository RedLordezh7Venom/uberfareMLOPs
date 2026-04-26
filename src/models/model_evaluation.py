import pandas as pd
import numpy as np
import os
import pickle
import json
import yaml
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.logger import logging

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Calculating evaluation metrics (RMSE, R2, MAE)...")
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            "rmse": float(rmse),
            "r2": float(r2),
            "mae": float(mae)
        }
        return metrics
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info(f'Model info saved to {file_path}')
    except Exception as e:
        logging.error(f'Error occurred while saving the model info: {e}')
        raise

def main():
    try:
        logging.info("--- Model Evaluation Process Started ---")
        config = load_params('params.yaml')
        
        # --- MLflow / DagsHub Auth ---
        # In CI: CAPSTONE_TEST env var is set → use token-based auth (no browser needed)
        # Locally: falls back to dagshub.init() which opens a browser for OAuth
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_tracking_uri("https://dagshub.com/RedLordezh7Venom/uberfareMLOPs.mlflow")
            logging.info("Using token-based DagsHub auth (CI mode).")
        else:
            dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', mlflow=True)
            logging.info("Using interactive DagsHub auth (local mode).")

        mlflow.set_experiment("UberFare_Production_Pipeline")

        with mlflow.start_run() as run:
            # 1. Load Model
            model_path = 'models/model.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 2. Load Data
            test_path = 'data/processed/test_final.csv'
            test_data = pd.read_csv(test_path)
            
            target = config['feature_engineering']['target']
            X_test = test_data.drop(columns=[target])
            y_test = test_data[target]
            
            # 3. Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            
            # 4. Save Locally
            os.makedirs('reports', exist_ok=True)
            with open('reports/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # 5. Log to MLflow
            logging.info("Logging metrics and model to MLflow...")
            mlflow.log_metrics(metrics)
            mlflow.log_params(config['model_building'])
            
            # Build input example to help the registry discover the model
            input_example = X_test.head(5) 

            # Direct Registration: This is the most robust way to solve the "Unable to find" sync issue.
            # It registers the model while the upload connection is still active.
            logging.info("Registering model directly during upload...")
            model_info_registry = mlflow.sklearn.log_model(
                model, 
                "model", 
                input_example=input_example,
                pip_requirements=["scikit-learn", "pandas", "numpy"],
                registered_model_name="UberFareRegressor"
            )
            
            # Log artifacts
            mlflow.log_artifact('reports/metrics.json')

            # 6. Save Model Info (For Registry Script to handle Stage Transition)
            model_version = model_info_registry.registered_model_version
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Add version to info for easier transition later
            with open('reports/experiment_info.json', 'r') as f:
                info = json.load(f)
            info['version'] = model_version
            with open('reports/experiment_info.json', 'w') as f:
                json.dump(info, f, indent=4)
            
            logging.info(f"✅ Evaluation Complete. RMSE: {metrics['rmse']}")

    except Exception as e:
        logging.error(f"❌ Model Evaluation Failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
