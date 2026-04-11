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

def main():
    try:
        logging.info("--- Model Evaluation Process Started ---")
        config = load_params('params.yaml')
        
        # Init DagsHub
        # Below code block is for production use
        # # -------------------------------------------------------------------------------------
        # # Set up DagsHub credentials for MLflow tracking
        # dagshub_token = os.getenv("CAPSTONE_TEST")
        # if not dagshub_token:
        #     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # dagshub_url = "https://dagshub.com"
        # repo_owner = "vikashdas770"
        # repo_name = "YT-Capstone-Project"

        # # Set up MLflow tracking URI
        # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        # # -------------------------------------------------------------------------------------

        dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', border=True)
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
            mlflow.sklearn.log_model(model, "model")
            
            # Log artifacts
            mlflow.log_artifact('reports/metrics.json')
            
            logging.info(f"✅ Evaluation Complete. RMSE: {metrics['rmse']}")

    except Exception as e:
        logging.error(f"❌ Model Evaluation Failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
