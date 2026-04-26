import unittest
import mlflow
import os
import pandas as pd
import pickle
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from src.logger import logging

class TestUberModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Auth Setup (using our environment-aware pattern)
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_tracking_uri("https://dagshub.com/RedLordezh7Venom/uberfareMLOPs.mlflow")
        
        # 2. Load latest Staging model
        cls.model_name = "UberFareRegressor"
        client = mlflow.MlflowClient()
        try:
            latest_version = client.get_latest_versions(cls.model_name, stages=["Staging"])[0].version
            model_uri = f'models:/{cls.model_name}/{latest_version}'
            cls.model = mlflow.pyfunc.load_model(model_uri)
            cls.version = latest_version
        except Exception as e:
            logging.warning(f"Could not load from registry, falling back to local: {e}")
            cls.model = pickle.load(open('models/model.pkl', 'rb'))
            cls.version = "local"

        # 3. Load Scaler
        cls.scaler = pickle.load(open('models/scaler.pkl', 'rb'))

        # 4. Load params
        with open('params.yaml', 'r') as f:
            cls.params = yaml.safe_load(f)

        # 5. Load test data
        cls.test_data = pd.read_csv('data/processed/test_final.csv')

    def test_model_loaded(self):
        """Verify model exists."""
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        """Verify model can predict on valid scaled input."""
        target = self.params['feature_engineering']['target']
        X_test = self.test_data.drop(columns=[target])
        
        # Check input columns match scaler expectations
        self.assertEqual(len(X_test.columns), 7) # dist_km, hour, day, month, year, dayofweek, passenger_count
        
        # Run one prediction
        sample = X_test.head(1)
        prediction = self.model.predict(sample)
        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):
        """Verify model meets minimum R2 threshold."""
        target = self.params['feature_engineering']['target']
        X_test = self.test_data.drop(columns=[target])
        y_test = self.test_data[target]

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # Threshold: Model should at least explain 50% of the variance
        expected_r2 = 0.50
        self.assertGreaterEqual(r2, expected_r2, f"Model R2 ({r2:.2f}) is below threshold ({expected_r2})")

if __name__ == "__main__":
    unittest.main()
