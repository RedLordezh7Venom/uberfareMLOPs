import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from exp2_feature_eng import load_and_preprocess # Reuse preprocessing
import mlflow
import mlflow.xgboost
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', border=True)

def train_xgboost():
    mlflow.set_experiment("UberFare_ModelComparison")
    
    with mlflow.start_run(run_name="XGBRegressor_GPU"):
        df = load_and_preprocess()
        
        features = [
            'pickup_longitude', 'pickup_latitude', 
            'dropoff_longitude', 'dropoff_latitude', 
            'passenger_count', 'hour', 'dayofweek', 'dist_travel_km'
        ]
        X = df[features]
        y = df['fare_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost parameters (with GPU if possible)
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "tree_method": "gpu_hist" # Set to 'hist' if no GPU
        }
        
        try:
            xgb = XGBRegressor(**params)
            xgb.fit(X_train, y_train)
        except Exception as e:
            print("GPU failed, falling back to CPU...")
            params["tree_method"] = "hist"
            xgb = XGBRegressor(**params)
            xgb.fit(X_train, y_train)
        
        mlflow.log_params(params)
        
        rmse = np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(xgb, "xgboost_model")
        
        print(f"XGBoost RMSE: {rmse}")

if __name__ == "__main__":
    train_xgboost()
