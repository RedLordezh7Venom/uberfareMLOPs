import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import dagshub
import logging

import os

# ========================== CONFIGURATION ==========================
CONFIG = {
    "repo_owner": "RedLordezh7Venom",
    "repo_name": "uberfareMLOPs",
    "experiment_name": "Feature Impact Analysis"
}

# Calculated Path to data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "uber.csv")

dagshub.init(repo_owner=CONFIG["repo_owner"], repo_name=CONFIG["repo_name"])
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== FEATURE ENGINEERING ==========================
def haversine_np(lon1, lat1, lon2, lat2):
    """Calculates haversine distance exactly as in the notebook"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df = df[df['fare_amount'] > 0]
    return df

# ========================== EXECUTION ==========================
def run_feature_test():
    df = load_data()
    
    # Feature Scenarios
    SCENARIOS = {
        "Raw_Coordinates": ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'],
        "Haversine_Only": ["dist_km", "passenger_count"]
    }

    # Prepare Engineered Feature
    df['dist_km'] = haversine_np(df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude)
    y = df['fare_amount']

    with mlflow.start_run(run_name="Feature Engineering Comparison"):
        for name, features in SCENARIOS.items():
            with mlflow.start_run(run_name=f"Data_Type: {name}", nested=True):
                X = df[features]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
                
                model = DecisionTreeRegressor(max_depth=10, random_state=10)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                mlflow.log_param("engineered_features", "True" if "dist_km" in features else "False")
                mlflow.log_metric("rmse", rmse)
                print(f"Scenario: {name} | RMSE: {rmse:.4f}")

if __name__ == "__main__":
    run_feature_test()
