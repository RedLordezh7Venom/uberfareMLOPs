import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', border=True)

def load_data():
    df = pd.read_csv('../data/raw/uber.csv')
    # Preprocessing
    df = df.drop(['Unnamed: 0', 'key'], axis=1)
    df.dropna(inplace=True)
    return df

def train_baseline():
    mlflow.set_experiment("UberFare_Baseline")
    
    with mlflow.start_run(run_name="RF_Baseline_NoFeatures"):
        df = load_data()
        
        # Select raw features only
        X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
        y = df['fare_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log parameters
        params = {"n_estimators": 100, "random_state": 42}
        mlflow.log_params(params)
        
        # Model
        rf = RandomForestRegressor(**params, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predictions & Metrics
        preds = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        
        # Log model
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"Baseline RMSE: {rmse}")

if __name__ == "__main__":
    train_baseline()
