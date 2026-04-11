import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, asin, sqrt
import mlflow
import mlflow.sklearn
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', border=True)

def distance_transform(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    for pos in range(len(longitude1)):
        long1, lati1, long2, lati2 = map(radians, [longitude1[pos], latitude1[pos], longitude2[pos], latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a)) * 6371
        travel_dist.append(c)
    return travel_dist

def load_and_preprocess():
    df = pd.read_csv('../data/raw/uber.csv')
    df.dropna(inplace=True)
    
    # Feature Engineering
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = df.assign(
        hour=df.pickup_datetime.dt.hour,
        day=df.pickup_datetime.dt.day,
        month=df.pickup_datetime.dt.month,
        year=df.pickup_datetime.dt.year,
        dayofweek=df.pickup_datetime.dt.dayofweek
    )
    
    df['dist_travel_km'] = distance_transform(
        df['pickup_longitude'].to_numpy(),
        df['pickup_latitude'].to_numpy(),
        df['dropoff_longitude'].to_numpy(),
        df['dropoff_latitude'].to_numpy()
    )
    
    # Filter outliers
    df = df[df['fare_amount'] > 0]
    df = df[df['dist_travel_km'] > 0]
    
    return df

def train_with_features():
    mlflow.set_experiment("UberFare_FeatureEngineering")
    
    with mlflow.start_run(run_name="RF_With_Distance_Time"):
        df = load_and_preprocess()
        
        # Features including engineered ones
        features = [
            'pickup_longitude', 'pickup_latitude', 
            'dropoff_longitude', 'dropoff_latitude', 
            'passenger_count', 'hour', 'dayofweek', 'dist_travel_km'
        ]
        X = df[features]
        y = df['fare_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        params = {"n_estimators": 100, "random_state": 42}
        mlflow.log_params(params)
        mlflow.log_dict({"features": features}, "features.json")
        
        rf = RandomForestRegressor(**params, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(rf, "model_featured")
        
        print(f"Feature Engineering RMSE: {rmse}")

if __name__ == "__main__":
    train_with_features()
