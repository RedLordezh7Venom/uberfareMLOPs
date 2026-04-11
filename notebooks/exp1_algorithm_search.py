import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import dagshub
import logging
import time

# ========================== CONFIGURATION ==========================
CONFIG = {
    "repo_owner": "RedLordezh7Venom",
    "repo_name": "uberfareMLOPs",
    "experiment_name": "Algorithm Search",
    "test_size": 0.3,
    "random_state": 10
}

# Setup DagsHub & MLflow
dagshub.init(repo_owner=CONFIG["repo_owner"], repo_name=CONFIG["repo_name"], border=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== DATA LOADING ==========================
def load_clean_data():
    df = pd.read_csv('../data/raw/uber.csv')
    df = df.drop(['Unnamed: 0', 'key'], axis=1)
    df.dropna(inplace=True)
    df = df[df['fare_amount'] > 0]
    return df

# ========================== EXECUTION ==========================
def run_search():
    df = load_clean_data()
    X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
    y = df['fare_amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])

    # Algorithms to compare
    ALGORITHMS = {
        "OLS_StatModels": "statmodels", # Custom logic for OLS
        "SGD_Regressor": SGDRegressor(alpha=0.1, max_iter=1000),
        "Decision_Tree_Base": DecisionTreeRegressor(random_state=10)
    }

    with mlflow.start_run(run_name="Primary Algorithm Search"):
        for name, model in ALGORITHMS.items():
            with mlflow.start_run(run_name=name, nested=True):
                logging.info(f"Running Experiment: {name}")
                start_time = time.time()
                
                if name == "OLS_StatModels":
                    # Custom OLS Logic
                    X_train_const = sm.add_constant(X_train)
                    fit_model = sm.OLS(y_train, X_train_const).fit()
                    y_pred = fit_model.predict(sm.add_constant(X_test))
                    # Stats don't have log_model in same way
                else:
                    # Sklearn Logic
                    if name == "SGD_Regressor":
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_train)
                        X_te = scaler.transform(X_test)
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_te)
                    mlflow.sklearn.log_model(model, "model")

                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                mlflow.log_metrics({"rmse": rmse, "r2": r2, "runtime_sec": time.time() - start_time})
                print(f"Model: {name} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

if __name__ == "__main__":
    run_search()
