import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub
import logging

# ========================== CONFIGURATION ==========================
CONFIG = {
    "repo_owner": "RedLordezh7Venom",
    "repo_name": "uberfareMLOPs",
    "experiment_name": "Tree Hyperparameter Tuning"
}

dagshub.init(repo_owner=CONFIG["repo_owner"], repo_name=CONFIG["repo_name"], border=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== DATA LOADING ==========================
def load_prepared_data():
    df = pd.read_csv('../data/raw/uber.csv')
    df.dropna(inplace=True)
    df = df[df['fare_amount'] > 0]
    
    # Simple engineering for tuning
    df['dist_km'] = np.sqrt((df.pickup_longitude - df.dropoff_longitude)**2 + (df.pickup_latitude - df.dropoff_latitude)**2)
    
    X = df[['dist_km', 'passenger_count', 'pickup_latitude', 'pickup_longitude']]
    y = df['fare_amount']
    return train_test_split(X, y, test_size=0.3, random_state=10)

# ========================== TUNING ==========================
def run_tuning():
    X_train, X_test, y_train, y_test = load_prepared_data()
    
    param_grid = {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 10, 20],
        "max_leaf_nodes": [None, 32, 64]
    }

    logging.info("Starting Grid Search...")
    
    with mlflow.start_run(run_name="DT GridSearchCV Optimization") as parent_run:
        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=10), param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log sub-runs for every param combination (Mirrors your reference exp3)
        for i, (params, score) in enumerate(zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"])):
            run_name = f"DT_Iter_{i}"
            with mlflow.start_run(run_name=run_name, nested=True):
                # We re-train quickly to get the test metrics (not just CV)
                model = DecisionTreeRegressor(**params, random_state=10)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                mlflow.log_params(params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("cv_mse", -score)

        # Log the heavy Winner
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(best_model, "best_tree_model")
        
        print(f"✅ Tuning Complete. Best Params: {grid_search.best_params_}")

if __name__ == "__main__":
    run_tuning()
