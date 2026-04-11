import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(X_train, y_train, params):
    try:
        logging.info("Initializing DecisionTreeRegressor with params from YAML...")
        model = DecisionTreeRegressor(
            max_depth=params['max_depth'],
            max_leaf_nodes=params['max_leaf_nodes'],
            random_state=params['random_state']
        )
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def main():
    try:
        logging.info("--- Model Building Process Started ---")
        config = load_params('params.yaml')
        
        # Load Final Features
        train_path = 'data/processed/train_final.csv'
        train_data = pd.read_csv(train_path)
        
        target = config['feature_engineering']['target']
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        
        # Train
        model = train_model(X_train, y_train, config['model_building'])
        
        # Save
        os.makedirs('models', exist_ok=True)
        model_path = 'models/model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        logging.info(f"✅ Model saved to {model_path}")

    except Exception as e:
        logging.error(f"❌ Model Building Failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
