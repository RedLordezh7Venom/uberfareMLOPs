import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.preprocessing import StandardScaler
from src.logger import logging

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters retrieved from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise

def apply_scaling(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, target: str):
    """Apply Standard Scaling to the numeric features."""
    try:
        logging.info(f"Applying StandardScaler to features: {features}")
        scaler = StandardScaler()

        # Split into X and y
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # Fit and transform
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame
        train_final = pd.DataFrame(X_train_scaled, columns=features)
        train_final[target] = y_train.values

        test_final = pd.DataFrame(X_test_scaled, columns=features)
        test_final[target] = y_test.values

        # Save Scaler for production
        os.makedirs('models', exist_ok=True)
        pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
        logging.info("Scaler successfully saved to models/scaler.pkl")

        return train_final, test_final
    except Exception as e:
        logging.error(f"Error during feature scaling: {e}")
        raise

def main():
    try:
        logging.info("--- Feature Engineering Started ---")
        params = load_params('params.yaml')
        feature_list = params['feature_engineering']['features']
        target_col = params['feature_engineering']['target']

        # Load data from the preprocessing output (processed folder)
        train_path = 'data/processed/train_processed.csv'
        test_path = 'data/processed/test_processed.csv'
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Loaded processed data. Train shape: {train_data.shape}")

        # Apply Scaling
        train_final, test_final = apply_scaling(train_data, test_data, feature_list, target_col)

        # Store the finalized data
        output_dir = "data/processed"
        train_final.to_csv(os.path.join(output_dir, "train_final.csv"), index=False)
        test_final.to_csv(os.path.join(output_dir, "test_final.csv"), index=False)
        
        logging.info(f"✅ Feature Engineering Complete. Ready for training.")

    except Exception as e:
        logging.error(f"❌ Feature Engineering Failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
