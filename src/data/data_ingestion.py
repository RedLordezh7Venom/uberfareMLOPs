import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split
from src.logger import logging # Using your custom logger

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    try:
        logging.info("--- Starting Data Ingestion ---")
        config = load_params('params.yaml')
        
        raw_data_path = config['data_ingestion']['raw_data_path']
        test_size = config['data_ingestion']['test_size']
        
        # Load the base dataset
        logging.info(f"Loading raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        
        # Initial Cleaning (Drop columns we don't need at all)
        df = df.drop(['Unnamed: 0', 'key'], axis=1)
        
        # Train-Test Split
        logging.info(f"Splitting data with test_size={test_size}")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # Create directories if they don't exist
        output_dir = "data/raw"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save split data
        train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        
        logging.info("✅ Ingestion Complete: train.csv and test.csv created in data/raw/")
        
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}", exc_info=True)

if __name__ == "__main__":
    main()
