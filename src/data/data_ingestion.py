import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split
from src.logger import logging 
from src.connections.s3_connection import s3_operations 

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file with error handling."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Successfully retrieved parameters from {params_path}")
        return params
    except FileNotFoundError:
        logging.error(f"Critical Error: Parameter file '{params_path}' not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw CSV from disk."""
    try:
        logging.info(f"Attempting to load raw data from: {data_path}")
        df = pd.read_csv(data_path)
        logging.info(f"Data successfully loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}: {e}")
        raise

def main():
    try:
        logging.info("--- Data Ingestion Process Started ---")
        
        # Load Config
        config = load_params('params.yaml')
        raw_data_path = config['data_ingestion']['raw_data_path']
        test_size = config['data_ingestion']['test_size']
        
        # Data Loading
        df = load_data(raw_data_path)
        
        # Initial Drop
        logging.info("Dropping unnecessary columns: ['Unnamed: 0', 'key']")
        df = df.drop(['Unnamed: 0', 'key'], axis=1)

        # Splitting
        logging.info(f"Initiating Train-Test split (Test Size: {test_size})")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # Directory Creation
        output_dir = "data/raw"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")
        
        # Saving
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        logging.info(f"Saving splits. Train rows: {len(train_data)}, Test rows: {len(test_data)}")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logging.info(f"✅ Ingestion Successful: Files saved to {train_path} and {test_path}")

    except Exception as e:
        logging.error(f"❌ Data Ingestion Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
