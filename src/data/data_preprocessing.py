import pandas as pd
import numpy as np
import os
import logging
from src.logger import logging

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculates haversine distance exactly as in the notebook"""
    try:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km
    except Exception as e:
        logging.error(f"Error calculating Haversine distance: {e}")
        raise

def preprocess_uber_data(df: pd.DataFrame, type_name: str) -> pd.DataFrame:
    """Preprocess the Uber dataset with extensive logging."""
    try:
        logging.info(f"Processing {type_name} data. Initial shape: {df.shape}")
        
        # 1. Handling Nulls
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logging.warning(f"Found {null_count} null values in {type_name} data. Removing them...")
            df.dropna(inplace=True)
            logging.info(f"Shape after null removal: {df.shape}")

        # 2. Distance Calculation
        logging.info("Calculating Haversine distance (km)...")
        df['dist_km'] = haversine_distance(
            df['pickup_longitude'], df['pickup_latitude'], 
            df['dropoff_longitude'], df['dropoff_latitude']
        )

        # 3. DateTime Transformation
        logging.info("Parsing pickup_datetime and extracting time features...")
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        
        # Drop rows where datetime couldn't be parsed
        invalid_dates = df['pickup_datetime'].isnull().sum()
        if invalid_dates > 0:
            logging.info(f"Removing {invalid_dates} rows with invalid date formats.")
            df.dropna(subset=['pickup_datetime'], inplace=True)

        df['hour'] = df['pickup_datetime'].dt.hour
        df['day'] = df['pickup_datetime'].dt.day
        df['month'] = df['pickup_datetime'].dt.month
        df['year'] = df['pickup_datetime'].dt.year
        df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
        
        # 4. Filter Fare Outliers
        negative_fares = (df['fare_amount'] <= 0).sum()
        if negative_fares > 0:
            logging.warning(f"Removing {negative_fares} rows with fare_amount <= 0.")
            df = df[df['fare_amount'] > 0]

        logging.info(f"Engineering complete for {type_name}. Final shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error during preprocessing of {type_name}: {e}")
        raise

def main():
    try:
        logging.info("--- Data Preprocessing Process Started ---")
        
        # Paths
        raw_train_path = 'data/raw/train.csv'
        raw_test_path = 'data/raw/test.csv'
        output_dir = "data/processed"

        # Load
        if not os.path.exists(raw_train_path):
            raise FileNotFoundError(f"Missing raw split files at {raw_train_path}")
            
        train_df = pd.read_csv(raw_train_path)
        test_df = pd.read_csv(raw_test_path)
        logging.info("Raw split files loaded successfully.")

        # Transform
        processed_train = preprocess_uber_data(train_df, "Train")
        processed_test = preprocess_uber_data(test_df, "Test")

        # Save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created processed data folder: {output_dir}")
            
        train_out = os.path.join(output_dir, "train_processed.csv")
        test_out = os.path.join(output_dir, "test_processed.csv")
        
        processed_train.to_csv(train_out, index=False)
        processed_test.to_csv(test_out, index=False)
        
        logging.info(f"✅ Preprocessing Successful. Processed files saved to {output_dir}")

    except Exception as e:
        logging.error(f"❌ Data Preprocessing Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()