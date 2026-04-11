import pandas as pd
import numpy as np
import os
import logging
from src.logger import logging

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def preprocess_uber_data(df):
    logging.info("Applying feature engineering (Haversine & DateTime)...")
    
    # Handle missing values
    df.dropna(inplace=True)
    
    # Calculate Travel Distance
    df['dist_km'] = haversine_distance(
        df['pickup_longitude'], df['pickup_latitude'], 
        df['dropoff_longitude'], df['dropoff_latitude']
    )
    
    # Parse pickup_datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df.dropna(subset=['pickup_datetime'], inplace=True)
    
    # Extract time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    
    # Filter logical fare issues
    df = df[df['fare_amount'] > 0]
    
    return df

def main():
    try:
        logging.info("--- Starting Data Preprocessing ---")
        
        # Load raw split data
        train_df = pd.read_csv('data/raw/train.csv')
        test_df = pd.read_csv('data/raw/test.csv')
        
        # Process both sets
        processed_train = preprocess_uber_data(train_df)
        processed_test = preprocess_uber_data(test_df)
        
        # Save to Processed folder
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        processed_train.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
        processed_test.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
        
        # --- OPTION: UPLOAD PROCESSED DATA TO S3 (Uncomment to use) ---
        # from src.connections.s3_connection import s3_operations
        # s3 = s3_operations(bucket_name="my-uber-bucket", aws_access_key="...", aws_secret_key="...")
        # s3.s3_client.upload_file(os.path.join(output_dir, "train_processed.csv"), "my-uber-bucket", "processed/train.csv")
        
        logging.info("✅ Preprocessing Complete: train_processed.csv and test_processed.csv created in data/processed/")
        
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}", exc_info=True)

if __name__ == "__main__":
    main()