import numpy as np
import pandas as pd

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def transform_input_data(pickup_datetime, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat, passenger_count):
    """Transforms raw inputs into engineered features for the model."""
    
    # 1. Distance Calculation
    dist_km = haversine_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    
    # 2. DateTime Extraction
    dt = pd.to_datetime(pickup_datetime)
    
    # Construct feature dictionary
    features = {
        "dist_km": dist_km,
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "year": dt.year,
        "dayofweek": dt.dayofweek,
        "passenger_count": int(passenger_count)
    }
    
    return pd.DataFrame([features])
