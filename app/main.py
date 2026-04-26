from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mlflow
import pickle
import os
import pandas as pd
import time
import dagshub
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from app.utils import transform_input_data

# --- MLflow / DagsHub Config ---
dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', mlflow=True)

app = FastAPI(title="Uber Fare Prediction Service")

# Prometheus setup
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry
)
PREDICTION_VALUE = Histogram(
    "model_prediction_fare_amount", "Distribution of predicted fares", registry=registry
)

# Template setup
templates = Jinja2Templates(directory="app/templates")

# --- Model & Scaler Loading ---
def load_assets():
    # 1. Load Scaler
    scaler_path = 'models/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 2. Load latest model from Staging
    model_name = "UberFareRegressor"
    client = mlflow.MlflowClient()
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])[0].version
        model_uri = f'models:/{model_name}/{latest_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        return scaler, model, latest_version
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        # Fallback to local if registry fails
        model = pickle.load(open('models/model.pkl', 'rb'))
        return scaler, model, "local"

SCALER, MODEL, VERSION = load_assets()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    return templates.TemplateResponse(request=request, name="index.html", context={"result": None, "version": VERSION})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pickup_datetime: str = Form(...),
    pickup_longitude: float = Form(...),
    pickup_latitude: float = Form(...),
    dropoff_longitude: float = Form(...),
    dropoff_latitude: float = Form(...),
    passenger_count: int = Form(...)
):
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    # 1. Transform raw inputs
    df_features = transform_input_data(
        pickup_datetime, pickup_longitude, pickup_latitude, 
        dropoff_longitude, dropoff_latitude, passenger_count
    )
    
    # 2. Apply Scaling
    target_col = "fare_amount" # The model expects the features we trained on
    # Note: We need to ensure the columns match the scaler's expectations
    features_list = ["dist_km", "hour", "day", "month", "year", "dayofweek", "passenger_count"]
    X_scaled = SCALER.transform(df_features[features_list])
    
    # 3. Predict
    prediction = MODEL.predict(pd.DataFrame(X_scaled, columns=features_list))
    fare = round(float(prediction[0]), 2)

    # Metrics
    PREDICTION_VALUE.observe(fare)
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return templates.TemplateResponse(request=request, name="index.html", context={
        "result": f"${fare}", 
        "version": VERSION
    })

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
