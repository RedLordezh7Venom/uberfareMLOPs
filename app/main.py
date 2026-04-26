from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import mlflow
import pickle
import os
import pandas as pd
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from app.utils import transform_input_data

# Prometheus setup (module-level is fine — no auth needed)
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

templates = Jinja2Templates(directory="app/templates")

# Global state populated at startup
SCALER = None
MODEL = None
VERSION = "unknown"

def setup_mlflow():
    """Authenticate with DagsHub/MLflow. Works in both CI and local."""
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri("https://dagshub.com/RedLordezh7Venom/uberfareMLOPs.mlflow")
    else:
        import dagshub
        dagshub.init(repo_owner='RedLordezh7Venom', repo_name='uberfareMLOPs', mlflow=True)

def load_assets():
    global SCALER, MODEL, VERSION
    # 1. Load Scaler
    with open('models/scaler.pkl', 'rb') as f:
        SCALER = pickle.load(f)

    # 2. Load model — try registry first, fall back to local
    setup_mlflow()
    model_name = "UberFareRegressor"
    client = mlflow.MlflowClient()
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])[0].version
        MODEL = mlflow.pyfunc.load_model(f'models:/{model_name}/{latest_version}')
        VERSION = latest_version
    except Exception as e:
        print(f"Registry load failed ({e}), falling back to local model.")
        MODEL = pickle.load(open('models/model.pkl', 'rb'))
        VERSION = "local"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup logic ONLY when the server actually starts — not on import."""
    load_assets()
    yield

app = FastAPI(title="Uber Fare Prediction Service", lifespan=lifespan)

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

    df_features = transform_input_data(
        pickup_datetime, pickup_longitude, pickup_latitude,
        dropoff_longitude, dropoff_latitude, passenger_count
    )

    features_list = ["dist_km", "hour", "day", "month", "year", "dayofweek", "passenger_count"]
    X_scaled = SCALER.transform(df_features[features_list])
    prediction = MODEL.predict(pd.DataFrame(X_scaled, columns=features_list))
    fare = round(float(prediction[0]), 2)

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
