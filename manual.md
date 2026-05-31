# Uber Fares MLOps Implementation Manual: A Step-by-Step Guide

This manual provides a detailed explanation of the MLOps pipeline architecture implemented in the Uber Fare Prediction project. It breaks down each core component of the pipeline, explaining its purpose (What), its necessity (Why), and its internal logic (How), complete with line-by-line analysis of key code segments.

---

## 1. Data Ingestion (`src/data/data_ingestion.py`)

### What
The Data Ingestion module reads the raw dataset from disk, drops redundant columns, and performs the initial train-test split.

### Why
Machine Learning models must be evaluated on unseen data to measure real-world generalization. Performing the split at the very beginning of the pipeline ensures that no subsequent steps (like scaling or outlier removal) accidentally "leak" information from the test set into the training set.

### How
It uses `yaml` to dynamically load the split ratio from `params.yaml`, drops specific ID columns, and utilizes `scikit-learn`'s `train_test_split` to create `train.csv` and `test.csv`.

### Line-by-Line Breakdown
```python
# Initial Drop
df = df.drop(['Unnamed: 0', 'key'], axis=1)

# Splitting
train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
```
* **Line 2 (`df = df.drop...`):** Removes index columns that carry no predictive power. If left in, tree-based models might overfit by memorizing these unique IDs.
* **Line 5 (`train_data, test_data = ...`):** Splits the data based on the `test_size` ratio defined in `params.yaml` (e.g., 0.2). The `random_state=42` ensures that every time this pipeline runs, the exact same rows end up in the training and testing sets, preserving reproducibility.

---

## 2. Feature Engineering & Scaling (`src/features/feature_engineering.py`)

### What
This module scales the numeric features to a standard range (mean of 0, standard deviation of 1) and serializes the scaler object for later use in production.

### Why
Many machine learning models, and especially the optimization algorithms used to train them, perform poorly if features exist on vastly different scales (e.g., `dist_km` might range from 0 to 50, while `passenger_count` ranges from 1 to 6). Scaling ensures all features contribute equally.

### How
It fits a `StandardScaler` strictly on the training data, transforms both train and test sets, and uses `pickle` to save the scaler to the `models/` directory.

### Line-by-Line Breakdown
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
...
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
```
* **Line 1 (`scaler.fit_transform(X_train)`):** Calculates the mean and standard deviation exclusively from `X_train` and scales it.
* **Line 2 (`scaler.transform(X_test)`):** Crucially, it applies the exact same mean and standard deviation found in the training set to the test set. Calling `fit` on the test set is a critical error known as "data leakage".
* **Line 4 (`pickle.dump...`):** Saves the fitted scaler as a binary artifact. The production API needs this exact file to apply identical transformations to live user inputs.

---

## 3. Automated Quality Gating (`tests/test_model.py`)

### What
An automated test suite that validates the latest Staging model's performance before allowing it to be promoted to Production.

### Why
Continuous Deployment (CD) is dangerous if broken models can overwrite working models. The quality gate acts as an automated "checkpoint" that enforces minimum performance thresholds (R²).

### How
It uses the `unittest` framework to load the Staging model from the MLflow Registry, evaluates it against the held-out test dataset, and asserts that the R² metric is above 0.50.

### Line-by-Line Breakdown
```python
y_pred = self.model.predict(X_test)
r2 = r2_score(y_test, y_pred)

expected_r2 = 0.50
self.assertGreaterEqual(r2, expected_r2, f"Model R2 ({r2:.2f}) is below threshold")
```
* **Line 1 (`y_pred = ...`):** Runs the newly trained Staging model on the test features.
* **Line 2 (`r2 = ...`):** Calculates the R-squared score, representing the proportion of the variance in the fare amount that is predictable from the features.
* **Line 4-5 (`self.assertGreaterEqual...`):** Hard stops the CI pipeline if the score is below 0.50. If this assertion fails, the script exits with an error code, causing GitHub Actions to halt deployment.

---

## 4. FastAPI Lifespan & Context (`app/main.py`)

### What
The FastAPI application serves the model over HTTP. The `lifespan` context manager handles loading the machine learning models into memory when the server boots up.

### Why
Models and scalers are large binary files. Loading them on every single API request would result in unacceptably high latency. By loading them globally at startup, predictions can be served in milliseconds.

### How
It uses FastAPI's asynchronous context manager (`@asynccontextmanager`) to trigger `load_assets()` exactly once before the server accepts traffic.

### Line-by-Line Breakdown
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield

app = FastAPI(title="Uber Fare Service", lifespan=lifespan)
```
* **Line 1 (`@asynccontextmanager`):** A decorator that allows FastAPI to treat this function as the setup/teardown handler for the server.
* **Line 3 (`load_assets()`):** Calls the function that deserializes `scaler.pkl` and downloads the Production model from the MLflow Registry into global variables.
* **Line 4 (`yield`):** This is where the server pauses and begins accepting live HTTP traffic. Anything written after `yield` would execute upon server shutdown.
* **Line 6 (`app = ...`):** Instantiates the API and attaches the lifespan manager so the framework knows to execute it.

---

## 5. Model Promotion CI/CD (`scripts/promote_model.py`)

### What
A script that officially transitions a model from the `Staging` stage to the `Production` stage in the MLflow Model Registry.

### Why
Once a model passes the Quality Gate, it needs to be flagged as the active model for downstream systems (like the Docker build process). The registry handles this state management.

### How
It uses the `mlflow.client` to interface with DagsHub, fetches the latest Staging version, and updates its tag to Production while archiving any existing Production model.

### Line-by-Line Breakdown
```python
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)
```
* **Line 1 (`client.transition...`):** Issues an API call to the tracking server to update the model's metadata.
* **Line 3 (`version=...`):** Explicitly specifies which version of the model is being promoted (identified dynamically earlier in the script).
* **Line 5 (`archive_existing_versions=True`):** A critical safety mechanism. It ensures that whatever model was previously in Production is safely moved to the "Archived" stage, rather than deleted, allowing for instant rollbacks if the new model fails in the real world.

---

## 6. Docker Containerization (`Dockerfile`)

### What
Docker packages the FastAPI application, its dependencies, and the trained model artifacts into a single, immutable, runnable image.

### Why
"It works on my machine" is the bane of software deployment. Docker ensures that the environment where the model is served in production is identical down to the OS-level libraries to the environment where it was built. 

### How
It uses a lightweight Python 3.12 base image, installs the specific `uv` resolved dependencies from `requirements.txt`, copies the application code, and uniquely, bakes the `models/` directory directly into the image to ensure rapid startup times.

### How It Works Under the Hood
Docker leverages Linux kernel features like **cgroups** (control groups) to limit resource usage (CPU, memory) and **namespaces** to isolate processes, networking, and file systems. When you build an image, Docker creates a series of read-only "layers" based on your `Dockerfile` instructions. When the container runs, it adds a thin read-write layer on top. This layered architecture ensures that containers are lightweight, start instantly, and share common OS libraries without needing a full virtual machine hypervisor.

### Line-by-Line Breakdown
```dockerfile
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY models/ ./models/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
* **Line 1 (`FROM python...`):** Starts from a minimal Linux environment containing only Python, keeping the final image size small.
* **Line 3 (`RUN pip install...`):** Installs required libraries without caching the downloaded installers, further reducing image bloat.
* **Line 5 (`COPY models/...`):** Bakes the actual `.pkl` artifacts into the container. This means the container doesn't rely on downloading the model from MLflow every time it boots, eliminating network dependency and speeding up autoscaling.
* **Line 6 (`CMD...`):** The default command that executes when the container starts, launching the Uvicorn ASGI server.

---

## 7. Kubernetes / EKS Deployment (`deployment.yaml`)

### What
Kubernetes (K8s) is the orchestration engine that runs the Docker container. AWS EKS (Elastic Kubernetes Service) is the managed cloud version of K8s.

### Why
While Docker runs a single container, it doesn't handle what happens if that container crashes, or if you get a sudden spike of 10,000 users. Kubernetes automatically restarts crashed pods, scales the number of running containers based on CPU load, and routes incoming web traffic to healthy containers.

### How
It uses a declarative YAML manifest consisting of two parts: a `Deployment` (which manages the pods/containers) and a `Service` (which exposes them to the network).

### How It Works Under the Hood
Kubernetes operates on a cluster of machines. The **Control Plane** (the brain) manages the state of the cluster, primarily using **etcd** (a highly available key-value store). The worker nodes run a **kubelet** agent that communicates with the Control Plane and executes containers (pods). When you apply a `Deployment`, the Control Plane compares the *desired state* (1 replica) with the *actual state* (0 replicas). It then schedules a pod on a healthy worker node. The `Service` component uses `kube-proxy` to maintain network routing rules, ensuring that traffic reaching port 30080 is correctly forwarded to the dynamic internal IP of the pod, even if the pod is destroyed and recreated on a different node.

### Line-by-Line Breakdown
```yaml
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: docker.io/library/uber-fare-app:latest
        env:
        - name: CAPSTONE_TEST
          value: "<token>"
```
* **Line 2 (`replicas: 1`):** Tells K8s to ensure exactly 1 instance of the app is always running. If it dies, K8s spawns a new one.
* **Line 7 (`image: ...`):** Points to the specific Docker image to run. In production, this points to AWS ECR.
* **Line 9-10 (`env: ...`):** Injects the DagsHub token as an environment variable at runtime. This keeps secrets out of the Dockerfile/image itself.

---

## 8. Prometheus & Grafana Monitoring

### What
Prometheus is a time-series metric scraper, and Grafana is a visualization dashboard. Together, they form the observability stack for the ML API.

### Why
Once a model is deployed, its accuracy can degrade over time (data drift). Also, you need to know if the API is slow or throwing errors. Prometheus collects metrics, and Grafana graphs them so engineers can set up alerts.

### How
The FastAPI app uses `prometheus_client` to expose an HTTP endpoint (`/metrics`). Prometheus periodically pings this endpoint to scrape the data, which Grafana then queries.

### How It Works Under the Hood
Unlike traditional monitoring tools that wait for apps to "push" data to them, Prometheus uses a **pull model**. It runs a time-series database and periodically sends HTTP GET requests to your app's `/metrics` endpoint to scrape the current state of the counters and histograms. These metrics are stored with timestamps. Grafana then connects to Prometheus as a data source and executes PromQL (Prometheus Query Language) queries to transform these raw time-series data points into visual dashboards and alerts.

### Line-by-Line Breakdown
```python
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"])
PREDICTION_VALUE = Histogram("model_prediction_fare_amount", "Fares")

# Inside /predict endpoint
PREDICTION_VALUE.observe(fare)
```
* **Line 1 (`Counter...`):** An ever-increasing counter tracking exactly how many times each API endpoint is hit.
* **Line 2 (`Histogram...`):** Tracks the distribution of data. Instead of just a count, it groups data into "buckets" (e.g., how many fares were $10-$20, $20-$30).
* **Line 5 (`PREDICTION_VALUE.observe(fare)`):** After the model predicts a fare, that value is logged. If Grafana suddenly shows all predicted fares are $0, we know the model is failing silently.

---

## 9. DagsHub & MLflow Tracking

### What
DagsHub provides the remote server hosting for both Git repositories and an MLflow Tracking Server. MLflow is the tool that logs experiment parameters, metrics, and models.

### Why
Data science involves hundreds of experiments (changing `max_depth`, testing new features). Without MLflow, it's impossible to remember which settings produced the best model. DagsHub centralizes this so the whole team can see the experiments.

### How
The pipeline calls `mlflow.log_params()` and `mlflow.sklearn.log_model()`. The `CAPSTONE_TEST` token authenticates the script to DagsHub via environment variables.

### How It Works Under the Hood
**MLflow** runs a Tracking Server backed by a database (to store parameters and metrics) and an artifact store (to save `.pkl` files). **DagsHub** provides a managed wrapper around MLflow, integrating it with a Git repository and an S3-compatible object store. When `mlflow.log_model()` is called, the Python MLflow client serializes the Scikit-Learn model and uploads the binary to the DagsHub S3 bucket, while simultaneously writing the metadata (R² score, hyperparameters) to the DagsHub MLflow database, linking everything to the specific Git commit hash.

### Line-by-Line Breakdown
```python
mlflow.set_tracking_uri("https://dagshub.com/RedLordezh7Venom/uberfareMLOPs.mlflow")
with mlflow.start_run():
    mlflow.log_params(params['model_building'])
    mlflow.log_metric("r2_score", r2)
```
* **Line 1 (`set_tracking_uri`):** Directs MLflow to send data over the internet to DagsHub, rather than saving it to a local folder.
* **Line 2 (`with mlflow.start_run()`):** Creates a unique ID for this specific training attempt.
* **Line 3 (`log_params`):** Saves the hyperparameters (like `max_depth`) used in this run.
* **Line 4 (`log_metric`):** Saves the resulting accuracy, allowing you to easily sort all experiments by highest R² on the DagsHub UI.

---

## 10. Data Version Control (DVC)

### What
DVC is exactly like Git, but engineered for large data files and machine learning pipelines.

### Why
Git cannot handle 500MB CSV files or 2GB model weights. DVC stores the actual heavy files in cloud storage (DagsHub S3), and only commits tiny text pointer files (`.dvc`) to Git. Furthermore, DVC acts as a pipeline orchestrator like `Make`.

### How
The `dvc.yaml` file defines a Directed Acyclic Graph (DAG) of pipeline stages. When `dvc repro` is called, DVC checks file hashes. If the raw data hasn't changed, it won't rerun data ingestion, saving massive amounts of compute time.

### How It Works Under the Hood
DVC implements **Content-Addressable Storage**. When you track a file (like `uber.csv`), DVC calculates its MD5 hash, moves the actual file to a hidden cache directory (`.dvc/cache`), and names it using the hash. It then creates a tiny `.dvc` text file containing that hash and leaves a symlink (or copy) in your workspace. When you push, only the tiny `.dvc` file goes to Git, while the massive cached file is uploaded to the DagsHub S3 remote. For pipelines (`dvc repro`), DVC builds a Directed Acyclic Graph (DAG) from `dvc.yaml` and calculates hashes for all inputs and outputs. If the hashes match the previous run, DVC knows the stage is fully cached and safely skips execution.

### Line-by-Line Breakdown
```yaml
model_evaluation:
  cmd: uv run python -m src.models.model_evaluation
  deps:
  - models/model.pkl
  - data/processed/test_final.csv
  metrics:
  - reports/metrics.json
```
* **Line 2 (`cmd...`):** The exact terminal command DVC runs for this stage.
* **Line 3-5 (`deps...`):** DVC watches these files. If `model.pkl` changes, DVC knows it *must* rerun the evaluation stage. If they haven't changed, it skips this step entirely.
* **Line 6-7 (`metrics...`):** Tells DVC that this JSON file contains important performance numbers. DVC can then use `dvc metrics diff` to show how accuracy changed between Git commits.

---

## 11. CI/CD (GitHub Actions)

### What
Continuous Integration / Continuous Deployment (CI/CD) automates the execution of the entire pipeline whenever code is pushed to GitHub.

### Why
Humans make mistakes; they forget to run tests or deploy the wrong model version. CI/CD enforces strict, automated rules. Code is only deployed if the pipeline succeeds and the model passes the Quality Gate.

### How
Defined in `.github/workflows/ci.yaml`, it spins up an ephemeral Linux server in the cloud, installs Python, runs `dvc repro`, executes unit tests, logs into AWS, builds Docker, and commands Kubernetes to update.

### How It Works Under the Hood
GitHub Actions relies on **ephemeral runner virtual machines**. When a push event occurs, GitHub provisions a brand-new, clean Linux VM. The YAML workflow defines a sequence of steps. First, the runner clones the code (`actions/checkout`). Then, it installs dependencies. Because the runner is destroyed after the job finishes, no state is maintained. This guarantees a sterile testing environment. The `secrets` context securely injects encrypted keys (like AWS credentials) into the runner's memory only for the duration of the job, preventing them from being exposed in logs or the codebase.

### Line-by-Line Breakdown
```yaml
- name: Deploy to EKS
  if: github.ref == 'refs/heads/main' && success()
  run: |
    aws eks update-kubeconfig --name $EKS_CLUSTER_NAME
    kubectl apply -f deployment.yaml
```
* **Line 2 (`if: ...`):** The most important rule. This step ONLY executes if the user merged code into the `main` branch, AND all previous steps (like the R² quality gate) returned `success()`.
* **Line 4 (`aws eks...`):** Authenticates the GitHub runner with your AWS cloud account.
* **Line 5 (`kubectl apply...`):** Commands the cloud Kubernetes cluster to pull the newly built Docker image and seamlessly replace the old running containers with zero downtime.
