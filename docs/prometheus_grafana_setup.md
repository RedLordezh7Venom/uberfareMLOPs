# Monitoring Setup Guide: Prometheus & Grafana

This guide explains how to set up Prometheus (for metric scraping) and Grafana (for dashboard visualization) on AWS EC2 instances to monitor the Uber Fare Prediction API.

While our FastAPI application exposes metrics at `/metrics`, we need external servers to actually collect and graph that data. We will use two separate EC2 instances to mimic a production-grade decoupled architecture.

---

## Part 1: Prometheus Setup (The Data Collector)

**What it does:** Prometheus operates on a "pull" model. It will periodically send HTTP requests to our Kubernetes LoadBalancer IP (e.g., `http://<your-eks-lb>:8000/metrics`) to collect data like request counts and latency, storing them in its internal time-series database.

### Step 1: Provision the EC2 Instance
1. Launch a new Ubuntu EC2 instance in AWS.
   - **Type:** `t3.medium` (Prometheus can be memory-intensive as data grows).
   - **Storage:** 20GB General Purpose SSD.
   - **Security Group:** Allow inbound traffic on port `9090` (Prometheus Web UI) and port `22` (SSH).

### Step 2: Connect and Update
SSH into your new instance:
```bash
ssh -i your-key.pem ubuntu@<prometheus-ec2-public-ip>
```
Update the system packages to ensure we have the latest security patches:
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 3: Download and Extract Prometheus
Download the compiled Linux binaries from GitHub:
```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
```
Extract the archive and rename the folder for simplicity:
```bash
tar -xvzf prometheus-2.46.0.linux-amd64.tar.gz
mv prometheus-2.46.0.linux-amd64 prometheus
```

### Step 4: Install to Standard System Paths
Move the configuration directory to `/etc` and the executable binary to `/usr/local/bin` so it can be run from anywhere:
```bash
sudo mv prometheus /etc/prometheus
sudo mv /etc/prometheus/prometheus /usr/local/bin/
```

### Step 5: Configure Prometheus to Scrape Our App
Prometheus needs to know *where* to pull metrics from. We configure this in `prometheus.yml`.

Open the file:
```bash
sudo nano /etc/prometheus/prometheus.yml
```
Replace the contents with the following. **Note:** Replace the target IP with the external LoadBalancer IP or NodePort IP of your Kubernetes deployed FastAPI app.
```yaml
global:
  scrape_interval: 15s  # How often to pull data

scrape_configs:
  - job_name: "uber-fare-fastapi"
    static_configs:
      # Replace this with your actual EKS LoadBalancer IP or NodePort URL
      - targets: ["a6bf6255d5f61470c9782b8955c98271-1409247973.us-east-1.elb.amazonaws.com:8000"]
```
Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`). Verify it looks correct:
```bash
cat /etc/prometheus/prometheus.yml
```

### Step 6: Run Prometheus
Verify the binary location:
```bash
which prometheus
```
Start the Prometheus server, passing it the configuration file we just made:
```bash
/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml
```
*Your Prometheus server is now running! You can view its raw UI by visiting `http://<prometheus-ec2-public-ip>:9090` in your browser.*

---

## Part 2: Grafana Setup (The Visualization Layer)

**What it does:** Grafana doesn't collect data itself. Instead, it connects to our Prometheus database and uses queries (PromQL) to draw beautiful charts, histograms, and set up alerts.

### Step 1: Provision the EC2 Instance
1. Launch a second Ubuntu EC2 instance.
   - **Type:** `t3.medium`.
   - **Storage:** 20GB General Purpose SSD.
   - **Security Group:** Allow inbound traffic on port `3000` (Grafana Web UI) and port `22` (SSH).

### Step 2: Connect and Update
SSH into the Grafana instance:
```bash
ssh -i your-key.pem ubuntu@<grafana-ec2-public-ip>
sudo apt update && sudo apt upgrade -y
```

### Step 3: Download and Install Grafana
Download the Debian package (`.deb`) for Grafana OSS (Open Source Software):
```bash
wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
```
Install the package using the `apt` package manager:
```bash
sudo apt install ./grafana_10.1.5_amd64.deb -y
```

### Step 4: Manage the Grafana Service
Unlike Prometheus (which we ran manually in the foreground), Grafana installs itself as a Linux `systemd` service. Let's start it and ensure it boots automatically if the server restarts.
```bash
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```
Check that it is running without errors:
```bash
sudo systemctl status grafana-server
```

### Step 5: Configure Grafana via Web UI
1. Open your browser and navigate to `http://<grafana-ec2-public-ip>:3000`.
2. Log in with the default credentials:
   - **Username:** `admin`
   - **Password:** `admin` *(It will prompt you to change this immediately).*

### Step 6: Connect Grafana to Prometheus
1. In the left sidebar, go to **Connections > Data Sources**.
2. Click **Add new data source** and select **Prometheus**.
3. In the **HTTP > URL** field, enter the URL of your Prometheus EC2 instance from Part 1:
   - `http://<prometheus-ec2-public-ip>:9090`
4. Scroll to the bottom and click **Save & Test**. You should see a green checkmark saying "Data source is working".

### Next Steps: Building Dashboards
Now that they are connected, you can create dashboards in Grafana. For example, to visualize the fare prediction distribution we created in our FastAPI app, you would add a new panel and use this PromQL query:
```promql
rate(model_prediction_fare_amount_sum[5m]) / rate(model_prediction_fare_amount_count[5m])
```
This queries the data Prometheus scraped from our Kubernetes cluster and visualizes it in real-time.
