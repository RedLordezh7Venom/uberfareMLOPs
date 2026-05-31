# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first (leverages Docker layer cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY params.yaml .

# models/ is intentionally NOT copied here.
# At startup, load_assets() in app/main.py will pull the model from
# the MLflow/DagsHub registry using the CAPSTONE_TEST env variable.
# If the registry is unavailable, ensure models/model.pkl is volume-mounted.

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
