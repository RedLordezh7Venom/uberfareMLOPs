import unittest
from fastapi.testclient import TestClient
from app.main import app
import os

class TestUberApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # We use the FastAPI TestClient
        cls.client = TestClient(app)

    def test_home_page(self):
        """Verify the home page loads correctly."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Check for Uber-specific UI elements we added
        self.assertIn("Uber Fare Prediction", response.text)
        self.assertIn("id=\"map\"", response.text)

    def test_predict_endpoint(self):
        """Verify the prediction POST request returns a fare."""
        # Sample payload for a ride in NYC
        payload = {
            "pickup_datetime": "2026-04-26T12:00",
            "pickup_longitude": -73.98,
            "pickup_latitude": 40.75,
            "dropoff_longitude": -73.99,
            "dropoff_latitude": 40.76,
            "passenger_count": 1
        }
        
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        
        # Check if the result is displayed (we look for the $ sign in our 'result' div)
        self.assertIn("Estimated Fare:", response.text)
        self.assertIn("$", response.text)

    def test_metrics_endpoint(self):
        """Verify Prometheus metrics are exposed."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("app_request_count", response.text)

if __name__ == "__main__":
    unittest.main()
