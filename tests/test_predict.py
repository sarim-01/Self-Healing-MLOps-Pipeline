"""
Unit tests for the /predict endpoint in serving/app.py
"""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from fastapi.testclient import TestClient

    # Mock the model so we don't need a real pkl file
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch("serving.app.model", mock_model), \
         patch("serving.app.model_version", 1), \
         patch("serving.app.model_features", ["age", "salary"]):
        from serving.app import app
        client = TestClient(app)
        yield client


def test_health_endpoint(client):
    """Test that /health returns 200 and correct response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_key(client):
    """Test that /predict returns a prediction key."""
    payload = {"features": {"age": 25.0, "salary": 50000.0}}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_predict_returns_confidence_key(client):
    """Test that /predict returns a confidence key."""
    payload = {"features": {"age": 25.0, "salary": 50000.0}}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data


def test_predict_returns_model_version(client):
    """Test that /predict returns model_version key."""
    payload = {"features": {"age": 25.0, "salary": 50000.0}}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data


def test_predict_confidence_between_0_and_1(client):
    """Test that confidence value is between 0 and 1."""
    payload = {"features": {"age": 25.0, "salary": 50000.0}}
    response = client.post("/predict", json=payload)

    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_metrics_endpoint(client):
    """Test that /metrics returns Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_accuracy" in response.text