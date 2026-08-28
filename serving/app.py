"""
ML Inference API
Endpoints:
  POST /predict  - run model inference
  GET  /health   - health check
  GET  /metrics  - Prometheus metrics

Runs ingestion as a background thread so metrics update in same process.
"""

import json
import logging
import os
import sys
import time
import threading
from contextlib import asynccontextmanager

import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exporter.metrics import (
    model_accuracy,
    response_delay_seconds,
    retrain_count_total,
    records_processed_total,
    distribution_drift_detected,
    feature_added,
    feature_removed,
    datalake_unavailable,
)
from ingestion.drift_detector import check_drift, load_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "model")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
API_URL = os.getenv("API_URL", "http://149.40.228.124:6500/records")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "45"))

model = None
model_version = 0
model_features = []


def load_latest_model():
    global model, model_version, model_features
    if not os.path.exists(META_PATH):
        logger.warning(f"No model metadata found at {META_PATH}")
        return
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    model_path = meta.get("model_path", "")
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return
    model = joblib.load(model_path)
    model_version = meta.get("version", 0)
    model_features = meta.get("features", [])
    acc = meta.get("accuracy", 0.0)
    model_accuracy.set(acc)
    logger.info(f"Loaded model v{model_version} | accuracy={acc:.4f}")


def ingestion_loop():
    """Background thread: polls API, updates metrics in same process."""
    logger.info("Ingestion background thread started.")
    prev_schema = []

    while True:
        try:
            resp = requests.get(API_URL, timeout=10)

            if resp.status_code == 503:
                logger.warning("API returned 503")
                datalake_unavailable.inc()
                time.sleep(POLL_INTERVAL)
                continue

            data = resp.json()
            records = data if isinstance(data, list) else data.get("records", [])

            if not records:
                time.sleep(POLL_INTERVAL)
                continue

            # Schema monitoring
            schema = list(records[0].keys()) if records else []
            if prev_schema:
                added = set(schema) - set(prev_schema)
                removed = set(prev_schema) - set(schema)
                for f in added:
                    logger.warning(f"Feature ADDED: {f}")
                    feature_added.inc()
                for f in removed:
                    logger.warning(f"Feature REMOVED: {f}")
                    feature_removed.inc()
            prev_schema = schema

            # Update records counter
            records_processed_total.inc(len(records))
            logger.info(f"Ingested {len(records)} records. Schema: {schema}")

            # Drift detection — reuses the same tested per-feature z-score logic
            # from ingestion/drift_detector.py, so production matches what's tested.
            try:
                baseline = load_baseline()
                drift_detected, drifted_features = check_drift(records, baseline)
                distribution_drift_detected.set(1 if drift_detected else 0)
                if drift_detected:
                    logger.warning(f"Drift detected in feature(s): {drifted_features}")
            except Exception as e:
                logger.error(f"Drift detection error: {e}")

        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            datalake_unavailable.inc()

        time.sleep(POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_latest_model()
    # Start ingestion in background thread
    t = threading.Thread(target=ingestion_loop, daemon=True)
    t.start()
    logger.info("Background ingestion thread started.")
    yield


app = FastAPI(title="MLOps Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: dict


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        load_latest_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet. Run training first.")

    start_time = time.time()
    try:
        if model_features:
            row = []
            for feat in model_features:
                val = request.features.get(feat, 0.0)
                try:
                    row.append(float(val))
                except:
                    row.append(0.0)
            X = np.array([row])
        else:
            X = np.array([[float(v) for v in request.features.values()]])

        prediction = int(model.predict(X)[0])
        confidence = float(np.max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else 1.0
        latency = time.time() - start_time
        response_delay_seconds.observe(latency)

        return PredictResponse(prediction=prediction, confidence=confidence, model_version=model_version)

    except Exception as e:
        latency = time.time() - start_time
        response_delay_seconds.observe(latency)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    load_latest_model()
    return {"status": "reloaded", "model_version": model_version}