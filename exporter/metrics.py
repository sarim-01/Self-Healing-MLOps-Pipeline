"""
Prometheus metrics definitions.
Every other module imports from here — do not define metrics elsewhere.
"""

from prometheus_client import Gauge, Counter, Histogram

# --- Model ---
model_accuracy = Gauge(
    'model_accuracy',
    'Current validation accuracy of the deployed model (0.0 - 1.0)'
)

# --- Ingestion ---
records_processed_total = Counter(
    'records_processed_total',
    'Total number of records ingested from the API since startup'
)

datalake_unavailable = Counter(
    'datalake_unavailable',
    'Number of times the /records endpoint returned 503'
)

# --- Schema ---
feature_added = Counter(
    'feature_added',
    'Number of features added to the schema since startup'
)

feature_removed = Counter(
    'feature_removed',
    'Number of features removed from the schema since startup'
)

# --- Drift ---
distribution_drift_detected = Gauge(
    'distribution_drift_detected',
    'Set to 1 when drift is detected in the current batch, 0 otherwise'
)

# --- Retraining ---
retrain_count_total = Counter(
    'retrain_count_total',
    'Total number of times the model has been retrained'
)

# --- Serving ---
response_delay_seconds = Histogram(
    'response_delay_seconds',
    'Latency of each /predict API call in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
)