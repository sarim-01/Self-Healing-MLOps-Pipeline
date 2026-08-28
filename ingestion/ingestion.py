"""
Data ingestion script.
- Polls /records endpoint every POLL_INTERVAL seconds
- Detects schema changes (feature added / removed)
- Detects distribution drift
- Handles 503 errors gracefully
- Triggers retraining when conditions are met
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import requests

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporter.metrics import (
    records_processed_total,
    datalake_unavailable,
    feature_added,
    feature_removed,
    distribution_drift_detected,
)
from ingestion.drift_detector import check_drift, load_baseline

# --- Config ---
API_URL = os.getenv("API_URL", "http://149.40.228.124:6500/records")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "45"))          # seconds
RETRAIN_AFTER_N_BATCHES = int(os.getenv("RETRAIN_AFTER_N_BATCHES", "5"))
DATA_PATH = "data/records.csv"
SCHEMA_PATH = "data/schema_baseline.json"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --- Slack alerting ---
def send_slack_alert(message: str):
    """Send a message to Slack via webhook URL from environment."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Slack alert failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Slack alert exception: {e}")


# --- Schema helpers ---
def load_schema() -> list:
    """Load saved schema from disk. Returns empty list if not found."""
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r") as f:
            return json.load(f)
    return []


def save_schema(schema: list):
    """Persist current schema to disk."""
    os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)


def detect_schema_changes(old_schema: list, new_schema: list) -> tuple[list, list]:
    """Return (added_features, removed_features) between two schemas."""
    old_set = set(old_schema)
    new_set = set(new_schema)
    added = list(new_set - old_set)
    removed = list(old_set - new_set)
    return added, removed


# --- CSV storage ---
def append_to_csv(records: list[dict], schema: list):
    """Append a batch of records to the CSV file."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    file_exists = os.path.exists(DATA_PATH)

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


# --- Retraining trigger ---
def trigger_retraining(reason: str):
    """Call the retraining orchestrator."""
    logger.info(f"Triggering retraining. Reason: {reason}")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "model/retrain_trigger.py", "--reason", reason],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            logger.info("Retraining completed successfully.")
        else:
            logger.error(f"Retraining failed:\n{result.stderr}")
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")


# --- Main loop ---
def run():
    logger.info("Starting ingestion loop...")
    logger.info(f"API: {API_URL} | Poll interval: {POLL_INTERVAL}s")

    current_schema = load_schema()
    baseline_stats = load_baseline()
    batches_since_retrain = 0
    schema_changed = False

    while True:
        try:
            logger.info(f"Fetching records from API...")
            resp = requests.get(API_URL, timeout=10)

            # --- Handle 503 ---
            if resp.status_code == 503:
                logger.warning("API returned 503 — service unavailable.")
                datalake_unavailable.inc()
                send_slack_alert("🚨 *DataLakeUnavailable*: Data source returned 503. Check API availability.")
                time.sleep(POLL_INTERVAL)
                continue

            resp.raise_for_status()
            payload = resp.json()

            # Handle both list and dict response formats
            if isinstance(payload, list):
                records = payload
                new_schema = list(records[0].keys()) if records else []
            else:
                new_schema = payload.get("schema", [])
                records = payload.get("records", [])

            if not records:
                logger.warning("Empty records batch received.")
                time.sleep(POLL_INTERVAL)
                continue

            logger.info(f"Received {len(records)} records. Schema: {new_schema}")

            # --- Schema change detection ---
            if current_schema:
                added, removed = detect_schema_changes(current_schema, new_schema)

                for f in added:
                    logger.warning(f"Schema change: feature ADDED → '{f}'")
                    feature_added.inc()
                    send_slack_alert(f"⚠️ *FeatureAdded*: New feature detected in schema: `{f}`. Retraining may be required.")
                    schema_changed = True

                for f in removed:
                    logger.warning(f"Schema change: feature REMOVED → '{f}'")
                    feature_removed.inc()
                    send_slack_alert(f"⚠️ *FeatureRemoved*: Feature `{f}` dropped from schema. Verify pipeline compatibility.")
                    schema_changed = True

            # Save updated schema
            save_schema(new_schema)
            current_schema = new_schema

            # --- Store records ---
            append_to_csv(records, new_schema)
            records_processed_total.inc(len(records))
            logger.info(f"Stored {len(records)} records. Total processed: running.")

            # --- Drift detection ---
            drift_detected, drifted_features = check_drift(records, baseline_stats)
            baseline_stats = {}  # drift_detector updates on disk; reload next cycle

            if drift_detected:
                distribution_drift_detected.set(1)
                send_slack_alert(
                    f"📊 *DistributionDrift*: Data distribution drift detected in features: "
                    f"`{', '.join(drifted_features)}`. Model may be stale."
                )
            else:
                distribution_drift_detected.set(0)

            # --- Retraining decision ---
            batches_since_retrain += 1
            should_retrain = (
                schema_changed or
                drift_detected or
                batches_since_retrain >= RETRAIN_AFTER_N_BATCHES
            )

            if should_retrain:
                reason = []
                if schema_changed:
                    reason.append("schema_change")
                if drift_detected:
                    reason.append("distribution_drift")
                if batches_since_retrain >= RETRAIN_AFTER_N_BATCHES:
                    reason.append(f"batch_threshold_{batches_since_retrain}")

                trigger_retraining("+".join(reason))
                batches_since_retrain = 0
                schema_changed = False

        except requests.exceptions.Timeout:
            logger.error("Request timed out.")
            datalake_unavailable.inc()

        except requests.exceptions.ConnectionError:
            logger.error("Connection error — API unreachable.")
            datalake_unavailable.inc()

        except Exception as e:
            logger.error(f"Unexpected error in ingestion loop: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()