"""
Auto-retraining orchestrator.
Called by ingestion.py when any trigger condition is met.
Runs training, redeploys to AWS, sends Slack notification.
"""

import argparse
import logging
import os
import subprocess
import sys

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exporter.metrics import retrain_count_total, model_accuracy as model_accuracy_gauge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def send_slack_alert(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
        return
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Slack notification failed: {resp.status_code}")
    except Exception as e:
        logger.error(f"Slack notification exception: {e}")


def run_training(reason: str) -> dict:
    """Run train.py as subprocess and return new metadata."""
    logger.info(f"Starting training subprocess. Reason: {reason}")
    result = subprocess.run(
        ["python", "model/train.py", "--reason", reason],
        capture_output=True, text=True, timeout=600
    )

    if result.stdout:
        logger.info(f"Training output:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"Training stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # Load updated metadata
    import json
    with open("model/model_meta.json", "r") as f:
        return json.load(f)


def run_deploy():
    """Run deploy.sh to push new model to EC2."""
    deploy_script = "deploy/deploy.sh"
    if not os.path.exists(deploy_script):
        logger.warning(f"Deploy script not found at {deploy_script} — skipping redeploy.")
        return

    logger.info("Running deployment script...")
    result = subprocess.run(
        ["bash", deploy_script],
        capture_output=True, text=True, timeout=300
    )

    if result.stdout:
        logger.info(f"Deploy output:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"Deploy stderr:\n{result.stderr}")

    if result.returncode == 0:
        logger.info("Deployment successful.")
    else:
        logger.error(f"Deployment failed with exit code {result.returncode}")


def trigger(reason: str):
    """
    Main entrypoint for retraining pipeline.
    1. Increment counter
    2. Run training
    3. Redeploy
    4. Notify Slack
    """
    logger.info(f"=== Retraining triggered. Reason: {reason} ===")

    # Increment Prometheus counter
    retrain_count_total.inc()

    # Notify Slack that retraining started
    send_slack_alert(
        f"🔄 *Retraining Started*\n"
        f"Reason: `{reason}`\n"
        f"Training new model version..."
    )

    try:
        # Run training
        meta = run_training(reason)
        new_accuracy = meta.get("accuracy", 0.0)
        new_version = meta.get("version", "?")

        # Update Prometheus gauge
        model_accuracy_gauge.set(new_accuracy)

        logger.info(f"New model: v{new_version}, accuracy={new_accuracy:.4f}")

        # Redeploy to AWS
        run_deploy()

        # Notify Slack with results
        status_emoji = "✅" if new_accuracy >= 0.80 else "⚠️"
        send_slack_alert(
            f"{status_emoji} *Retraining Complete*\n"
            f"Reason: `{reason}`\n"
            f"Model version: `v{new_version}`\n"
            f"New accuracy: `{new_accuracy:.4f}`\n"
            f"{'Accuracy above threshold ✅' if new_accuracy >= 0.80 else 'WARNING: Accuracy below 0.80 ⚠️'}"
        )

    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}", exc_info=True)
        send_slack_alert(
            f"❌ *Retraining Failed*\n"
            f"Reason: `{reason}`\n"
            f"Error: `{str(e)}`"
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", type=str, default="manual", help="Trigger reason")
    args = parser.parse_args()
    trigger(args.reason)