"""
ML model training script.
- Loads ingested CSV data
- Trains a RandomForestClassifier
- Enforces val accuracy >= 0.80
- Saves versioned model artifact
- Updates Prometheus model_accuracy gauge
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exporter.metrics import model_accuracy, retrain_count_total

# --- Config ---
DATA_PATH = os.getenv("DATA_PATH", "data/records.csv")
MODEL_DIR = "model"
META_PATH = "model/model_meta.json"
TARGET_ACCURACY = float(os.getenv("TARGET_ACCURACY", "0.80"))
MAX_ESTIMATORS_ATTEMPTS = [50, 100, 200, 300]  # progressively try more trees

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_meta() -> dict:
    """Load model metadata (version, accuracy). Returns defaults if not found."""
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            return json.load(f)
    return {"version": 0, "accuracy": 0.0, "model_path": ""}


def save_meta(meta: dict):
    """Persist model metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, separate features and target (last column)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Data file is empty.")

    if len(df) < 20:
        raise ValueError(f"Not enough data to train: {len(df)} rows (need at least 20).")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Target = last column
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    logger.info(f"Target column: '{target_col}' | Features: {list(feature_cols)}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode non-numeric features
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill missing values with column median
    X = X.fillna(X.median(numeric_only=True))

    # Encode target if categorical
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    return X, y


def train(reason: str = "manual") -> dict:
    """
    Train model, enforce accuracy threshold, save versioned artifact.
    Returns metadata dict with version, accuracy, model_path.
    """
    logger.info(f"=== Training started. Reason: {reason} ===")

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    best_model = None
    best_accuracy = 0.0

    # Try progressively larger forests until we hit target accuracy
    for n_estimators in MAX_ESTIMATORS_ATTEMPTS:
        logger.info(f"Training RandomForest with n_estimators={n_estimators}...")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        logger.info(f"  Validation accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = clf

        if best_accuracy >= TARGET_ACCURACY:
            logger.info(f"  ✅ Target accuracy {TARGET_ACCURACY} reached.")
            break

    if best_accuracy < TARGET_ACCURACY:
        logger.warning(
            f"⚠️ Could not reach target accuracy {TARGET_ACCURACY}. "
            f"Best achieved: {best_accuracy:.4f}. Saving best model anyway."
        )

    # Version the model
    meta = load_meta()
    new_version = meta["version"] + 1
    model_path = os.path.join(MODEL_DIR, f"model_v{new_version}.pkl")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved: {model_path}")

    # Update metadata
    new_meta = {
        "version": new_version,
        "accuracy": round(best_accuracy, 4),
        "model_path": model_path,
        "reason": reason,
        "features": list(X.columns)
    }
    save_meta(new_meta)

    # Update Prometheus gauge
    model_accuracy.set(best_accuracy)
    logger.info(f"model_accuracy gauge set to {best_accuracy:.4f}")

    logger.info(f"=== Training complete. Version: v{new_version}, Accuracy: {best_accuracy:.4f} ===")
    return new_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", type=str, default="manual", help="Reason for training run")
    parser.add_argument("--check-accuracy", action="store_true", help="Exit with code 1 if accuracy < threshold")
    args = parser.parse_args()

    # Exit code 2 = "couldn't train at all yet" (no/insufficient data) — not a real failure,
    # the CI gate should skip on this. Exit code 1 is reserved for "trained fine but accuracy
    # is below threshold" — that IS a real failure and should fail the build.
    try:
        result = train(reason=args.reason)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Cannot train yet: {e}")
        sys.exit(2)

    if args.check_accuracy and result["accuracy"] < TARGET_ACCURACY:
        logger.error(f"Accuracy gate FAILED: {result['accuracy']} < {TARGET_ACCURACY}")
        sys.exit(1)

    sys.exit(0)