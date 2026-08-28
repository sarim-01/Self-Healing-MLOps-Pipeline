"""
Distribution drift detection using z-score comparison.
Compares per-feature mean/std of current batch against a stored baseline.
"""

import json
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

BASELINE_PATH = "data/stats_baseline.json"
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "2.0"))


def compute_stats(records: list[dict]) -> dict:
    """Compute mean and std for each numeric feature in a batch of records."""
    if not records:
        return {}

    stats = {}
    keys = records[0].keys()

    for key in keys:
        values = []
        for r in records:
            try:
                values.append(float(r[key]))
            except (ValueError, TypeError):
                continue  # skip non-numeric

        if values:
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else 0.0,
                "count": len(values)
            }

    return stats


def load_baseline() -> dict:
    """Load saved baseline stats from disk. Returns empty dict if not found."""
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_baseline(stats: dict):
    """Persist current stats as the new baseline."""
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def check_drift(records: list[dict], baseline: dict) -> tuple[bool, list[str]]:
    """
    Compare current batch stats against baseline using z-score.

    Returns:
        drift_detected (bool): True if any feature exceeds threshold
        drifted_features (list): Names of features that drifted
    """
    if not baseline:
        logger.info("No baseline yet — skipping drift check, saving current as baseline.")
        current_stats = compute_stats(records)
        save_baseline(current_stats)
        return False, []

    current_stats = compute_stats(records)
    drifted_features = []

    for feature, curr in current_stats.items():
        if feature not in baseline:
            continue  # new feature — handled by schema monitor

        base = baseline[feature]
        base_std = base["std"]

        if base_std == 0:
            # if baseline had no variance, any change is drift
            if curr["mean"] != base["mean"]:
                logger.warning(f"Drift detected in '{feature}': baseline std=0, mean changed.")
                drifted_features.append(feature)
            continue

        z_score = abs(curr["mean"] - base["mean"]) / base_std

        if z_score > DRIFT_THRESHOLD:
            logger.warning(
                f"Drift detected in '{feature}': "
                f"z-score={z_score:.2f} (threshold={DRIFT_THRESHOLD}), "
                f"baseline_mean={base['mean']:.3f}, current_mean={curr['mean']:.3f}"
            )
            drifted_features.append(feature)

    drift_detected = len(drifted_features) > 0

    if drift_detected:
        logger.warning(f"Drift detected in {len(drifted_features)} feature(s): {drifted_features}")
    else:
        logger.info("No distribution drift detected.")

    # Update baseline with current stats after check
    save_baseline(current_stats)

    return drift_detected, drifted_features