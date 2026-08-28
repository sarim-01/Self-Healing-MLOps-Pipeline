"""
Unit tests for distribution drift detection in drift_detector.py
"""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.drift_detector import compute_stats, check_drift


def test_compute_stats():
    """Test that stats are computed correctly."""
    records = [
        {"age": 25, "salary": 50000},
        {"age": 30, "salary": 60000},
        {"age": 35, "salary": 70000},
    ]
    stats = compute_stats(records)

    assert "age" in stats
    assert "salary" in stats
    assert stats["age"]["mean"] == 30.0
    assert stats["salary"]["mean"] == 60000.0


def test_drift_detected():
    """Test that significant distribution shift is flagged as drift."""
    # Baseline: age around 25-35
    baseline_records = [{"age": float(i)} for i in range(25, 36)]
    baseline_stats = compute_stats(baseline_records)

    # Drifted: age around 80-90 (completely different distribution)
    drifted_records = [{"age": float(i)} for i in range(80, 91)]

    drift_detected, drifted_features = check_drift(drifted_records, baseline_stats)

    assert drift_detected is True
    assert "age" in drifted_features


def test_no_drift():
    """Test that similar distributions do not trigger drift."""
    # Baseline
    baseline_records = [{"age": float(i)} for i in range(25, 36)]
    baseline_stats = compute_stats(baseline_records)

    # Similar distribution — slight variation only
    similar_records = [{"age": float(i)} for i in range(26, 37)]

    drift_detected, drifted_features = check_drift(similar_records, baseline_stats)

    assert drift_detected is False
    assert len(drifted_features) == 0


def test_empty_baseline_no_drift():
    """Test that empty baseline skips drift check gracefully."""
    records = [{"age": 25.0}, {"age": 30.0}]
    drift_detected, drifted_features = check_drift(records, {})

    assert drift_detected is False
    assert len(drifted_features) == 0