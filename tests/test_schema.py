"""
Unit tests for schema change detection in ingestion.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.ingestion import detect_schema_changes


def test_feature_added():
    """Test that new features are correctly detected."""
    old_schema = ["age", "salary", "city"]
    new_schema = ["age", "salary", "city", "score"]

    added, removed = detect_schema_changes(old_schema, new_schema)

    assert "score" in added
    assert len(removed) == 0


def test_feature_removed():
    """Test that removed features are correctly detected."""
    old_schema = ["age", "salary", "city", "score"]
    new_schema = ["age", "salary", "city"]

    added, removed = detect_schema_changes(old_schema, new_schema)

    assert "score" in removed
    assert len(added) == 0


def test_no_schema_change():
    """Test that identical schemas produce no changes."""
    old_schema = ["age", "salary", "city"]
    new_schema = ["age", "salary", "city"]

    added, removed = detect_schema_changes(old_schema, new_schema)

    assert len(added) == 0
    assert len(removed) == 0


def test_multiple_changes():
    """Test detection of multiple simultaneous schema changes."""
    old_schema = ["age", "salary", "city"]
    new_schema = ["age", "score", "region"]

    added, removed = detect_schema_changes(old_schema, new_schema)

    assert "score" in added
    assert "region" in added
    assert "salary" in removed
    assert "city" in removed