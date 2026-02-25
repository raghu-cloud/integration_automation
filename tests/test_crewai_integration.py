"""
Tests for the CrewAI endee integration (integrations/crewai_endee/tools.py).

Verifies that:
  1. Basic search works without the new parameters (backward-compat).
  2. prefilter_cardinality_threshold is accepted and forwarded.
  3. filter_boost_percentage is accepted and forwarded.
  4. Filters are passed correctly.
  5. Input validation rejects out-of-range values.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integrations.crewai_endee.tools import EndeeSearchInput, EndeeSearchTool


@pytest.fixture()
def mock_index():
    idx = MagicMock()
    idx.query.return_value = [
        {"id": "1", "score": 0.95, "metadata": {"text": "hello world"}}
    ]
    return idx


@pytest.fixture()
def tool(mock_index):
    # Bypass __init__ because the fallback BaseTool stub (used when crewai_tools
    # is not installed) has no __init__ and rejects keyword arguments.
    # The only attribute _run() needs is self.index, so we set it directly.
    t = object.__new__(EndeeSearchTool)
    t.index = mock_index
    return t


# ── Backward compatibility ────────────────────────────────────────────────────

def test_basic_search_no_filter(tool, mock_index):
    """query() is called with sensible defaults when no filter is given."""
    result = tool._run(query_vector=[0.1, 0.2, 0.3], top_k=5)

    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        prefilter_cardinality_threshold=10_000,
        filter_boost_percentage=0,
    )
    assert len(result) == 1
    assert result[0]["score"] == 0.95


# ── New parameter: prefilter_cardinality_threshold ────────────────────────────

def test_prefilter_threshold_forwarded(tool, mock_index):
    """prefilter_cardinality_threshold is forwarded to index.query()."""
    tool._run(query_vector=[0.1], top_k=3, prefilter_cardinality_threshold=50_000)

    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 50_000


def test_prefilter_threshold_minimum(tool, mock_index):
    """Value of 1_000 (minimum) is accepted."""
    tool._run(query_vector=[0.1], top_k=3, prefilter_cardinality_threshold=1_000)
    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 1_000


def test_prefilter_threshold_maximum(tool, mock_index):
    """Value of 1_000_000 (maximum) is accepted."""
    tool._run(query_vector=[0.1], top_k=3, prefilter_cardinality_threshold=1_000_000)
    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 1_000_000


# ── New parameter: filter_boost_percentage ────────────────────────────────────

def test_filter_boost_forwarded(tool, mock_index):
    """filter_boost_percentage is forwarded to index.query()."""
    tool._run(query_vector=[0.1], top_k=3, filter_boost_percentage=20)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter_boost_percentage"] == 20


def test_filter_boost_zero(tool, mock_index):
    """Default boost of 0 is forwarded correctly."""
    tool._run(query_vector=[0.1], top_k=3)
    _, kwargs = mock_index.query.call_args
    assert kwargs["filter_boost_percentage"] == 0


# ── Filter handling ───────────────────────────────────────────────────────────

def test_filter_passed_through(tool, mock_index):
    """Metadata filter list is forwarded to index.query()."""
    flt = [{"category": {"$eq": "A"}}]
    tool._run(query_vector=[0.1], top_k=3, filter=flt)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter"] == flt


def test_no_filter_key_when_none(tool, mock_index):
    """'filter' key is absent from index.query() call when filter=None."""
    tool._run(query_vector=[0.1], top_k=3, filter=None)
    _, kwargs = mock_index.query.call_args
    assert "filter" not in kwargs


# ── Input schema validation ───────────────────────────────────────────────────

def test_schema_rejects_threshold_below_min():
    with pytest.raises(Exception):
        EndeeSearchInput(query_vector=[0.1], prefilter_cardinality_threshold=999)


def test_schema_rejects_threshold_above_max():
    with pytest.raises(Exception):
        EndeeSearchInput(query_vector=[0.1], prefilter_cardinality_threshold=1_000_001)


def test_schema_rejects_boost_above_100():
    with pytest.raises(Exception):
        EndeeSearchInput(query_vector=[0.1], filter_boost_percentage=101)


def test_schema_rejects_boost_below_zero():
    with pytest.raises(Exception):
        EndeeSearchInput(query_vector=[0.1], filter_boost_percentage=-1)
