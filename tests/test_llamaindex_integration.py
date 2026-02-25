"""
Tests for the LlamaIndex endee integration (integrations/llamaindex_endee/vector_store.py).

Verifies that:
  1. query() works without the new parameters (backward-compat).
  2. prefilter_cardinality_threshold is forwarded to index.query().
  3. filter_boost_percentage is forwarded to index.query().
  4. query_kwargs overrides are respected.
  5. LlamaIndex MetadataFilters are converted to endee filter syntax.
  6. VectorStoreQueryResult has nodes with correct text content.
  7. add() / delete() raise NotImplementedError.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from integrations.llamaindex_endee.vector_store import EndeeVectorStore


@pytest.fixture()
def mock_index():
    idx = MagicMock()
    idx.query.return_value = [
        {"id": "x1", "score": 0.85, "metadata": {"text": "result one"}},
        {"id": "x2", "score": 0.75, "metadata": {"text": "result two"}},
    ]
    return idx


@pytest.fixture()
def store(mock_index):
    return EndeeVectorStore(index=mock_index)


def _make_query(embedding=None, top_k=5, filters=None, query_kwargs=None):
    """Build a minimal VectorStoreQuery-like object."""
    q = SimpleNamespace(
        query_embedding=embedding or [0.1, 0.2, 0.3],
        similarity_top_k=top_k,
        filters=filters,
        query_kwargs=query_kwargs or {},
    )
    return q


# ── Backward compatibility ────────────────────────────────────────────────────

def test_query_returns_result(store, mock_index):
    q = _make_query()
    result = store.query(q)

    assert len(result.nodes) == 2
    assert result.nodes[0].text == "result one"
    assert result.similarities[0] == 0.85


def test_default_parameters_forwarded(store, mock_index):
    q = _make_query(top_k=7)
    store.query(q)

    _, kwargs = mock_index.query.call_args
    assert kwargs["top_k"] == 7
    assert kwargs["prefilter_cardinality_threshold"] == 10_000
    assert kwargs["filter_boost_percentage"] == 0


# ── New parameter: prefilter_cardinality_threshold ────────────────────────────

def test_prefilter_threshold_via_direct_arg(store, mock_index):
    q = _make_query()
    store.query(q, prefilter_cardinality_threshold=20_000)

    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 20_000


def test_prefilter_threshold_via_query_kwargs(store, mock_index):
    q = _make_query(query_kwargs={"prefilter_cardinality_threshold": 75_000})
    store.query(q)

    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 75_000


def test_query_kwargs_overrides_direct_arg(store, mock_index):
    """query_kwargs should take precedence over the direct argument."""
    q = _make_query(query_kwargs={"prefilter_cardinality_threshold": 50_000})
    store.query(q, prefilter_cardinality_threshold=10_000)

    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 50_000


# ── New parameter: filter_boost_percentage ────────────────────────────────────

def test_filter_boost_via_direct_arg(store, mock_index):
    q = _make_query()
    store.query(q, filter_boost_percentage=15)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter_boost_percentage"] == 15


def test_filter_boost_via_query_kwargs(store, mock_index):
    q = _make_query(query_kwargs={"filter_boost_percentage": 40})
    store.query(q)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter_boost_percentage"] == 40


# ── Filter conversion ─────────────────────────────────────────────────────────

def test_no_filter_when_none(store, mock_index):
    q = _make_query(filters=None)
    store.query(q)
    _, kwargs = mock_index.query.call_args
    assert "filter" not in kwargs


def test_metadata_filter_converted(store, mock_index):
    """LlamaIndex MetadataFilter → endee filter list."""
    li_filter = SimpleNamespace(
        filters=[
            SimpleNamespace(key="category", value="A", operator="=="),
            SimpleNamespace(key="year", value=2024, operator=">="),
        ]
    )
    q = _make_query(filters=li_filter)
    store.query(q)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter"] == [
        {"category": {"$eq": "A"}},
        {"year": {"$gte": 2024}},
    ]


# ── Result structure ──────────────────────────────────────────────────────────

def test_result_ids_populated(store, mock_index):
    q = _make_query()
    result = store.query(q)
    assert result.ids == ["x1", "x2"]


# ── Not-implemented stubs ─────────────────────────────────────────────────────

def test_add_raises(store):
    with pytest.raises(NotImplementedError):
        store.add([])


def test_delete_raises(store):
    with pytest.raises(NotImplementedError):
        store.delete("some-doc-id")
