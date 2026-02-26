"""
Tests for the LangChain endee integration (integrations/langchain_endee/vectorstore.py).

Verifies that:
  1. similarity_search works without the new parameters (backward-compat).
  2. prefilter_cardinality_threshold is forwarded to index.query().
  3. filter_boost_percentage is forwarded to index.query().
  4. Filters are passed through correctly.
  5. similarity_search_with_score returns (Document, float) tuples.
  6. add_texts / from_texts raise NotImplementedError.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integrations.langchain_endee.vectorstore import EndeeVectorStore


@pytest.fixture()
def mock_index():
    idx = MagicMock()
    idx.query.return_value = [
        {"id": "a", "score": 0.9, "metadata": {"text": "doc one"}},
        {"id": "b", "score": 0.8, "metadata": {"text": "doc two"}},
    ]
    return idx


@pytest.fixture()
def embed_fn():
    return lambda text: [0.1, 0.2, 0.3]


@pytest.fixture()
def store(mock_index, embed_fn):
    return EndeeVectorStore(index=mock_index, embedding_function=embed_fn)


# ── Backward compatibility ────────────────────────────────────────────────────

def test_similarity_search_returns_documents(store, mock_index):
    docs = store.similarity_search("hello", k=2)

    assert len(docs) == 2
    assert docs[0].page_content == "doc one"
    assert docs[1].page_content == "doc two"


def test_default_parameters_forwarded(store, mock_index):
    store.similarity_search("hello", k=3)

    _, kwargs = mock_index.query.call_args
    assert kwargs["top_k"] == 3
    assert kwargs["prefilter_cardinality_threshold"] == 10_000
    assert kwargs["filter_boost_percentage"] == 0


# ── New parameter: prefilter_cardinality_threshold ────────────────────────────

def test_prefilter_threshold_forwarded(store, mock_index):
    store.similarity_search("test", k=4, prefilter_cardinality_threshold=25_000)

    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 25_000


# ── New parameter: filter_boost_percentage ────────────────────────────────────

def test_filter_boost_forwarded(store, mock_index):
    store.similarity_search("test", k=4, filter_boost_percentage=30)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter_boost_percentage"] == 30


# ── Filter handling ───────────────────────────────────────────────────────────

def test_filter_passed_through(store, mock_index):
    flt = [{"category": {"$eq": "B"}}]
    store.similarity_search("test", k=2, filter=flt)

    _, kwargs = mock_index.query.call_args
    assert kwargs["filter"] == flt


def test_no_filter_key_when_none(store, mock_index):
    store.similarity_search("test", k=2, filter=None)
    _, kwargs = mock_index.query.call_args
    assert "filter" not in kwargs


# ── similarity_search_with_score ──────────────────────────────────────────────

def test_similarity_search_with_score(store, mock_index):
    results = store.similarity_search_with_score("hello", k=2)

    assert len(results) == 2
    doc, score = results[0]
    assert doc.page_content == "doc one"
    assert score == 0.9


def test_similarity_search_with_score_new_params(store, mock_index):
    store.similarity_search_with_score(
        "hello",
        k=2,
        prefilter_cardinality_threshold=5_000,
        filter_boost_percentage=10,
    )
    _, kwargs = mock_index.query.call_args
    assert kwargs["prefilter_cardinality_threshold"] == 5_000
    assert kwargs["filter_boost_percentage"] == 10


# ── Not-implemented stubs ─────────────────────────────────────────────────────

def test_add_texts_raises(store):
    with pytest.raises(NotImplementedError):
        store.add_texts(["some text"])


def test_from_texts_raises(embed_fn):
    with pytest.raises(NotImplementedError):
        EndeeVectorStore.from_texts(["text"], embed_fn)
