"""
LlamaIndex VectorStore integration for the endee vector database client.

Wraps endee's Index.query() as a LlamaIndex-compatible BasePydanticVectorStore
so it integrates cleanly into any LlamaIndex retrieval pipeline.

Version compatibility: endee >= 0.1.13, llama-index-core >= 0.11
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

# llama_index is an optional dependency
try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError:
    # Minimal stubs so the module is importable without llama-index installed
    class TextNode:  # type: ignore[no-redef]
        def __init__(self, text: str = "", metadata: dict | None = None):
            self.text = text
            self.metadata = metadata or {}

    class BaseNode(TextNode):  # type: ignore[no-redef]
        pass

    class VectorStoreQuery:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class VectorStoreQueryResult:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BasePydanticVectorStore:  # type: ignore[no-redef]
        stores_text: bool = True


class EndeeVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex VectorStore backed by an endee vector index.

    Usage::

        from endee import Index
        from integrations.llamaindex_endee import EndeeVectorStore
        from llama_index.core import VectorStoreIndex

        index        = Index(name="my-index", url="https://...", api_key="...")
        vector_store = EndeeVectorStore(index=index)
        li_index     = VectorStoreIndex.from_vector_store(vector_store)
        retriever    = li_index.as_retriever(similarity_top_k=5)

    To use the new endee 0.1.13 filter parameters, pass them either directly
    to ``query()`` or via ``VectorStoreQuery.query_kwargs``::

        from llama_index.core.vector_stores.types import VectorStoreQuery

        q = VectorStoreQuery(
            query_embedding=[...],
            similarity_top_k=5,
            query_kwargs={
                "prefilter_cardinality_threshold": 5_000,
                "filter_boost_percentage": 20,
            },
        )
        result = vector_store.query(q)
    """

    stores_text: bool = True

    def __init__(self, index: Any) -> None:
        self._index = index

    # ── Core query method ─────────────────────────────────────────────────────

    def query(
        self,
        query: VectorStoreQuery,
        prefilter_cardinality_threshold: int = 10_000,
        filter_boost_percentage: int = 0,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Run a similarity query against the endee index.

        Args:
            query: LlamaIndex VectorStoreQuery containing the embedding and
                   retrieval parameters.
            prefilter_cardinality_threshold: Controls when endee switches from
                HNSW filtered search to brute-force prefiltering on the matched
                subset. When the number of vectors matching the active filter is
                at or below this value, brute-force prefiltering is used instead
                of HNSW filtered search.
                Valid range: 1,000 – 1,000,000 (default: 10,000).
                A lower value (e.g. 1,000) applies prefiltering only for highly
                selective filters; a higher value (e.g. 1,000,000) applies it
                for almost all filtered queries.
                Can also be supplied via ``query.query_kwargs`` as
                ``"prefilter_cardinality_threshold"``.
            filter_boost_percentage: Expands the internal HNSW candidate pool
                by this percentage when a filter is active, compensating for
                candidates that are traversed but then eliminated by the filter.
                For example, a value of 20 instructs the engine to fetch 20%
                more candidates before applying the filter, improving recall at
                the cost of slightly higher per-query latency.
                Valid range: 0 – 100 (default: 0). A value of 0 disables the
                boost entirely, preserving the existing behavior.
                Can also be supplied via ``query.query_kwargs`` as
                ``"filter_boost_percentage"``.

        Returns:
            VectorStoreQueryResult with matched TextNode objects, similarity
            scores, and document IDs.
        """
        # Allow callers to override thresholds via query_kwargs without
        # changing the VectorStoreQuery API.
        query_kwargs: dict[str, Any] = getattr(query, "query_kwargs", {}) or {}
        prefilter_cardinality_threshold = query_kwargs.get(
            "prefilter_cardinality_threshold", prefilter_cardinality_threshold
        )
        filter_boost_percentage = query_kwargs.get(
            "filter_boost_percentage", filter_boost_percentage
        )

        endee_kwargs: dict[str, Any] = dict(
            vector=query.query_embedding,
            top_k=getattr(query, "similarity_top_k", 10),
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
        )

        # Map LlamaIndex MetadataFilters → endee filter list
        li_filters = getattr(query, "filters", None)
        if li_filters is not None:
            endee_kwargs["filter"] = self._convert_filters(li_filters)

        results = self._index.query(**endee_kwargs)

        nodes: list[TextNode] = []
        scores: list[float] = []
        ids: list[str] = []

        for r in results:
            metadata = r.get("metadata", {})
            node = TextNode(
                text=metadata.get("text", ""),
                metadata=metadata,
            )
            nodes.append(node)
            scores.append(r.get("score", 0.0))
            ids.append(r.get("id", ""))

        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    # ── Ingestion stubs ───────────────────────────────────────────────────────

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        raise NotImplementedError(
            "EndeeVectorStore.add() is not supported. "
            "Use the endee client directly for ingestion."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError(
            "EndeeVectorStore.delete() is not supported. "
            "Use the endee client directly for deletion."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _convert_filters(li_filters: Any) -> list[dict]:
        """
        Convert a LlamaIndex MetadataFilters object into endee filter syntax.

        LlamaIndex: MetadataFilter(key="category", value="A", operator="==")
        endee:      [{"category": {"$eq": "A"}}]
        """
        _OP_MAP = {
            "==": "$eq",
            "!=": "$ne",
            ">": "$gt",
            ">=": "$gte",
            "<": "$lt",
            "<=": "$lte",
            "in": "$in",
            "nin": "$nin",
        }

        converted: list[dict] = []
        for f in getattr(li_filters, "filters", []):
            op = _OP_MAP.get(str(getattr(f, "operator", "==")), "$eq")
            converted.append({f.key: {op: f.value}})
        return converted