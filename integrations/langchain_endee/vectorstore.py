"""LangChain VectorStore implementation backed by the endee vector database client."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

try:
    import endee  # noqa: F401
except ImportError:
    endee = None  # type: ignore[assignment]


class EndeeVectorStore(VectorStore):
    """LangChain VectorStore wrapper around an endee ``Index``.

    This class exposes ``similarity_search`` and ``similarity_search_with_score``
    methods that delegate to :py:meth:`endee.Index.query`.  Both new parameters
    introduced in endee 0.1.13 — ``prefilter_cardinality_threshold`` and
    ``filter_boost_percentage`` — are surfaced as explicit keyword arguments with
    defaults that match the endee defaults, so all existing callers continue to
    work without modification.

    Args:
        index: A pre-constructed endee Index instance.
        embedding_function: A callable that maps a query string to a list of floats.
        text_key: The metadata key that stores the document's raw text.
            Defaults to ``"text"``.
    """

    def __init__(
        self,
        index: Any,
        embedding_function: Callable[[str], List[float]],
        *,
        text_key: str = "text",
    ) -> None:
        self._index = index
        self._embedding_function = embedding_function
        self._text_key = text_key

    # ------------------------------------------------------------------
    # LangChain VectorStore abstract interface
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> None:
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError("add_texts is not supported for EndeeVectorStore")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        *,
        prefilter_cardinality_threshold: int = 10_000,
        filter_boost_percentage: float = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Return the *k* most similar documents for *query*.

        Args:
            query: The natural-language query string to search for.
            k: Number of results to return. Defaults to ``4``.
            filter: Optional metadata filter expression forwarded to
                :py:meth:`endee.Index.query`. Omitted when ``None``.
            prefilter_cardinality_threshold: Maximum number of vectors considered
                during pre-filtering before switching to post-filtering.
                Valid range is 1 000 – 1 000 000. Defaults to ``10_000``.
            filter_boost_percentage: Percentage boost applied to the relevance
                score of vectors that match the supplied filter.
                Valid range is 0 – 100. Defaults to ``0.0``.
            **kwargs: Additional keyword arguments forwarded to
                :py:meth:`endee.Index.query`.

        Returns:
            List of :class:`~langchain_core.documents.Document` objects ordered
            by descending similarity.
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        *,
        prefilter_cardinality_threshold: int = 10_000,
        filter_boost_percentage: float = 0.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return the *k* most similar documents together with their scores.

        Args:
            query: The natural-language query string to search for.
            k: Number of results to return. Defaults to ``4``.
            filter: Optional metadata filter expression forwarded to
                :py:meth:`endee.Index.query`. Omitted when ``None``.
            prefilter_cardinality_threshold: Maximum number of vectors considered
                during pre-filtering before switching to post-filtering.
                Valid range is 1 000 – 1 000 000. Defaults to ``10_000``.
            filter_boost_percentage: Percentage boost applied to the relevance
                score of vectors that match the supplied filter.
                Valid range is 0 – 100. Defaults to ``0.0``.
            **kwargs: Additional keyword arguments forwarded to
                :py:meth:`endee.Index.query`.

        Returns:
            List of ``(Document, score)`` tuples ordered by descending
            similarity score.
        """
        query_vector = self._embedding_function(query)

        query_kwargs: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": k,
            "include_metadata": True,
            "prefilter_cardinality_threshold": prefilter_cardinality_threshold,
            "filter_boost_percentage": filter_boost_percentage,
            **kwargs,
        }
        if filter is not None:
            query_kwargs["filter"] = filter

        response = self._index.query(**query_kwargs)

        results: List[Tuple[Document, float]] = []
        for match in response:
            metadata = dict(match.get("metadata", {}))
            page_content = metadata.pop(self._text_key, "")
            doc = Document(page_content=page_content, metadata=metadata)
            score: float = match.get("score", 0.0)
            results.append((doc, score))

        return results

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type[EndeeVectorStore],
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> EndeeVectorStore:
        raise NotImplementedError("from_texts is not supported for EndeeVectorStore")

    @classmethod
    def from_documents(
        cls: Type[EndeeVectorStore],
        documents: List[Document],
        embedding: Any,
        **kwargs: Any,
    ) -> EndeeVectorStore:
        raise NotImplementedError(
            "from_documents is not supported for EndeeVectorStore"
        )
