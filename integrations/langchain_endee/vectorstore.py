"""LangChain VectorStore implementation backed by the endee vector database client."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

import endee


class EndeeVectorStore(VectorStore):
    """LangChain VectorStore wrapper around an endee ``Index``.

    This class exposes ``similarity_search`` and ``similarity_search_with_score``
    methods that delegate to :py:meth:`endee.Index.query`.  Both new parameters
    introduced in endee 0.1.13 — ``prefilter_cardinality_threshold`` and
    ``filter_boost_percentage`` — are surfaced as explicit keyword arguments with
    defaults that match the endee defaults, so all existing callers continue to
    work without modification.

    Args:
        index: A pre-constructed :class:`endee.Index` instance.
        embedding: A LangChain :class:`~langchain_core.embeddings.Embeddings`
            implementation used to embed query strings.
        text_key: The metadata key that stores the document's raw text.
            Defaults to ``"text"``.
    """

    def __init__(
        self,
        index: endee.Index,
        embedding: Embeddings,
        *,
        text_key: str = "text",
    ) -> None:
        self._index = index
        self._embedding = embedding
        self._text_key = text_key

    # ------------------------------------------------------------------
    # LangChain VectorStore abstract interface
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed *texts* and upsert them into the endee index.

        Args:
            texts: Iterable of raw text strings to index.
            metadatas: Optional list of metadata dicts aligned with *texts*.
            **kwargs: Extra arguments forwarded to :py:meth:`endee.Index.upsert`.

        Returns:
            List of vector IDs assigned by endee.
        """
        text_list = list(texts)
        vectors = self._embedding.embed_documents(text_list)

        if metadatas is None:
            metadatas = [{} for _ in text_list]

        items = [
            {
                "id": meta.get("id", str(i)),
                "values": vec,
                "metadata": {self._text_key: text, **meta},
            }
            for i, (text, vec, meta) in enumerate(zip(text_list, vectors, metadatas))
        ]
        response = self._index.upsert(vectors=items, **kwargs)
        return [item["id"] for item in items]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
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
                :py:meth:`endee.Index.query`.
            prefilter_cardinality_threshold: Maximum number of vectors considered
                during pre-filtering before switching to post-filtering.  Valid
                range is 1 000 – 1 000 000.  Defaults to ``10_000`` (endee
                default).
            filter_boost_percentage: Percentage boost applied to the relevance
                score of vectors that match the supplied filter.  Valid range is
                0 – 100.  Defaults to ``0.0`` (endee default, no boost).
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
        filter: Optional[Dict[str, Any]] = None,
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
                :py:meth:`endee.Index.query`.
            prefilter_cardinality_threshold: Maximum number of vectors considered
                during pre-filtering before switching to post-filtering.  Valid
                range is 1 000 – 1 000 000.  Defaults to ``10_000`` (endee
                default).
            filter_boost_percentage: Percentage boost applied to the relevance
                score of vectors that match the supplied filter.  Valid range is
                0 – 100.  Defaults to ``0.0`` (endee default, no boost).
            **kwargs: Additional keyword arguments forwarded to
                :py:meth:`endee.Index.query`.

        Returns:
            List of ``(Document, score)`` tuples ordered by descending
            similarity score.
        """
        query_vector = self._embedding.embed_query(query)

        response = self._index.query(
            vector=query_vector,
            top_k=k,
            filter=filter,
            include_metadata=True,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
            **kwargs,
        )

        results: List[Tuple[Document, float]] = []
        for match in response.get("matches", []):
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
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        index: Optional[endee.Index] = None,
        text_key: str = "text",
        **kwargs: Any,
    ) -> EndeeVectorStore:
        """Create an :class:`EndeeVectorStore` from a list of raw texts.

        Args:
            texts: Raw text strings to embed and index.
            embedding: Embeddings implementation for encoding the texts.
            metadatas: Optional list of metadata dicts aligned with *texts*.
            index: An existing :class:`endee.Index` to upsert into.  Required
                unless the subclass overrides index creation.
            text_key: Metadata key used to store raw text.  Defaults to
                ``"text"``.
            **kwargs: Extra arguments forwarded to :py:meth:`add_texts`.

        Returns:
            A populated :class:`EndeeVectorStore` instance.

        Raises:
            ValueError: If *index* is not provided.
        """
        if index is None:
            raise ValueError(
                "An endee.Index instance must be supplied via the 'index' argument."
            )
        store = cls(index=index, embedding=embedding, text_key=text_key)
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls: Type[EndeeVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        index: Optional[endee.Index] = None,
        text_key: str = "text",
        **kwargs: Any,
    ) -> EndeeVectorStore:
        """Create an :class:`EndeeVectorStore` from a list of LangChain documents.

        Args:
            documents: :class:`~langchain_core.documents.Document` objects to
                embed and index.
            embedding: Embeddings implementation for encoding document content.
            index: An existing :class:`endee.Index` to upsert into.  Required
                unless the subclass overrides index creation.
            text_key: Metadata key used to store raw text.  Defaults to
                ``"text"``.
            **kwargs: Extra arguments forwarded to :py:meth:`add_texts`.

        Returns:
            A populated :class:`EndeeVectorStore` instance.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            index=index,
            text_key=text_key,
            **kwargs,
        )
