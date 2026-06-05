"""
Semantic search over social posts.

When the sentence-transformers stack is available, queries are ranked by cosine
similarity in embedding space (true semantic search). When it is not available
(or the model fails to load), the searcher degrades gracefully to a keyword
overlap score so the endpoint keeps working in lightweight environments.
"""

from __future__ import annotations

from typing import Any

from shared.lib.logging import get_logger

logger = get_logger(__name__)


class SemanticSearch:
    """Rank social posts against a free-text query."""

    def __init__(self) -> None:
        self._generator: Any | None = None
        self._initialized = False
        self._semantic_available = False

    def _ensure_generator(self) -> Any | None:
        """Lazily load the embedding generator, tolerating a missing ML stack."""
        if self._initialized:
            return self._generator

        self._initialized = True
        try:
            from .embed import EmbeddingGenerator

            self._generator = EmbeddingGenerator()
            self._semantic_available = True
            logger.info("Semantic search using embedding model")
        except Exception as exc:  # noqa: BLE001 - optional heavy dependency
            self._generator = None
            self._semantic_available = False
            logger.warning(
                f"Embedding model unavailable, falling back to keyword search: {exc}"
            )
        return self._generator

    @property
    def is_semantic(self) -> bool:
        """Whether semantic (embedding) ranking is active."""
        self._ensure_generator()
        return self._semantic_available

    def search(
        self,
        query: str,
        posts: list[dict[str, Any]],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return posts ranked by relevance to ``query`` (highest first)."""
        if not query or not posts:
            return []

        generator = self._ensure_generator()
        if generator is not None:
            try:
                return self._semantic_search(generator, query, posts, limit)
            except Exception as exc:  # noqa: BLE001 - fall back on any failure
                logger.warning(f"Semantic search failed, using keyword search: {exc}")

        return self._keyword_search(query, posts, limit)

    def _semantic_search(
        self,
        generator: Any,
        query: str,
        posts: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        texts = [post.get("text", "") for post in posts]
        query_embedding = generator.embed_single_text(query)
        candidate_embeddings = generator.embed_texts(texts)

        ranked = generator.find_similar_texts(
            query_embedding, candidate_embeddings, top_k=limit, threshold=0.0
        )

        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            post = dict(posts[idx])
            post["relevance"] = round(float(score), 4)
            results.append(post)
        return results

    def _keyword_search(
        self, query: str, posts: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        query_terms = {term for term in query.lower().split() if term}
        scored: list[tuple[float, dict[str, Any]]] = []

        for post in posts:
            text = post.get("text", "").lower()
            if not text:
                continue
            overlap = sum(1 for term in query_terms if term in text)
            if query.lower() in text:
                overlap += len(query_terms)  # boost exact phrase matches
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                enriched = dict(post)
                enriched["relevance"] = round(min(score, 1.0), 4)
                scored.append((score, enriched))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [post for _, post in scored[:limit]]
