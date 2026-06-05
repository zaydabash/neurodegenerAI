"""
Live topic clustering over a corpus of social posts.

Uses BERTopic when it is importable (as advertised); otherwise falls back to a
genuine scikit-learn pipeline (TF-IDF + KMeans with per-cluster top terms), so
``/v1/trends/top`` always returns topics actually computed from the corpus
rather than a canned list.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from shared.lib.logging import get_logger

logger = get_logger(__name__)


def cluster_posts(posts: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
    """Cluster posts into trending topics, highest score first."""
    texts = [p.get("text", "") for p in posts if p.get("text")]
    if len(texts) < 3:
        return []

    try:
        return _cluster_bertopic(posts, texts, k)
    except Exception as exc:  # noqa: BLE001 - BERTopic is optional/heavy
        logger.info(f"BERTopic unavailable, using scikit-learn clustering: {exc}")
        return _cluster_sklearn(posts, texts, k)


def _cluster_bertopic(
    posts: list[dict[str, Any]], texts: list[str], k: int
) -> list[dict[str, Any]]:
    from bertopic import BERTopic

    model = BERTopic(nr_topics=k, calculate_probabilities=False, verbose=False)
    assignments, _ = model.fit_transform(texts)

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, topic_id in enumerate(assignments):
        if topic_id != -1:  # skip outliers
            grouped[topic_id].append(idx)

    results = []
    for topic_id, members in grouped.items():
        keywords = [w for w, _ in model.get_topic(topic_id)[:8]]
        label = keywords[0] if keywords else f"topic_{topic_id}"
        results.append(_build_topic(label, keywords, members, posts, texts, len(texts)))
    results.sort(key=lambda t: t["trending_score"], reverse=True)
    return results[:k]


def _cluster_sklearn(
    posts: list[dict[str, Any]], texts: list[str], k: int
) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_clusters = max(1, min(k, len(texts) // 2 if len(texts) >= 4 else 1))
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=500, ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(matrix)

    results = []
    for cluster_id in range(n_clusters):
        members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not members:
            continue
        centroid = km.cluster_centers_[cluster_id]
        top_term_idx = centroid.argsort()[::-1][:8]
        keywords = [terms[i] for i in top_term_idx if centroid[i] > 0]
        label = keywords[0] if keywords else f"topic_{cluster_id}"
        results.append(_build_topic(label, keywords, members, posts, texts, len(texts)))

    results.sort(key=lambda t: t["trending_score"], reverse=True)
    return results[:k]


def _build_topic(
    label: str,
    keywords: list[str],
    members: list[int],
    posts: list[dict[str, Any]],
    texts: list[str],
    total: int,
) -> dict[str, Any]:
    volume = len(members)
    trending_score = round(min(volume / max(total, 1), 1.0), 4)
    growth_rate = _growth_rate([posts[i] for i in members])
    representative = [texts[i] for i in members[:3]]
    return {
        "topic": label,
        "keywords": keywords or [label],
        "trending_score": trending_score,
        "volume": volume,
        "growth_rate": growth_rate,
        "representative_posts": representative,
    }


def _growth_rate(cluster_posts: list[dict[str, Any]]) -> float:
    """Fraction of recent vs older posts as a simple growth proxy."""
    times = []
    for post in cluster_posts:
        ts = post.get("timestamp")
        if isinstance(ts, str):
            try:
                times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                continue
    if len(times) < 2:
        return 0.0
    times.sort()
    midpoint = times[len(times) // 2]
    recent = sum(1 for t in times if t >= midpoint)
    older = max(len(times) - recent, 1)
    return round((recent - older) / len(times), 4)
