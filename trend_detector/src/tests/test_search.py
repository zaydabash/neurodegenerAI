"""Tests for semantic search with keyword fallback."""

from trend_detector.src.pipeline.search import SemanticSearch

POSTS = [
    {"text": "Breakthrough in AI and machine learning", "source": "reddit"},
    {"text": "New vaccine shows promise in clinical trial", "source": "twitter"},
    {"text": "Stock market rallies on tech earnings", "source": "reddit"},
    {"text": "Deep learning models beat human benchmarks", "source": "reddit"},
]


def test_search_returns_relevant_posts():
    searcher = SemanticSearch()
    results = searcher.search("machine learning", POSTS, limit=10)
    assert results, "expected at least one match"
    # Every returned post carries a relevance score.
    assert all("relevance" in post for post in results)


def test_search_ranks_exact_phrase_first():
    searcher = SemanticSearch()
    results = searcher.search("deep learning", POSTS, limit=10)
    assert results
    assert "deep learning" in results[0]["text"].lower()


def test_search_empty_query_returns_empty():
    searcher = SemanticSearch()
    assert searcher.search("", POSTS, limit=10) == []


def test_search_empty_corpus_returns_empty():
    searcher = SemanticSearch()
    assert searcher.search("anything", [], limit=10) == []


def test_search_respects_limit():
    searcher = SemanticSearch()
    results = searcher.search("reddit AI learning market tech", POSTS, limit=2)
    assert len(results) <= 2
