"""Finds the subtitle chunks that are relevant to a question (hybrid search)."""

import math
import re
from dataclasses import replace

from chromadb.api.models.Collection import Collection

from config.settings import settings
from rag.embeddings import embed_query, embed_texts
from rag.vector_store import (
    SearchResult,
    get_collection,
    list_movie_titles,
    search_chunks,
)

# Results with a lower similarity score than this are treated as "not relevant".
# Tune it by looking at the scores printed by scripts.search_demo.
MIN_RELEVANCE_SCORE = 0.20

POOL_MULTIPLIER = 4  # each search looks at 4x as many chunks as we finally keep
RRF_K = 60  # standard constant for rank fusion

LEADING_WORDS = ("the ", "a ", "an ")

# Very common words that say nothing about the topic of a question.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "what", "which", "who", "whom",
    "when", "where", "why", "how", "does", "did", "this", "that", "these",
    "those", "with", "about", "from", "into", "say", "says", "said", "tell",
    "tells", "told", "there", "their", "them", "they", "you", "your", "his",
    "her", "him", "she", "has", "have", "had", "not", "but", "can", "could",
    "would", "should", "will", "any", "all", "some", "movie", "film", "scene",
}


def clean_query(question: str) -> str:
    """Remove extra spaces and line breaks."""
    return " ".join(question.split())


def _normalize(text: str) -> str:
    """Lowercase and remove punctuation so titles can be compared."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _title_key(title: str) -> str:
    """'The Lion King' -> 'lion king' (people often skip the leading 'The')."""
    key = _normalize(title)
    for word in LEADING_WORDS:
        if key.startswith(word):
            return key[len(word) :]
    return key


def detect_movie_title(question: str, titles: list[str]) -> str | None:
    """Return the movie title mentioned in the question.

    Returns None if no title, or more than one title, is mentioned.
    """
    padded_question = f" {_normalize(question)} "
    matches = [t for t in titles if f" {_title_key(t)} " in padded_question]
    return matches[0] if len(matches) == 1 else None


def find_movie_in_question(
    question: str, collection: Collection | None = None
) -> str | None:
    """Look up the stored titles and detect which one the question mentions."""
    return detect_movie_title(question, list_movie_titles(collection))


def filter_relevant(
    results: list[SearchResult], min_score: float = MIN_RELEVANCE_SCORE
) -> list[SearchResult]:
    """Drop weak matches."""
    return [result for result in results if result.score >= min_score]


# ---------- keyword search ----------


def stem(word: str) -> str:
    """A very simple stemmer: 'stars' -> 'star', 'fingers' -> 'finger'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(question: str) -> list[str]:
    """The words of the question that carry meaning (no common words)."""
    words: list[str] = []
    for raw in re.findall(r"[a-z0-9]+", question.lower()):
        if len(raw) <= 2 or raw in STOPWORDS:
            continue
        word = stem(raw)
        if word not in STOPWORDS:
            words.append(word)
    return list(dict.fromkeys(words))  # remove duplicates, keep order


def keyword_scores(words: list[str], texts: list[str]) -> list[float]:
    """Score each text by the question words it contains. Rare words count more."""
    token_sets = [{stem(w) for w in re.findall(r"[a-z0-9]+", t.lower())} for t in texts]
    total = len(texts)
    scores = [0.0] * total

    for word in words:
        containing = sum(1 for tokens in token_sets if word in tokens)
        if containing == 0:
            continue
        weight = math.log(total / containing)  # 0 if every text has the word
        for index, tokens in enumerate(token_sets):
            if word in tokens:
                scores[index] += weight
    return scores


def _result_from_stored(text: str, meta: dict, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=str(meta["chunk_id"]),
        movie_id=str(meta["movie_id"]),
        movie_title=str(meta["movie_title"]),
        start_time=str(meta["start_time"]),
        end_time=str(meta["end_time"]),
        subtitle_start_index=int(meta["subtitle_start_index"]),
        subtitle_end_index=int(meta["subtitle_end_index"]),
        text=text,
        score=score,
    )


def keyword_search(
    question: str,
    movie_title: str | None = None,
    collection: Collection | None = None,
    limit: int = 30,
) -> list[SearchResult]:
    """Find chunks that contain the question's rare words (like Ctrl+F)."""
    words = content_words(question)
    if not words:
        return []

    collection = get_collection() if collection is None else collection
    where = {"movie_title": movie_title} if movie_title else None
    data = collection.get(where=where, include=["documents", "metadatas"])

    texts = data["documents"]
    scores = keyword_scores(words, texts)
    ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    return [
        _result_from_stored(texts[i], data["metadatas"][i], 0.0)
        for i in ranked[:limit]
        if scores[i] > 0
    ]


def fuse_rankings(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Merge several ranked lists of chunk IDs (Reciprocal Rank Fusion).

    A chunk earns 1 / (k + position) points from every list it appears in.
    """
    points: dict[str, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            points[chunk_id] = points.get(chunk_id, 0.0) + 1.0 / (k + position)
    return sorted(points, key=lambda chunk_id: points[chunk_id], reverse=True)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------- the main function ----------


def retrieve(
    question: str,
    movie_title: str | None = None,
    top_k: int | None = None,
    min_score: float = MIN_RELEVANCE_SCORE,
    collection: Collection | None = None,
) -> list[SearchResult]:
    """Hybrid search: meaning + keywords, merged, then weak matches dropped."""
    query = clean_query(question)
    top_k = top_k or settings.top_k
    pool = top_k * POOL_MULTIPLIER

    semantic = search_chunks(query, top_k=pool, movie_title=movie_title, collection=collection)
    keyword = keyword_search(query, movie_title, collection, limit=pool)

    by_id = {hit.chunk_id: hit for hit in semantic}

    # Chunks found only by keywords have no meaning-score yet: compute it.
    keyword_only = [hit for hit in keyword if hit.chunk_id not in by_id]
    if keyword_only:
        query_vector = embed_query(query)
        vectors = embed_texts([hit.text for hit in keyword_only])
        for hit, vector in zip(keyword_only, vectors):
            by_id[hit.chunk_id] = replace(hit, score=_dot(query_vector, vector))

    order = fuse_rankings(
        [[hit.chunk_id for hit in semantic], [hit.chunk_id for hit in keyword]]
    )
    best = [by_id[chunk_id] for chunk_id in order[:top_k]]
    return filter_relevant(best, min_score)