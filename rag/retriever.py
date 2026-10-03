"""Finds the subtitle chunks that are relevant to a question."""

import re

from chromadb.api.models.Collection import Collection

from rag.vector_store import SearchResult, list_movie_titles, search_chunks

# Results with a lower similarity score than this are treated as "not relevant".
# Tune it by looking at the scores printed by scripts.search_demo.
MIN_RELEVANCE_SCORE = 0.20

LEADING_WORDS = ("the ", "a ", "an ")


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


def retrieve(
    question: str,
    movie_title: str | None = None,
    top_k: int | None = None,
    min_score: float = MIN_RELEVANCE_SCORE,
    collection: Collection | None = None,
) -> list[SearchResult]:
    """Search the vector database and keep only relevant chunks."""
    hits = search_chunks(
        clean_query(question),
        top_k=top_k,
        movie_title=movie_title,
        collection=collection,
    )
    return filter_relevant(hits, min_score)