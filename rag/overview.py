"""Helpers for whole-movie requests, like 'summarize the movie'."""

from chromadb.api.models.Collection import Collection

from rag.retriever import _normalize, _result_from_stored, content_words, stem
from rag.vector_store import SearchResult, get_collection

DEFAULT_SAMPLE_SIZE = 24

# Words that ask for "the whole thing" rather than for a specific scene.
GENERIC_WORDS = {
    "summary", "summarize", "summarise", "recap", "overview", "story", "plot",
    "whole", "entire", "full", "short", "brief", "briefly", "analysis",
    "analyze", "analyse", "happen", "please", "everything", "explain",
    "describe", "give",
}


def is_whole_movie_request(text: str, movie_title: str) -> bool:
    """True if the text asks about the movie as a whole, not about a specific scene.

    'a summary of Orbit Seven' -> True
    'a summary of the interrogation scene in Orbit Seven' -> False
    """
    title_words = {stem(word) for word in _normalize(movie_title).split()}
    remaining = [
        word
        for word in content_words(text)
        if word not in title_words and word not in GENERIC_WORDS
    ]
    return not remaining


def spread_indices(total: int, count: int) -> list[int]:
    """Pick `count` positions spread evenly over `total` items (first and last included)."""
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def sample_movie_chunks(
    movie_title: str,
    count: int = DEFAULT_SAMPLE_SIZE,
    collection: Collection | None = None,
) -> list[SearchResult]:
    """Pick chunks evenly spread across the whole movie, in story order."""
    collection = get_collection() if collection is None else collection
    data = collection.get(where={"movie_title": movie_title}, include=["documents", "metadatas"])

    items = sorted(
        zip(data["documents"], data["metadatas"]),
        key=lambda pair: float(pair[1]["start_seconds"]),
    )
    chosen = [items[i] for i in spread_indices(len(items), count)]
    return [_result_from_stored(text, meta, 1.0) for text, meta in chosen]