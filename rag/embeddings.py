"""Turns text into embeddings (lists of numbers) using sentence-transformers."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading is slow)."""
    return SentenceTransformer(settings.embedding_model_name)


def get_max_sequence_length() -> int:
    """How many tokens of a text the model actually reads."""
    return int(get_embedding_model().max_seq_length)


def embed_texts(
    texts: list[str], batch_size: int = 32, show_progress: bool = False
) -> list[list[float]]:
    """Return one embedding per text."""
    if not texts:
        return []
    vectors = get_embedding_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single question."""
    return embed_texts([query])[0]