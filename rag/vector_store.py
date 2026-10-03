"""Stores chunks in a Chroma vector database and searches them."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from config.settings import settings
from ingestion.chunker import Chunk
from rag.embeddings import embed_query, embed_texts


@dataclass(frozen=True)
class SearchResult:
    """One search hit, with everything needed to cite it."""

    chunk_id: str
    movie_id: str
    movie_title: str
    start_time: str
    end_time: str
    subtitle_start_index: int
    subtitle_end_index: int
    text: str
    score: float  # similarity: closer to 1.0 means closer in meaning


def _make_client(db_path: Path | None = None) -> chromadb.ClientAPI:
    path = db_path or settings.vector_db_path
    return chromadb.PersistentClient(path=str(path))


def get_collection(
    db_path: Path | None = None, collection_name: str | None = None
) -> Collection:
    """Open the collection (create it if it doesn't exist yet)."""
    name = collection_name or settings.collection_name
    return _make_client(db_path).get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # compare vectors by cosine similarity
    )


def reset_collection(
    db_path: Path | None = None, collection_name: str | None = None
) -> Collection:
    """Delete everything in the collection and return a fresh empty one."""
    name = collection_name or settings.collection_name
    client = _make_client(db_path)
    try:
        client.delete_collection(name)
    except Exception:
        pass  # it didn't exist yet, which is fine
    return get_collection(db_path, collection_name)


def _resolve(collection: Collection | None) -> Collection:
    return get_collection() if collection is None else collection


def add_chunks(
    chunks: list[Chunk], collection: Collection | None = None, batch_size: int = 64
) -> int:
    """Embed chunks and store them. Safe to run again (same IDs are overwritten)."""
    collection = _resolve(collection)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            metadatas=[chunk.to_metadata() for chunk in batch],
            embeddings=embed_texts(texts),
        )
    return len(chunks)


def count_chunks(collection: Collection | None = None) -> int:
    return _resolve(collection).count()


def list_movie_titles(collection: Collection | None = None) -> list[str]:
    """All movie titles currently stored, sorted."""
    data = _resolve(collection).get(include=["metadatas"])
    return sorted({str(meta["movie_title"]) for meta in data["metadatas"]})


def search_chunks(
    query: str,
    top_k: int | None = None,
    movie_title: str | None = None,
    collection: Collection | None = None,
) -> list[SearchResult]:
    """Find the chunks whose meaning is closest to the query."""
    collection = _resolve(collection)
    total = collection.count()
    if total == 0 or not query.strip():
        return []

    top_k = top_k or settings.top_k
    where = {"movie_title": movie_title} if movie_title else None

    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[SearchResult] = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            SearchResult(
                chunk_id=str(meta["chunk_id"]),
                movie_id=str(meta["movie_id"]),
                movie_title=str(meta["movie_title"]),
                start_time=str(meta["start_time"]),
                end_time=str(meta["end_time"]),
                subtitle_start_index=int(meta["subtitle_start_index"]),
                subtitle_end_index=int(meta["subtitle_end_index"]),
                text=text,
                score=1.0 - float(distance),
            )
        )
    return hits