"""Reads all .srt files, builds chunks, and stores them in the vector database.

Run from the project root:
    python -m scripts.ingest_movies --reset
"""

import argparse

from config.settings import settings
from ingestion.ingest import ingest_directory
from rag.embeddings import get_max_sequence_length
from rag.vector_store import add_chunks, count_chunks, reset_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Index subtitle files.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete everything in the vector database before indexing.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection()
        print("Vector database cleared.")

    print(f"Reading subtitles from {settings.subtitles_dir}")
    chunks = ingest_directory()
    movie_count = len({chunk.movie_id for chunk in chunks})
    print(f"Found {len(chunks)} chunks from {movie_count} movies.")

    # Roughly 3 words = 4 tokens, so 75% of the token limit is a safe word limit.
    word_limit = int(get_max_sequence_length() * 0.75)
    too_long = [c for c in chunks if len(c.text.split()) > word_limit]
    if too_long:
        print(
            f"Warning: {len(too_long)} chunks have more than ~{word_limit} words, "
            "so the embedding model will not read their last words."
        )

    print("Embedding and storing (the first run downloads the model)...")
    add_chunks(chunks)
    print(f"Done. Total chunks in database: {count_chunks()}")


if __name__ == "__main__":
    main()