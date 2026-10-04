"""Finds subtitle files whose movie is not in the vector database yet."""

from pathlib import Path

from ingestion.srt_parser import make_movie_title


def find_new_subtitle_files(directory: Path, indexed_titles: list[str]) -> list[Path]:
    """Return the .srt files whose movie title is not among the indexed titles."""
    indexed = set(indexed_titles)
    return [
        path
        for path in sorted(directory.glob("*.srt"))
        if make_movie_title(path) not in indexed
    ]