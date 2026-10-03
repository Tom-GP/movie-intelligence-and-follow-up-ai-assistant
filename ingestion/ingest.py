"""End-to-end ingestion: .srt files -> parsed -> cleaned -> chunks."""

from dataclasses import replace
from pathlib import Path

from config.settings import settings
from ingestion.chunker import Chunk, build_chunks
from ingestion.cleaner import clean_text
from ingestion.srt_parser import SubtitleLine, make_movie_id, parse_srt_file


def clean_lines(lines: list[SubtitleLine]) -> list[SubtitleLine]:
    """Clean every line's text and drop lines that become empty."""
    cleaned: list[SubtitleLine] = []
    for line in lines:
        text = clean_text(line.text)
        if text:
            cleaned.append(replace(line, text=text)) 
    return cleaned


def ingest_file(
    path: Path,
    min_seconds: int = settings.chunk_min_seconds,
    max_seconds: int = settings.chunk_max_seconds,
) -> list[Chunk]:
    """Process one .srt file into chunks."""
    lines = parse_srt_file(path)
    lines = clean_lines(lines)
    return build_chunks(
        lines,
        movie_id=make_movie_id(path),
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def ingest_directory(directory: Path = settings.subtitles_dir) -> list[Chunk]:
    """Process every .srt file in a folder."""
    srt_files = sorted(directory.glob("*.srt"))
    if not srt_files:
        raise FileNotFoundError(f"No .srt files found in {directory}")

    all_chunks: list[Chunk] = []
    for path in srt_files:
        all_chunks.extend(ingest_file(path))
    return all_chunks


def main() -> None:
    chunks = ingest_directory()
    movie_count = len({chunk.movie_id for chunk in chunks})
    print(f"Total: {len(chunks)} chunks from {movie_count} movies")

    for chunk in chunks:
        print(
            f"{chunk.movie_title} | {chunk.chunk_id} | "
            f"{chunk.start_time} --> {chunk.end_time} | "
            f"subtitles {chunk.subtitle_start_index}-{chunk.subtitle_end_index}"
        )

    print("\nPreview of the first chunk:\n")
    print(chunks[0].text)


if __name__ == "__main__":
    main()