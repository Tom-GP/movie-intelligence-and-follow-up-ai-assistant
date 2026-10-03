"""Reads an .srt file and turns each subtitle into a SubtitleLine object."""

import re
from dataclasses import dataclass
from pathlib import Path

import pysrt

# Words that stay lowercase in a movie title (unless they come first).
SMALL_WORDS = {"a", "an", "and", "of", "the", "in", "on", "to"}


@dataclass(frozen=True)
class SubtitleLine:
    """One subtitle entry from the .srt file."""

    movie_title: str
    subtitle_id: int
    start_time: str  # for example "00:01:04,200 showing citations"
    end_time: str
    start_seconds: float  # for example 64.2 doing some operations
    end_seconds: float
    text: str


def milliseconds_to_timestamp(milliseconds: int) -> str:
    """Turn 3723450 into '01:02:03,450'."""
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def make_movie_id(path: Path) -> str:
    """'The Lighthouse-Keeper.srt' -> 'the_lighthouse_keeper'."""
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")


def make_movie_title(path: Path) -> str:
    """'dust_and_thunder.srt' -> 'Dust and Thunder'."""
    words = path.stem.replace("_", " ").replace("-", " ").split()
    titled_words = []
    for position, word in enumerate(words):
        if position > 0 and word.lower() in SMALL_WORDS:
            titled_words.append(word.lower())
        else:
            titled_words.append(word.capitalize())
    return " ".join(titled_words)


def _open_srt(path: Path) -> pysrt.SubRipFile:
    """Open an .srt file, trying UTF-8 first and then a common older encoding."""
    try:
        return pysrt.open(str(path), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pysrt.open(str(path), encoding="latin-1")


def parse_srt_file(path: Path) -> list[SubtitleLine]:
    """Read one .srt file and return its subtitles in order."""
    movie_title = make_movie_title(path)
    lines: list[SubtitleLine] = []

    for item in _open_srt(path):
        start_ms = item.start.ordinal  # milliseconds since the start of the movie
        end_ms = item.end.ordinal
        lines.append(
            SubtitleLine(
                movie_title=movie_title,
                subtitle_id=item.index,
                start_time=milliseconds_to_timestamp(start_ms),
                end_time=milliseconds_to_timestamp(end_ms),
                start_seconds=start_ms / 1000,
                end_seconds=end_ms / 1000,
                text=item.text,
            )
        )

    return lines