"""Groups subtitle lines into time-aware chunks of roughly 30-90 seconds."""

from dataclasses import dataclass

from ingestion.srt_parser import SubtitleLine

# A silence this long (in seconds) between two lines counts as a natural break.
GAP_BREAK_SECONDS = 5.0


@dataclass(frozen=True)
class Chunk:
    """A piece of dialogue plus the facts needed to cite it."""

    chunk_id: str
    movie_id: str
    movie_title: str
    text: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    subtitle_start_index: int
    subtitle_end_index: int

    def to_metadata(self) -> dict[str, str | int | float]:
        """Everything except the text. This is what we store next to the vector."""
        return {
            "chunk_id": self.chunk_id,
            "movie_id": self.movie_id,
            "movie_title": self.movie_title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "subtitle_start_index": self.subtitle_start_index,
            "subtitle_end_index": self.subtitle_end_index,
        }


def _should_start_new_group(
    current: list[SubtitleLine],
    next_line: SubtitleLine,
    min_seconds: int,
    max_seconds: int,
) -> bool:
    """Decide whether next_line belongs in a new chunk instead of the current one."""
    first_start = current[0].start_seconds
    current_duration = current[-1].end_seconds - first_start
    duration_if_added = next_line.end_seconds - first_start
    gap = next_line.start_seconds - current[-1].end_seconds

    too_long = duration_if_added > max_seconds
    natural_pause = current_duration >= min_seconds and gap >= GAP_BREAK_SECONDS
    return too_long or natural_pause


def _group_lines(
    lines: list[SubtitleLine], min_seconds: int, max_seconds: int
) -> list[list[SubtitleLine]]:
    groups: list[list[SubtitleLine]] = []
    current: list[SubtitleLine] = []

    for line in lines:
        if current and _should_start_new_group(current, line, min_seconds, max_seconds):
            groups.append(current)
            current = []
        current.append(line)

    if current:
        groups.append(current)
    return groups


def _merge_short_last_group(
    groups: list[list[SubtitleLine]], min_seconds: int, max_seconds: int
) -> list[list[SubtitleLine]]:
    """Avoid a tiny leftover chunk at the end by joining it to the previous one."""
    if len(groups) < 2:
        return groups

    last = groups[-1]
    last_duration = last[-1].end_seconds - last[0].start_seconds
    merged_duration = last[-1].end_seconds - groups[-2][0].start_seconds

    if last_duration < min_seconds and merged_duration <= max_seconds:
        groups[-2] = groups[-2] + last
        groups.pop()
    return groups


def _make_chunk(group: list[SubtitleLine], movie_id: str, number: int) -> Chunk:
    first, last = group[0], group[-1]
    return Chunk(
        chunk_id=f"{movie_id}_{number:04d}",
        movie_id=movie_id,
        movie_title=first.movie_title,
        text="\n".join(line.text for line in group),
        start_time=first.start_time,
        end_time=last.end_time,
        start_seconds=first.start_seconds,
        end_seconds=last.end_seconds,
        subtitle_start_index=first.subtitle_id,
        subtitle_end_index=last.subtitle_id,
    )


def build_chunks(
    lines: list[SubtitleLine],
    movie_id: str,
    min_seconds: int,
    max_seconds: int,
) -> list[Chunk]:
    """Turn cleaned subtitle lines (in time order) into chunks."""
    groups = _group_lines(lines, min_seconds, max_seconds)
    groups = _merge_short_last_group(groups, min_seconds, max_seconds)
    return [
        _make_chunk(group, movie_id, number)
        for number, group in enumerate(groups, start=1)
    ]