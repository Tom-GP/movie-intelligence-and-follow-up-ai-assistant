from ingestion.chunker import build_chunks
from ingestion.ingest import ingest_file
from ingestion.srt_parser import SubtitleLine, milliseconds_to_timestamp
from config.settings import PROJECT_ROOT

SUBTITLES = PROJECT_ROOT / "data" / "subtitles"


def make_line(subtitle_id: int, start: int, end: int) -> SubtitleLine:
    """Build a fake subtitle line for testing."""
    return SubtitleLine(
        movie_title="Test Movie",
        subtitle_id=subtitle_id,
        start_time=milliseconds_to_timestamp(start * 1000),
        end_time=milliseconds_to_timestamp(end * 1000),
        start_seconds=float(start),
        end_seconds=float(end),
        text="Hello there.",
    )


def test_continuous_dialogue_is_split_at_max_length():
    # 20 lines, each 3 seconds, back to back: 0s to 60s.
    lines = [make_line(i + 1, i * 3, i * 3 + 3) for i in range(20)]
    chunks = build_chunks(lines, "test_movie", min_seconds=15, max_seconds=30)

    assert len(chunks) == 2
    assert all(c.end_seconds - c.start_seconds <= 30 for c in chunks)
    assert chunks[0].subtitle_start_index == 1
    assert chunks[0].subtitle_end_index == 10
    assert chunks[1].subtitle_start_index == 11


def test_long_pause_starts_a_new_chunk():
    # A 3-second line every 10 seconds, so every pause is 7 seconds.
    lines = [make_line(i + 1, i * 10, i * 10 + 3) for i in range(12)]
    chunks = build_chunks(lines, "test_movie", min_seconds=15, max_seconds=30)

    assert len(chunks) == 4
    assert chunks[0].end_seconds == 23.0
    assert chunks[0].end_time == "00:00:23,000"


def test_no_subtitle_is_lost_or_repeated():
    lines = [make_line(i + 1, i * 3, i * 3 + 3) for i in range(20)]
    chunks = build_chunks(lines, "test_movie", min_seconds=15, max_seconds=30)

    covered = sum(c.subtitle_end_index - c.subtitle_start_index + 1 for c in chunks)
    assert covered == 20


def test_metadata_has_citation_fields_but_not_text():
    lines = [make_line(1, 0, 3)]
    chunk = build_chunks(lines, "test_movie", 15, 30)[0]
    metadata = chunk.to_metadata()

    assert metadata["movie_title"] == "Test Movie"
    assert metadata["start_time"] == "00:00:00,000"
    assert metadata["end_time"] == "00:00:03,000"
    assert metadata["chunk_id"] == "test_movie_0001"
    assert "text" not in metadata


def test_empty_input_gives_no_chunks():
    assert build_chunks([], "test_movie", 15, 30) == []


def test_sample_file_is_cleaned_and_chunked():
    chunks = ingest_file(SUBTITLES / "the_lighthouse_keeper.srt", 30, 90)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_time == "00:01:04,200"  # [waves crashing] was removed
    assert chunk.end_time == "00:01:50,000"  # [music] was removed
    assert "<i>" not in chunk.text
    assert "[" not in chunk.text


def test_short_leftover_chunk_is_merged():
    # In orbit_seven.srt a 7-second pause comes before the last line.
    # That last line is too short to stand alone, so it joins the previous chunk.
    chunks = ingest_file(SUBTITLES / "orbit_seven.srt", 30, 90)

    assert len(chunks) == 1
    assert chunks[0].end_time == "00:02:58,500"