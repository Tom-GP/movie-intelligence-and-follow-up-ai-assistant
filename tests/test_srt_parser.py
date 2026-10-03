from pathlib import Path

import pytest

from config.settings import PROJECT_ROOT
from ingestion.srt_parser import (
    make_movie_id,
    make_movie_title,
    milliseconds_to_timestamp,
    parse_srt_file,
)

LIGHTHOUSE = PROJECT_ROOT / "data" / "subtitles" / "the_lighthouse_keeper.srt"


def test_parses_all_subtitles():
    lines = parse_srt_file(LIGHTHOUSE)
    assert len(lines) == 11


def test_timestamps_are_correct():
    second = parse_srt_file(LIGHTHOUSE)[1]
    assert second.subtitle_id == 2
    assert second.start_time == "00:01:04,200"
    assert second.end_time == "00:01:07,500"
    assert second.start_seconds == pytest.approx(64.2)
    assert second.end_seconds == pytest.approx(67.5)


def test_movie_title_is_added_to_every_line():
    lines = parse_srt_file(LIGHTHOUSE)
    assert all(line.movie_title == "The Lighthouse Keeper" for line in lines)


def test_milliseconds_to_timestamp():
    assert milliseconds_to_timestamp(3_723_450) == "01:02:03,450"
    assert milliseconds_to_timestamp(0) == "00:00:00,000"


def test_title_and_id_from_filename():
    assert make_movie_title(Path("dust_and_thunder.srt")) == "Dust and Thunder"
    assert make_movie_title(Path("orbit_seven.srt")) == "Orbit Seven"
    assert make_movie_id(Path("The Lighthouse-Keeper.srt")) == "the_lighthouse_keeper"