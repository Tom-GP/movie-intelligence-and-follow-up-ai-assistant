from ingestion.new_files import find_new_subtitle_files


def test_files_whose_movie_is_not_indexed_are_found(tmp_path):
    (tmp_path / "big_fish.srt").write_text("x")
    (tmp_path / "orbit_seven.srt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    new_files = find_new_subtitle_files(tmp_path, ["Orbit Seven"])

    assert [path.name for path in new_files] == ["big_fish.srt"]


def test_nothing_is_new_when_everything_is_indexed(tmp_path):
    (tmp_path / "orbit_seven.srt").write_text("x")
    assert find_new_subtitle_files(tmp_path, ["Orbit Seven"]) == []