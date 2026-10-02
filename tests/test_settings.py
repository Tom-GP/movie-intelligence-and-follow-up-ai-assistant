from config.settings import PROJECT_ROOT, settings


def test_project_root_has_config_folder():
    assert (PROJECT_ROOT / "config").is_dir()


def test_default_values_are_sensible():
    assert settings.smtp_port > 0
    assert settings.top_k > 0
    assert settings.chunk_min_seconds < settings.chunk_max_seconds


def test_sample_subtitles_exist():
    srt_files = list(settings.subtitles_dir.glob("*.srt"))
    assert len(srt_files) >= 3