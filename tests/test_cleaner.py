from ingestion.cleaner import clean_text


def test_removes_italic_tags():
    assert clean_text("<i>Hello there</i>") == "Hello there"


def test_removes_sound_notes():
    assert clean_text("[music]") == ""
    assert clean_text("(sighs) Fine.") == "Fine."


def test_removes_music_symbols():
    assert clean_text("♪ la la la ♪") == "la la la"


def test_removes_style_codes():
    assert clean_text("{\\an8}Top of the screen") == "Top of the screen"


def test_joins_lines_and_fixes_spaces():
    assert clean_text("Where were you\non the night   of the storm?") == (
        "Where were you on the night of the storm?"
    )


def test_keeps_speaker_names():
    assert clean_text("Mara: I know.") == "Mara: I know."


def test_no_stray_dash_after_sound_note_removed():
    assert clean_text("- (MUFASA GRUNTS)\n- Dad!") == "Dad!"


def test_two_speaker_lines_lose_their_dashes():
    assert clean_text("- Where's Sarabi?\n- She's leading the charge.") == (
        "Where's Sarabi? She's leading the charge."
    )


def test_line_with_only_dash_and_sound_becomes_empty():
    assert clean_text("- (GROWLS)\n- (ZAZU YELPS, WHIMPERS)") == ""