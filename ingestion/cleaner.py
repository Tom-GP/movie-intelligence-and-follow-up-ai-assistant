"""Removes subtitle noise (tags, sound notes, music symbols) from text."""

import re

HTML_TAG = re.compile(r"<[^>]+>")  # <i>, </i>, <b>, <font ...>
STYLE_CODE = re.compile(r"\{[^}]*\}")  # {\an8}
SQUARE_NOTE = re.compile(r"\[[^\]]*\]")  # [music], [door slams]
ROUND_NOTE = re.compile(r"\([^)]*\)")  # (sighs), (laughing)
MUSIC_SYMBOLS = re.compile(r"[♪♫♬]")
DIALOGUE_DASH = re.compile(r"^\s*-+\s*")  # "- Hello" at the start of a line


def clean_text(text: str) -> str:
    """Return the spoken words only. Returns '' if nothing is left."""
    text = HTML_TAG.sub("", text)
    text = STYLE_CODE.sub("", text)
    text = SQUARE_NOTE.sub("", text)
    text = ROUND_NOTE.sub("", text)
    text = MUSIC_SYMBOLS.sub("", text)

    # Clean each line on its own, so a dash left alone on a line disappears.
    cleaned_lines = []
    for line in text.splitlines():
        line = DIALOGUE_DASH.sub("", line)
        line = " ".join(line.split())
        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines)