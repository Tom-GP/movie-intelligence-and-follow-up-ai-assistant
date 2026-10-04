import pytest

pytest.importorskip("streamlit.testing.v1")  # skipped on a very old Streamlit

from streamlit.testing.v1 import AppTest  # noqa: E402

import mcp_tools.email_client as email_client  # noqa: E402
from config.settings import PROJECT_ROOT  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    """The app, started headlessly, with a fake email status (no subprocess)."""
    monkeypatch.setattr(
        email_client,
        "get_email_status",
        lambda dry_run=None: "dry-run: emails are checked but NOT sent",
    )
    test = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    test.run()
    return test


def test_app_starts_without_errors(app):
    assert not app.exception
    assert "Movie Intelligence Assistant" in app.title[0].value


def test_greeting_gets_a_clarification_reply(app):
    app.chat_input[0].set_value("hi").run()

    assert not app.exception
    page_text = " ".join(item.value for item in app.markdown).lower()
    assert "tell me a bit more" in page_text