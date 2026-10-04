from dataclasses import replace

import pytest

from config.settings import settings as real_settings
from mcp_tools import email_server
from mcp_tools.email_client import (
    EmailSendError,
    list_email_tools,
    send_email_via_mcp,
)
from mcp_tools.email_server import (
    build_message,
    email_status,
    send_email,
    validate_email_input,
)


def use_settings(monkeypatch, **changes):
    """Temporarily change settings inside the email server module."""
    monkeypatch.setattr(email_server, "settings", replace(real_settings, **changes))


# ---------- the server's own functions (no subprocess) ----------


def test_valid_email_input_passes():
    validate_email_input("john@example.com", "Hello", "Some text")


def test_invalid_recipient_is_rejected():
    with pytest.raises(ValueError):
        validate_email_input("not-an-email", "Hello", "Some text")


def test_header_injection_is_rejected():
    with pytest.raises(ValueError):
        validate_email_input("john@example.com", "Hi\nBcc: evil@example.com", "Text")
    with pytest.raises(ValueError):
        validate_email_input("john@example.com\nBcc: evil@example.com", "Hi", "Text")


def test_empty_and_oversized_content_is_rejected():
    with pytest.raises(ValueError):
        validate_email_input("john@example.com", "  ", "Text")
    with pytest.raises(ValueError):
        validate_email_input("john@example.com", "Hi", "")
    with pytest.raises(ValueError):
        validate_email_input("john@example.com", "Hi", "x" * (email_server.MAX_BODY_LENGTH + 1))


def test_build_message_has_the_expected_headers():
    message = build_message("from@example.com", "john@example.com", "Hello", "Body text")
    assert message["From"] == "from@example.com"
    assert message["To"] == "john@example.com"
    assert message["Subject"] == "Hello"
    assert "Body text" in message.get_content()


def test_dry_run_never_calls_smtp(monkeypatch):
    use_settings(monkeypatch, email_dry_run=True)

    def must_not_be_called(message):
        raise AssertionError("SMTP must not be used in a dry run")

    monkeypatch.setattr(email_server, "deliver", must_not_be_called)
    result = send_email("john@example.com", "Hello", "Body text")
    assert result.startswith("DRY RUN")


def test_real_send_hands_one_message_to_smtp(monkeypatch):
    use_settings(
        monkeypatch,
        email_dry_run=False,
        smtp_host="smtp.test",
        smtp_user="user@test.com",
        smtp_password="pw",
        email_from="from@test.com",
    )
    sent = []
    monkeypatch.setattr(email_server, "deliver", lambda message: sent.append(message))

    result = send_email("john@example.com", "Hello", "Body text")

    assert "john@example.com" in result
    assert len(sent) == 1
    assert sent[0]["To"] == "john@example.com"
    assert sent[0]["From"] == "from@test.com"
    assert sent[0]["Subject"] == "Hello"


def test_missing_smtp_settings_are_an_error(monkeypatch):
    use_settings(monkeypatch, email_dry_run=False, smtp_host="", smtp_user="", smtp_password="")
    with pytest.raises(ValueError, match="SMTP"):
        send_email("john@example.com", "Hello", "Body text")


def test_email_status_describes_each_mode(monkeypatch):
    use_settings(monkeypatch, email_dry_run=True)
    assert email_status().startswith("dry-run")

    use_settings(monkeypatch, email_dry_run=False, smtp_host="", smtp_user="", smtp_password="")
    assert email_status().startswith("not configured")

    use_settings(
        monkeypatch,
        email_dry_run=False,
        smtp_host="smtp.test",
        smtp_user="u@test.com",
        smtp_password="pw",
        email_from="u@test.com",
    )
    assert email_status().startswith("ready")


# ---------- the client talking to the real server (starts a subprocess) ----------


def test_client_lists_the_server_tools():
    assert list_email_tools(dry_run=True) == ["email_status", "send_email"]


def test_client_sends_through_the_server_in_dry_run():
    result = send_email_via_mcp("john@example.com", "Hello", "Body text", dry_run=True)
    assert result.ok is True
    assert result.message.startswith("DRY RUN")


def test_client_reports_a_rejection_from_the_server():
    with pytest.raises(EmailSendError, match="not a valid email"):
        send_email_via_mcp("not-an-email", "Hello", "Body text", dry_run=True)