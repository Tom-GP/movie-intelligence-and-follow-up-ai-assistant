from agent.agent import handle_message
from agent.prompts import EMAIL_SYSTEM_PROMPT
from agent.router import route_request
from mcp_tools.email_client import EmailResult
from rag.prompt import SUMMARY_SYSTEM_PROMPT

TITLES = ["Desert Mail", "Orbit Seven", "The Lion King"]

EMAIL_REPLY = (
    "Subject: Orbit Seven recap\n\n"
    "Hi,\nTomas shut down the recycler because the core overheated [1].\n\n"
    "Best,\nMovie Assistant"
)


def fake_llm(reply: str):
    def generate(system_prompt: str, user_prompt: str) -> str:
        return reply

    return generate


def must_not_be_called(*args) -> str:
    raise AssertionError("This should not have been called.")


class FakeSender:
    def __init__(self):
        self.calls = []

    def __call__(self, recipient: str, subject: str, body: str) -> EmailResult:
        self.calls.append((recipient, subject, body))
        return EmailResult(ok=True, message="DRY RUN: pretend it was sent.")


def test_send_email_at_the_end_of_a_question_is_an_email():
    request = route_request("Whose life he took? Send email about it", TITLES)
    assert request.intent == "email"
    assert request.recipient_email is None
    assert request.query == "Whose life he took"


def test_send_email_phrases_are_found_but_plain_questions_are_not():
    request = route_request("Please write an email about the ending of Desert Mail", TITLES)
    assert request.intent == "email"
    assert request.movie_title == "Desert Mail"

    assert route_request("Why does Mara send the letter?", TITLES).intent == "information"


def test_invalid_address_keeps_the_question_open(sample_store):
    first = handle_message(
        "Email a summary of Orbit Seven",
        collection=sample_store,
        generate_fn=must_not_be_called,
        send_email_fn=must_not_be_called,
    )
    assert first.kind == "clarification"
    assert first.request.missing_fields == ["recipient_email"]

    second = handle_message(
        "to2.com",
        pending=first.request,
        collection=sample_store,
        generate_fn=must_not_be_called,
        send_email_fn=must_not_be_called,
    )
    assert second.kind == "clarification"
    assert "valid email address" in second.text
    assert second.request.missing_fields == ["recipient_email"]

    sender = FakeSender()
    third = handle_message(
        "sarah@example.com",
        pending=second.request,
        collection=sample_store,
        generate_fn=fake_llm(EMAIL_REPLY),
        send_email_fn=sender,
    )
    assert third.kind == "email_sent"
    assert sender.calls[0][0] == "sarah@example.com"


def test_story_request_uses_the_summary_prompt(sample_store):
    seen = {}

    def recording_llm(system_prompt: str, user_prompt: str) -> str:
        seen["system"] = system_prompt
        return "Two crew members race to save the ship [1]."

    response = handle_message(
        "Tell the story of Orbit Seven in short",
        collection=sample_store,
        generate_fn=recording_llm,
    )
    assert response.kind == "answer"
    assert seen["system"] == SUMMARY_SYSTEM_PROMPT


def test_email_prompt_forbids_placeholders_and_sets_the_sign_off():
    assert "Movie Assistant" in EMAIL_SYSTEM_PROMPT
    assert "placeholder" in EMAIL_SYSTEM_PROMPT.lower()