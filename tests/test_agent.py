from agent.agent import handle_message
from mcp_tools.email_client import EmailResult, EmailSendError
from rag.prompt import NOT_FOUND_TEXT

EMAIL_REPLY = (
    "Subject: Orbit Seven recap\n\n"
    "Hi! Tomas shut down the recycler because the core overheated [1].\n\n"
    "Best,\nMovie Assistant"
)


def fake_llm(reply: str):
    def generate(system_prompt: str, user_prompt: str) -> str:
        return reply

    return generate


def must_not_be_called(system_prompt: str, user_prompt: str) -> str:
    raise AssertionError("The LLM should not be called for this message.")


class FakeSender:
    """A pretend email tool that records what it was asked to send."""

    def __init__(self):
        self.calls = []

    def __call__(self, recipient: str, subject: str, body: str) -> EmailResult:
        self.calls.append((recipient, subject, body))
        return EmailResult(ok=True, message="DRY RUN: pretend it was sent.")


def test_information_request_gets_a_cited_answer(sample_store):
    response = handle_message(
        "Why is the lamp logbook missing a page?",
        collection=sample_store,
        generate_fn=fake_llm("A page was torn out of the logbook [1]."),
    )
    assert response.kind == "answer"
    assert "torn out" in response.text
    assert response.sources[0].startswith("[1] The Lighthouse Keeper")
    assert "Sources:" in response.format()


def test_email_request_is_written_and_sent_through_the_email_tool(sample_store):
    sender = FakeSender()
    response = handle_message(
        "Send me a summary of Orbit Seven at john@example.com",
        collection=sample_store,
        generate_fn=fake_llm(EMAIL_REPLY),
        send_email_fn=sender,
    )

    assert response.kind == "email_sent"
    assert response.sources == ["[1] Orbit Seven — 00:02:14–00:02:58"]
    assert "john@example.com" in response.text

    assert len(sender.calls) == 1
    recipient, subject, body = sender.calls[0]
    assert recipient == "john@example.com"
    assert subject == "Orbit Seven recap"
    assert "Sources:" in body
    assert "- [1] Orbit Seven — 00:02:14–00:02:58" in body


def test_greeting_gets_a_clarification_question(sample_store):
    response = handle_message(
        "hi", collection=sample_store, generate_fn=must_not_be_called
    )
    assert response.kind == "clarification"
    assert response.sources == []


def test_movie_in_message_limits_the_search(sample_store):
    response = handle_message(
        "In Orbit Seven, why was the oxygen recycler shut down?",
        collection=sample_store,
        generate_fn=fake_llm("The core was overheating [1]."),
    )
    assert response.kind == "answer"
    assert response.sources[0].startswith("[1] Orbit Seven")


def test_follow_up_reply_completes_the_request(sample_store):
    first = handle_message(
        'Who says "I know"?',
        collection=sample_store,
        generate_fn=must_not_be_called,
    )
    assert first.kind == "clarification"
    assert "3 movies" in first.text

    second = handle_message(
        "Orbit Seven",
        pending=first.request,
        collection=sample_store,
        generate_fn=fake_llm("Captain Reyes says it [1]."),
    )
    assert second.kind == "answer"
    assert second.sources == ["[1] Orbit Seven — 00:02:14–00:02:58"]


def test_unrelated_reply_starts_a_new_request(sample_store):
    first = handle_message(
        "Email Sarah a summary of Orbit Seven",
        collection=sample_store,
        generate_fn=must_not_be_called,
    )
    assert first.kind == "clarification"
    assert first.request.missing_fields == ["recipient_email"]

    second = handle_message(
        "Why is the lamp logbook missing a page?",
        pending=first.request,
        collection=sample_store,
        generate_fn=fake_llm("A page was torn out [1]."),
    )
    assert second.kind == "answer"


def test_whole_movie_question_uses_chunks_from_the_whole_film(sample_store):
    response = handle_message(
        "Summarize Orbit Seven",
        collection=sample_store,
        generate_fn=fake_llm("Two crew members race to save the ship [1]."),
    )
    assert response.kind == "answer"
    assert response.sources == ["[1] Orbit Seven — 00:02:14–00:02:58"]


def test_nothing_is_sent_when_the_llm_finds_nothing(sample_store):
    sender = FakeSender()
    response = handle_message(
        "Send me a summary of Orbit Seven at john@example.com",
        collection=sample_store,
        generate_fn=fake_llm(NOT_FOUND_TEXT),
        send_email_fn=sender,
    )
    assert response.kind == "answer"
    assert "nothing was sent" in response.text
    assert sender.calls == []


def test_a_failed_send_is_reported_as_an_error(sample_store):
    def failing_sender(recipient, subject, body):
        raise EmailSendError("SMTP login failed")

    response = handle_message(
        "Send me a summary of Orbit Seven at john@example.com",
        collection=sample_store,
        generate_fn=fake_llm(EMAIL_REPLY),
        send_email_fn=failing_sender,
    )
    assert response.kind == "error"
    assert "SMTP login failed" in response.text