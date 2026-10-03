import pytest
from pydantic import ValidationError

from agent.router import detect_requested_content, route_request
from models.schemas import AgentRequest

TITLES = ["Dust and Thunder", "Orbit Seven", "The Lion King", "The Dictator"]


def test_question_is_routed_as_information():
    request = route_request("What happens when Michael meets Sarah?", TITLES)
    assert request.intent == "information"
    assert request.query == "What happens when Michael meets Sarah?"
    assert request.recipient_email is None


def test_movie_title_in_question_is_detected():
    request = route_request("What does Scar say in the Lion King?", TITLES)
    assert request.intent == "information"
    assert request.movie_title == "The Lion King"


def test_email_command_extracts_recipient_movie_and_topic():
    message = "Send me a summary of the confrontation in Orbit Seven at john@example.com."
    request = route_request(message, TITLES)

    assert request.intent == "email"
    assert request.recipient_email == "john@example.com"
    assert request.movie_title == "Orbit Seven"
    assert request.requested_content == "summary"
    assert "confrontation" in request.query
    assert "@" not in request.query
    assert "send" not in request.query.lower()


def test_email_to_a_name_has_no_recipient_address():
    request = route_request(
        "Email Sarah a summary of the interrogation scene from Orbit Seven.", TITLES
    )
    assert request.intent == "email"
    assert request.recipient_email is None
    assert request.movie_title == "Orbit Seven"


def test_polite_email_request_is_email():
    request = route_request("Can you email me the scene summary?", TITLES)
    assert request.intent == "email"
    assert request.recipient_email is None


def test_send_in_the_middle_of_a_question_is_not_email():
    request = route_request("Why does Mara send the letter?", TITLES)
    assert request.intent == "information"


def test_send_after_and_is_email():
    request = route_request("Summarize the last scene and send it to Bob", TITLES)
    assert request.intent == "email"
    assert request.recipient_email is None


def test_email_address_alone_makes_it_email():
    request = route_request("the scene analysis for alice@example.com please", TITLES)
    assert request.intent == "email"
    assert request.recipient_email == "alice@example.com"


def test_empty_message_needs_clarification():
    request = route_request("   ", TITLES)
    assert request.intent == "clarification"
    assert request.clarification_question


def test_greeting_needs_clarification():
    for message in ["hi", "Hello!", "thanks", "help me"]:
        assert route_request(message, TITLES).intent == "clarification"


def test_single_word_needs_clarification():
    assert route_request("Aladeen?", TITLES).intent == "clarification"


def test_requested_content_detection():
    assert detect_requested_content("send me a summary") == "summary"
    assert detect_requested_content("analyze the quote for me") == "quote analysis"
    assert detect_requested_content("email an analysis of it") == "analysis"
    assert detect_requested_content("send me the scene") is None


def test_bad_recipient_email_is_rejected():
    with pytest.raises(ValidationError):
        AgentRequest(intent="email", recipient_email="not-an-email")


def test_clarification_needs_a_question():
    with pytest.raises(ValidationError):
        AgentRequest(intent="clarification")