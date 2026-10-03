from agent.agent import handle_message


def fake_llm(reply: str):
    def generate(system_prompt: str, user_prompt: str) -> str:
        return reply

    return generate


def must_not_be_called(system_prompt: str, user_prompt: str) -> str:
    raise AssertionError("The LLM should not be called for this message.")


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


def test_email_request_is_recognized_without_calling_the_llm(sample_store):
    response = handle_message(
        "Send me a summary of Orbit Seven at john@example.com",
        collection=sample_store,
        generate_fn=must_not_be_called,
    )
    assert response.kind == "email_pending"
    assert response.request.recipient_email == "john@example.com"
    assert response.request.movie_title == "Orbit Seven"
    assert "john@example.com" in response.text


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