from agent import ambiguity
from agent.ambiguity import (
    QuoteMatch,
    apply_reply,
    check_ambiguity,
    choose_candidates,
    extract_quote,
    find_quote_matches,
    load_chunk,
)
from agent.router import route_request
from models.schemas import AgentRequest
from rag.vector_store import SearchResult, get_collection, list_movie_titles

TITLES = ["Dust and Thunder", "Orbit Seven", "The Lighthouse Keeper"]


def make_hit(title: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=f"{title}_0001",
        movie_id=title,
        movie_title=title,
        start_time="00:00:01,000",
        end_time="00:00:02,000",
        subtitle_start_index=1,
        subtitle_end_index=2,
        text="hello",
        score=score,
    )


def pending_email_request() -> AgentRequest:
    return AgentRequest(
        intent="clarification",
        original_message="Email Sarah a summary of Orbit Seven",
        movie_title="Orbit Seven",
        pending_intent="email",
        missing_fields=["recipient_email"],
        clarification_question="What email address should I send the summary to?",
    )


# ---------- quotes ----------


def test_extract_quote_finds_quoted_text():
    assert extract_quote('Who says "I know"?') == "I know"
    assert extract_quote("Who says “I know”?") == "I know"
    assert extract_quote("Who says I know?") is None


def test_find_quote_matches_across_movies(sample_store):
    matches = find_quote_matches("I know", collection=sample_store)
    assert [m.movie_title for m in matches] == TITLES


def test_find_quote_matches_can_filter_by_movie(sample_store):
    matches = find_quote_matches("I know", movie_title="Orbit Seven", collection=sample_store)
    assert len(matches) == 1
    assert matches[0].chunk_id == "orbit_seven_0001"


def test_quote_matching_ignores_case_and_punctuation(sample_store):
    assert len(find_quote_matches("I KNOW!", collection=sample_store)) == 3


def test_quote_in_several_movies_asks_which_movie(sample_store):
    titles = list_movie_titles(sample_store)
    request = route_request('Who says "I know"?', titles)
    checked = check_ambiguity(request, sample_store, "")

    assert checked.intent == "clarification"
    assert checked.pending_intent == "information"
    assert checked.missing_fields == ["movie_title"]
    assert checked.options == TITLES
    assert "3 movies" in checked.clarification_question
    assert "1. Dust and Thunder (1 scene)" in checked.clarification_question


def test_quote_in_one_movie_pins_the_scene(sample_store):
    titles = list_movie_titles(sample_store)
    request = route_request('Who says "I know" in Orbit Seven?', titles)
    checked = check_ambiguity(request, sample_store, "")

    assert checked.intent == "information"
    assert checked.chunk_id == "orbit_seven_0001"
    assert checked.movie_title == "Orbit Seven"


def test_quote_in_several_scenes_asks_which_scene(monkeypatch):
    def fake_matches(quote, movie_title=None, collection=None):
        return [
            QuoteMatch("m_0001", "Movie M", "00:10:00,000", "00:10:40,000", 600.0),
            QuoteMatch("m_0002", "Movie M", "00:50:00,000", "00:50:30,000", 3000.0),
        ]

    monkeypatch.setattr(ambiguity, "find_quote_matches", fake_matches)
    request = AgentRequest(
        intent="information",
        original_message='Who says "I know"?',
        movie_title="Movie M",
        query='Who says "I know"?',
    )
    checked = check_ambiguity(request, None, "")

    assert checked.intent == "clarification"
    assert checked.missing_fields == ["scene"]
    assert checked.options == ["00:10:00–00:10:40", "00:50:00–00:50:30"]
    assert "2. 00:50:00–00:50:30" in checked.clarification_question


def test_reply_picks_a_scene(monkeypatch):
    def fake_matches(quote, movie_title=None, collection=None):
        return [
            QuoteMatch("m_0001", "Movie M", "00:10:00,000", "00:10:40,000", 600.0),
            QuoteMatch("m_0002", "Movie M", "00:50:00,000", "00:50:30,000", 3000.0),
        ]

    monkeypatch.setattr(ambiguity, "find_quote_matches", fake_matches)
    pending = AgentRequest(
        intent="clarification",
        original_message='Who says "I know"?',
        movie_title="Movie M",
        pending_intent="information",
        missing_fields=["scene"],
        clarification_question="Which scene?",
    )
    resumed = apply_reply(pending, "2", ["Movie M"], "")

    assert resumed.intent == "information"
    assert resumed.chunk_id == "m_0002"
    assert resumed.missing_fields == []


# ---------- recipient and movie ----------


def test_missing_recipient_is_asked(sample_store):
    request = route_request("Email Sarah a summary of Orbit Seven", TITLES)
    checked = check_ambiguity(request, sample_store, "")

    assert checked.intent == "clarification"
    assert checked.pending_intent == "email"
    assert checked.missing_fields == ["recipient_email"]
    assert "email address" in checked.clarification_question


def test_me_uses_the_default_recipient(sample_store):
    request = route_request("Email me a summary of Orbit Seven", TITLES)
    checked = check_ambiguity(request, sample_store, "me@example.com")

    assert checked.intent == "email"
    assert checked.recipient_email == "me@example.com"


def test_me_without_a_default_recipient_asks(sample_store):
    request = route_request("Email me a summary of Orbit Seven", TITLES)
    checked = check_ambiguity(request, sample_store, "")

    assert checked.intent == "clarification"
    assert checked.missing_fields == ["recipient_email"]
    assert "DEFAULT_RECIPIENT_EMAIL" in checked.clarification_question


def test_missing_movie_for_email_is_asked(tmp_path):
    empty = get_collection(db_path=tmp_path, collection_name="empty_missing_movie")
    request = route_request(
        "Send a summary of the final confrontation to john@example.com", [])
    checked = check_ambiguity(request, empty, "")

    assert checked.intent == "clarification"
    assert checked.missing_fields == ["movie_title"]
    assert "Which movie" in checked.clarification_question


def test_missing_movie_and_recipient_are_asked_together(tmp_path):
    empty = get_collection(db_path=tmp_path, collection_name="empty_both_missing")
    request = route_request("Send the scene to John", [])
    checked = check_ambiguity(request, empty, "")

    assert set(checked.missing_fields) == {"movie_title", "recipient_email"}
    assert "movie" in checked.clarification_question
    assert "email address" in checked.clarification_question


# ---------- candidate movies ----------


def test_choose_candidates_keeps_only_close_movies():
    hits = [make_hit("A", 0.50), make_hit("B", 0.46), make_hit("C", 0.30)]
    assert choose_candidates(hits, margin=0.08) == ["A", "B"]

    hits = [make_hit("A", 0.50), make_hit("B", 0.30)]
    assert choose_candidates(hits, margin=0.08) == ["A"]

    hits = [make_hit("A", 0.25), make_hit("B", 0.24)]
    assert choose_candidates(hits) == []  # nothing matches clearly


def test_single_clear_candidate_movie_is_used(sample_store):
    titles = list_movie_titles(sample_store)
    request = route_request("Why did Tomas shut down the oxygen recycler?", titles)
    checked = check_ambiguity(request, sample_store, "")

    assert checked.intent == "information"
    assert checked.movie_title == "Orbit Seven"


# ---------- reading replies ----------


def test_reply_picks_a_movie_by_number(sample_store):
    titles = list_movie_titles(sample_store)
    request = route_request('Who says "I know"?', titles)
    pending = check_ambiguity(request, sample_store, "")

    resumed = apply_reply(pending, "2", titles, "")
    assert resumed.intent == "information"
    assert resumed.movie_title == "Orbit Seven"
    assert resumed.clarification_question is None


def test_reply_can_give_a_recipient_or_me():
    pending = pending_email_request()

    resumed = apply_reply(pending, "john@example.com", TITLES, "")
    assert resumed.intent == "email"
    assert resumed.recipient_email == "john@example.com"

    resumed = apply_reply(pending, "me", TITLES, "me@example.com")
    assert resumed.recipient_email == "me@example.com"

    assert apply_reply(pending, "banana", TITLES, "") is None


def test_long_reply_is_treated_as_a_new_question():
    pending = AgentRequest(
        intent="clarification",
        original_message='Who says "I know"?',
        pending_intent="information",
        missing_fields=["movie_title"],
        options=TITLES,
        clarification_question="Which movie?",
    )
    reply = "Why did Tomas shut down the oxygen recycler in Orbit Seven today"
    assert apply_reply(pending, reply, TITLES, "") is None


def test_load_chunk_returns_the_stored_chunk(sample_store):
    chunk = load_chunk("orbit_seven_0001", sample_store)
    assert chunk.movie_title == "Orbit Seven"
    assert chunk.start_time == "00:02:14,000"
    assert load_chunk("does_not_exist", sample_store) is None