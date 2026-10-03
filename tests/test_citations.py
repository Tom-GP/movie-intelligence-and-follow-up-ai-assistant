from rag.prompt import NOT_FOUND_TEXT, build_context
from rag.qa import (
    Answer,
    Citation,
    answer_question,
    extract_cited_numbers,
    format_timestamp,
)
from rag.vector_store import SearchResult, get_collection


def fake_llm(reply: str):
    """A pretend LLM that always returns the same reply."""

    def generate(system_prompt: str, user_prompt: str) -> str:
        return reply

    return generate


def make_result(movie: str = "The Lion King") -> SearchResult:
    return SearchResult(
        chunk_id="the_lion_king_0022",
        movie_id="the_lion_king",
        movie_title=movie,
        start_time="00:32:11,847",
        end_time="00:32:53,510",
        subtitle_start_index=499,
        subtitle_end_index=514,
        text="Hello.",
        score=0.5,
    )


LIGHTHOUSE_QUESTION = "Why is the lamp logbook missing a page?"


def test_format_timestamp_drops_milliseconds():
    assert format_timestamp("00:32:11,847") == "00:32:11"


def test_extract_cited_numbers_reads_common_formats():
    text = "First [1]. Second [2, 3] and again [1]. Invalid [9]."
    assert extract_cited_numbers(text, max_number=3) == [1, 2, 3]


def test_extract_cited_numbers_ignores_out_of_range():
    assert extract_cited_numbers("See [7] and [0].", max_number=3) == []


def test_build_context_numbers_excerpts_and_hides_timestamps():
    context = build_context([make_result("Movie A"), make_result("Movie B")])
    assert "[1] (movie: Movie A)" in context
    assert "[2] (movie: Movie B)" in context
    assert "00:32:11" not in context


def test_citation_label_comes_from_metadata():
    citation = Citation(
        number=1,
        chunk_id="the_lion_king_0022",
        movie_title="The Lion King",
        start_time="00:32:11,847",
        end_time="00:32:53,510",
        subtitle_start_index=499,
        subtitle_end_index=514,
    )
    assert citation.label() == "The Lion King — 00:32:11–00:32:53"


def test_answer_cites_only_chunks_the_model_used(sample_store):
    answer = answer_question(
        LIGHTHOUSE_QUESTION,
        collection=sample_store,
        generate_fn=fake_llm("A page was torn out of the logbook [1]."),
    )
    assert answer.found_evidence is True
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.movie_title == "The Lighthouse Keeper"
    assert citation.start_time == "00:01:04,200"
    assert citation.end_time == "00:01:50,000"


def test_model_cannot_invent_a_citation(sample_store):
    answer = answer_question(
        LIGHTHOUSE_QUESTION,
        collection=sample_store,
        generate_fn=fake_llm("It happened [1] and also [42] and [0]."),
    )
    assert [c.number for c in answer.citations] == [1]


def test_not_found_answer_has_no_citations(sample_store):
    answer = answer_question(
        LIGHTHOUSE_QUESTION,
        collection=sample_store,
        generate_fn=fake_llm(NOT_FOUND_TEXT),
    )
    assert answer.found_evidence is False
    assert answer.citations == []


def test_missing_markers_fall_back_to_retrieved_sources(sample_store):
    answer = answer_question(
        LIGHTHOUSE_QUESTION,
        collection=sample_store,
        generate_fn=fake_llm("A page is missing from the logbook."),
    )
    assert len(answer.citations) >= 1
    assert answer.citations[0].movie_title == "The Lighthouse Keeper"


def test_empty_database_never_calls_the_llm(tmp_path):
    empty = get_collection(db_path=tmp_path, collection_name="empty_test")

    def must_not_be_called(system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("The LLM should not be called without evidence.")

    answer = answer_question(
        "Anything at all?", collection=empty, generate_fn=must_not_be_called
    )
    assert answer.found_evidence is False
    assert answer.citations == []


def test_movie_filter_is_applied(sample_store):
    answer = answer_question(
        "someone stole a horse",
        movie_title="Orbit Seven",
        min_score=0.0,
        collection=sample_store,
        generate_fn=fake_llm("Something happens [1]."),
    )
    assert answer.citations[0].movie_title == "Orbit Seven"
    assert answer.movie_filter == "Orbit Seven"


def test_formatted_answer_lists_sources():
    citation = Citation(1, "id_1", "The Lighthouse Keeper", "00:01:04,200", "00:01:50,000", 2, 10)
    answer = Answer("A page was removed [1].", [citation], True)
    assert answer.format() == (
        "A page was removed [1].\n\n"
        "Sources:\n"
        "- [1] The Lighthouse Keeper — 00:01:04–00:01:50"
    )