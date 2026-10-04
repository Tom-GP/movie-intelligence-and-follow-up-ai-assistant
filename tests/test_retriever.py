from rag.retriever import (
    content_words,
    detect_movie_title,
    filter_relevant,
    fuse_rankings,
    keyword_scores,
    keyword_search,
)
from rag.vector_store import SearchResult

TITLES = ["Dust and Thunder", "Orbit Seven", "The Lion King", "The Dictator"]


def make_result(score: float) -> SearchResult:
    return SearchResult(
        chunk_id="x_0001",
        movie_id="x",
        movie_title="X",
        start_time="00:00:01,000",
        end_time="00:00:02,000",
        subtitle_start_index=1,
        subtitle_end_index=2,
        text="hello",
        score=score,
    )


def test_detect_movie_title_finds_title_in_question():
    assert detect_movie_title("In Orbit Seven, why is the air running out?", TITLES) == "Orbit Seven"


def test_detect_movie_title_works_without_leading_the():
    question = "What does Scar say in lion king?"
    assert detect_movie_title(question, TITLES) == "The Lion King"


def test_detect_movie_title_returns_none_when_unclear():
    assert detect_movie_title("Who is Aladeen?", TITLES) is None
    assert detect_movie_title("Compare the Lion King and Orbit Seven", TITLES) is None


def test_filter_relevant_drops_weak_results():
    results = [make_result(0.45), make_result(0.10), make_result(0.20)]
    kept = filter_relevant(results, min_score=0.20)
    assert [r.score for r in kept] == [0.45, 0.20]


def test_content_words_drop_common_words():
    words = content_words("What does Mufasa tell Simba about the stars?")
    assert words == ["mufasa", "simba", "star"]


def test_keyword_scores_favor_rare_words():
    texts = ["mufasa simba talk", "mufasa simba stars", "mufasa simba king"]
    scores = keyword_scores(["mufasa", "simba", "star"], texts)
    assert scores[0] == 0.0  # only has the common words
    assert scores[2] == 0.0
    assert scores[1] > 0.0  # has the rare word "stars"


def test_fuse_rankings_rewards_chunks_found_by_both_methods():
    order = fuse_rankings([["a", "b", "c"], ["c", "d"]])
    assert order == ["c", "a", "b", "d"]


def test_fuse_rankings_keeps_chunks_found_by_only_one_method():
    order = fuse_rankings([["a"], ["b"]])
    assert set(order) == {"a", "b"}


def test_keyword_search_finds_an_exact_word(sample_store):
    hits = keyword_search("Why is the lamp logbook missing a page?", collection=sample_store)
    assert [hit.chunk_id for hit in hits] == ["the_lighthouse_keeper_0001"]