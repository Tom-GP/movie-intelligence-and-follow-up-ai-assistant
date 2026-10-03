from rag.retriever import detect_movie_title, filter_relevant
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