import math

import pytest

from config.settings import PROJECT_ROOT
from ingestion.ingest import ingest_file
from rag.embeddings import embed_query, embed_texts
from rag.vector_store import (
    add_chunks,
    count_chunks,
    get_collection,
    list_movie_titles,
    reset_collection,
    search_chunks,
)

SUBTITLES = PROJECT_ROOT / "data" / "subtitles"
SAMPLE_FILES = ["the_lighthouse_keeper.srt", "orbit_seven.srt", "dust_and_thunder.srt"]


def load_sample_chunks():
    chunks = []
    for name in SAMPLE_FILES:
        chunks.extend(ingest_file(SUBTITLES / name, 30, 90))
    return chunks


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A throw-away vector database holding only the 3 fake sample movies."""
    db_path = tmp_path_factory.mktemp("chroma_test")
    collection = get_collection(db_path=db_path, collection_name="test_movies")
    add_chunks(load_sample_chunks(), collection=collection)
    return collection


def test_embeddings_return_one_vector_per_text():
    vectors = embed_texts(["hello there", "general Kenobi"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


def test_embeddings_are_normalized():
    vector = embed_query("A dog waits at the station.")
    length = math.sqrt(sum(x * x for x in vector))
    assert length == pytest.approx(1.0, abs=1e-3)


def test_similar_meaning_is_closer():
    cat, kitten, stocks = embed_texts(
        [
            "The cat sits on the mat.",
            "A kitten is resting on the rug.",
            "Stock markets fell sharply today.",
        ]
    )
    assert dot(cat, kitten) > dot(cat, stocks)


def test_add_chunks_stores_all_chunks(store):
    assert count_chunks(store) == 3


def test_adding_same_chunks_twice_does_not_duplicate(store):
    add_chunks(load_sample_chunks(), collection=store)
    assert count_chunks(store) == 3


def test_list_movie_titles(store):
    assert list_movie_titles(store) == [
        "Dust and Thunder",
        "Orbit Seven",
        "The Lighthouse Keeper",
    ]


def test_exact_quote_finds_right_movie(store):
    hits = search_chunks(
        "Then why is the lamp logbook missing a page?", top_k=1, collection=store
    )
    assert hits[0].movie_title == "The Lighthouse Keeper"


def test_semantic_search_finds_right_movie(store):
    hits = search_chunks(
        "someone stole a girl's horse and sold it", top_k=1, collection=store
    )
    assert hits[0].movie_title == "Dust and Thunder"


def test_movie_filter_limits_results(store):
    hits = search_chunks(
        "someone stole a horse", movie_title="Orbit Seven", collection=store
    )
    assert len(hits) == 1
    assert hits[0].movie_title == "Orbit Seven"


def test_timestamps_and_indexes_are_preserved(store):
    hit = search_chunks(
        "lamp logbook missing a page fingerprints", top_k=1, collection=store
    )[0]
    assert hit.chunk_id == "the_lighthouse_keeper_0001"
    assert hit.start_time == "00:01:04,200"
    assert hit.end_time == "00:01:50,000"
    assert hit.subtitle_start_index == 2
    assert hit.subtitle_end_index == 10


def test_reset_collection_empties_it(tmp_path):
    collection = get_collection(db_path=tmp_path, collection_name="reset_test")
    add_chunks(load_sample_chunks(), collection=collection)
    assert count_chunks(collection) == 3

    fresh = reset_collection(db_path=tmp_path, collection_name="reset_test")
    assert count_chunks(fresh) == 0


def test_blank_query_returns_nothing(store):
    assert search_chunks("   ", collection=store) == []