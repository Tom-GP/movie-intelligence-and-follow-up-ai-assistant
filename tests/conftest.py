"""Shared test setup: a small throw-away database with the 3 fake movies."""

import pytest

from config.settings import PROJECT_ROOT
from ingestion.ingest import ingest_file
from rag.vector_store import add_chunks, get_collection

SUBTITLES = PROJECT_ROOT / "data" / "subtitles"
SAMPLE_FILES = ["the_lighthouse_keeper.srt", "orbit_seven.srt", "dust_and_thunder.srt"]


@pytest.fixture(scope="session")
def sample_store(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("chroma_qa")
    collection = get_collection(db_path=db_path, collection_name="qa_test_movies")

    chunks = []
    for name in SAMPLE_FILES:
        chunks.extend(ingest_file(SUBTITLES / name, 30, 90))
    add_chunks(chunks, collection=collection)
    return collection