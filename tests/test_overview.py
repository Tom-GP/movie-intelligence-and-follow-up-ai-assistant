from rag.overview import is_whole_movie_request, sample_movie_chunks, spread_indices


def test_spread_indices_picks_evenly():
    assert spread_indices(9, 3) == [0, 4, 8]


def test_spread_indices_returns_everything_when_there_are_few():
    assert spread_indices(5, 10) == [0, 1, 2, 3, 4]


def test_whole_movie_requests_are_recognized():
    assert is_whole_movie_request("Tell the story of Lion King in short?", "The Lion King")
    assert is_whole_movie_request("a summary of Orbit Seven", "Orbit Seven")
    assert is_whole_movie_request("What happens in Dust and Thunder?", "Dust and Thunder")


def test_specific_requests_are_not_whole_movie():
    assert not is_whole_movie_request(
        "Why did Tomas shut down the oxygen recycler?", "Orbit Seven"
    )
    assert not is_whole_movie_request(
        "a summary of the interrogation scene in Orbit Seven", "Orbit Seven"
    )


def test_sample_movie_chunks_returns_the_movies_chunks(sample_store):
    chunks = sample_movie_chunks("Orbit Seven", collection=sample_store)
    assert [chunk.chunk_id for chunk in chunks] == ["orbit_seven_0001"]