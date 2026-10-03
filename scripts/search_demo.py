"""Try a search by hand.

Examples (run from the project root):
    python -m scripts.search_demo "a dog waiting at the train station"
    python -m scripts.search_demo "stars and kings" --movie "The Lion King"
"""

import argparse

from rag.vector_store import list_movie_titles, search_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the subtitle database.")
    parser.add_argument("query", help="Your question or a quote")
    parser.add_argument("--movie", default=None, help="Only search this movie title")
    parser.add_argument("--top", type=int, default=3, help="How many results")
    args = parser.parse_args()

    hits = search_chunks(args.query, top_k=args.top, movie_title=args.movie)
    if not hits:
        print("No results. Movies in the database:", list_movie_titles())
        return

    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:.2f}] {hit.movie_title} | {hit.start_time} --> {hit.end_time}")
        print(f"   {hit.text[:200]}...\n")


if __name__ == "__main__":
    main()