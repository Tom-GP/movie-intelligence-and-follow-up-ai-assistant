"""Ask a question about the movies.

Examples (run from the project root):
    python -m scripts.ask "Why did Tomas shut down the oxygen recycler?"
    python -m scripts.ask "What does Mufasa say about the stars?" --movie "The Lion King"
    python -m scripts.ask "What does Mufasa say about the stars?" --debug
"""

import argparse

from rag.llm import LLMError
from rag.qa import answer_question, format_timestamp
from rag.retriever import MIN_RELEVANCE_SCORE, find_movie_in_question, retrieve


def show_retrieved(question: str, movie: str | None, top_k: int | None) -> None:
    """Print the chunks the search found, including ones below the score cut-off."""
    movie = movie or find_movie_in_question(question)
    results = retrieve(question, movie_title=movie, top_k=top_k, min_score=0.0)

    print(f"--- Debug: movie filter = {movie} ---")
    for number, result in enumerate(results, start=1):
        kept = "sent to LLM" if result.score >= MIN_RELEVANCE_SCORE else "below cut-off"
        start = format_timestamp(result.start_time)
        end = format_timestamp(result.end_time)
        preview = result.text.replace("\n", " ")[:150]
        print(f"[{number}] score {result.score:.2f} ({kept}) | {result.movie_title} | {start}-{end}")
        print(f"    {preview}...")
    print("--- End debug ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the movies.")
    parser.add_argument("question")
    parser.add_argument("--movie", default=None, help="Only search this movie title")
    parser.add_argument("--top", type=int, default=None, help="How many chunks to retrieve")
    parser.add_argument("--debug", action="store_true", help="Show the retrieved chunks")
    args = parser.parse_args()

    if args.debug:
        show_retrieved(args.question, args.movie, args.top)

    try:
        answer = answer_question(args.question, movie_title=args.movie, top_k=args.top)
    except LLMError as error:
        print(f"LLM setup problem: {error}")
        return

    print(answer.format())
    if answer.movie_filter:
        print(f"\n(Searched only in: {answer.movie_filter})")


if __name__ == "__main__":
    main()