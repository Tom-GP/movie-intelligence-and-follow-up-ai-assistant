"""Answers questions with RAG. Citations are built from metadata, never by the LLM."""

import re
from dataclasses import dataclass
from typing import Callable

from chromadb.api.models.Collection import Collection

from rag.llm import generate
from rag.prompt import NOT_FOUND_TEXT, SYSTEM_PROMPT, build_user_prompt
from rag.retriever import MIN_RELEVANCE_SCORE, find_movie_in_question, retrieve
from rag.vector_store import SearchResult

NO_RESULTS_TEXT = "I could not find anything in the subtitles that matches your question."

# Matches [1], [2, 3], and so on.
MARKER = re.compile(r"\[([\d,\s]+)\]")


def format_timestamp(timestamp: str) -> str:
    """'00:32:11,847' -> '00:32:11' (the exact value stays in the metadata)."""
    return timestamp.split(",")[0]


@dataclass(frozen=True)
class Citation:
    """A source, copied from the stored metadata."""

    number: int
    chunk_id: str
    movie_title: str
    start_time: str
    end_time: str
    subtitle_start_index: int
    subtitle_end_index: int

    def label(self) -> str:
        start = format_timestamp(self.start_time)
        end = format_timestamp(self.end_time)
        return f"{self.movie_title} — {start}–{end}"


@dataclass(frozen=True)
class Answer:
    """The final result shown to the user."""

    text: str
    citations: list[Citation]
    found_evidence: bool
    movie_filter: str | None = None

    def format(self) -> str:
        if not self.citations:
            return self.text
        lines = [self.text, "", "Sources:"]
        lines += [f"- [{c.number}] {c.label()}" for c in self.citations]
        return "\n".join(lines)


def extract_cited_numbers(text: str, max_number: int) -> list[int]:
    """Find the [n] markers in the LLM's reply. Invalid numbers are ignored."""
    found: list[int] = []
    for match in MARKER.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                number = int(part)
                if 1 <= number <= max_number and number not in found:
                    found.append(number)
    return found


def build_citations(results: list[SearchResult], numbers: list[int]) -> list[Citation]:
    """Turn excerpt numbers into citations using the retrieved metadata."""
    citations = []
    for number in numbers:
        result = results[number - 1]
        citations.append(
            Citation(
                number=number,
                chunk_id=result.chunk_id,
                movie_title=result.movie_title,
                start_time=result.start_time,
                end_time=result.end_time,
                subtitle_start_index=result.subtitle_start_index,
                subtitle_end_index=result.subtitle_end_index,
            )
        )
    return citations


LLMFunction = Callable[[str, str], str]


def answer_from_results(
    question: str,
    results: list[SearchResult],
    movie_title: str | None = None,
    generate_fn: LLMFunction = generate,
    system_prompt: str = SYSTEM_PROMPT,
) -> Answer:
    """Ask the LLM to answer from chunks we already have, and attach citations."""
    if not results:
        return Answer(NO_RESULTS_TEXT, [], False, movie_title)

    reply = generate_fn(system_prompt, build_user_prompt(question, results)).strip()
    if reply.startswith(NOT_FOUND_TEXT):
        return Answer(NOT_FOUND_TEXT, [], False, movie_title)

    numbers = extract_cited_numbers(reply, max_number=len(results))
    if not numbers:
        # The LLM forgot the markers: fall back to listing every excerpt we gave it.
        numbers = list(range(1, len(results) + 1))

    return Answer(reply, build_citations(results, numbers), True, movie_title)


def answer_question(
    question: str,
    movie_title: str | None = None,
    top_k: int | None = None,
    min_score: float = MIN_RELEVANCE_SCORE,
    collection: Collection | None = None,
    generate_fn: LLMFunction = generate,
) -> Answer:
    """Retrieve evidence, ask the LLM, and attach citations."""
    if movie_title is None:
        movie_title = find_movie_in_question(question, collection)

    results = retrieve(
        question,
        movie_title=movie_title,
        top_k=top_k,
        min_score=min_score,
        collection=collection,
    )
    return answer_from_results(question, results, movie_title, generate_fn)