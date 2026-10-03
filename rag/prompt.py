"""The prompts we send to the LLM."""

from rag.vector_store import SearchResult

# The LLM must reply with exactly this sentence when the excerpts don't help.
NOT_FOUND_TEXT = "I could not find this in the subtitles."

SYSTEM_PROMPT = f"""You answer questions about movies using ONLY the numbered subtitle excerpts you are given.

Rules:
1. Use only facts that appear in the excerpts. Do not use outside knowledge about the movie and do not guess.
2. After each claim, add the number of the excerpt that supports it, like [1] or [2]. Only use numbers that exist.
3. Never write timestamps or a "Sources" list. The system adds the sources itself.
4. If the question quotes a line, look for that line in the excerpts, ignoring capital letters and punctuation. If you find it, answer from that excerpt.
5. Subtitles often do not say who is speaking. Name a speaker only when the excerpt shows it (for example "SIMBA: ..." or a name used in the dialogue). If the excerpt contains the line but does not say who speaks it, say that the subtitles do not name the speaker. You may then add what the surrounding dialogue suggests, clearly marked as an inference (for example "it is probably X, because ...").
6. Only if the excerpts contain nothing relevant to the question, reply with exactly this sentence and nothing else: {NOT_FOUND_TEXT}
7. Keep the answer short and clear (2-5 sentences)."""


def build_context(results: list[SearchResult]) -> str:
    """Number the excerpts [1], [2], ... so the LLM can point at them."""
    blocks = []
    for number, result in enumerate(results, start=1):
        blocks.append(f"[{number}] (movie: {result.movie_title})\n{result.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    return f"Subtitle excerpts:\n\n{build_context(results)}\n\nQuestion: {question}"