"""Prompts for the email-writing step."""

from models.schemas import AgentRequest
from rag.prompt import NOT_FOUND_TEXT, build_context
from rag.vector_store import SearchResult

EMAIL_SYSTEM_PROMPT = f"""You write short, friendly emails about movies using ONLY the numbered subtitle excerpts you are given.

Rules:
1. Use only facts that appear in the excerpts. Do not use outside knowledge and do not guess.
2. After each claim, add the number of the excerpt that supports it, like [1] or [2]. Only use numbers that exist.
3. Never write timestamps or a "Sources" list. The system adds the sources itself.
4. Subtitles often do not name the speakers. Do not invent who says what.
5. If the excerpts are spread across a whole movie, tell the story in order: what happens, who does what, and how it develops. Ignore song lyrics and repeated chants. Do not claim to know the ending unless an excerpt shows it.
6. The first line of your reply must be exactly in this form: Subject: <a short subject>
   Then leave one blank line, then write the email body.
7. The body is plain text with no markdown, about 80-200 words. Start with a short greeting such as "Hi,". End with exactly these two lines: "Best," and "Movie Assistant". Never use placeholders such as [Your Name] or [Recipient].
8. If the excerpts contain nothing relevant, reply with exactly this sentence and nothing else: {NOT_FOUND_TEXT}"""


def build_email_user_prompt(request: AgentRequest, results: list[SearchResult]) -> str:
    content = request.requested_content or "summary"
    topic = request.query or "the movie"
    movie = request.movie_title or "the movie"
    return (
        f"Write an email containing a {content} about: {topic}\n"
        f"Movie: {movie}\n\n"
        f"Subtitle excerpts:\n\n{build_context(results)}"
    )