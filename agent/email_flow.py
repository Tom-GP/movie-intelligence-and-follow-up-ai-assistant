"""The email workflow: find evidence, write the email, send it through MCP."""

import re
from dataclasses import dataclass
from typing import Callable

from chromadb.api.models.Collection import Collection

from agent.ambiguity import load_chunk
from agent.prompts import EMAIL_SYSTEM_PROMPT, build_email_user_prompt
from mcp_tools.email_client import EmailResult, EmailSendError
from models.schemas import AgentRequest, AgentResponse
from rag.overview import is_whole_movie_request, sample_movie_chunks
from rag.prompt import NOT_FOUND_TEXT
from rag.qa import (
    NO_RESULTS_TEXT,
    Citation,
    LLMFunction,
    build_citations,
    extract_cited_numbers,
)
from rag.retriever import retrieve
from rag.vector_store import SearchResult

SendFunction = Callable[[str, str, str], EmailResult]

SUBJECT_LINE = re.compile(r"^\s*subject\s*:\s*(.+?)\s*$", re.IGNORECASE)
MARKERS = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")  # [1] or [1, 2]


@dataclass(frozen=True)
class EmailDraft:
    subject: str
    body: str  # without the Sources list
    citations: list[Citation]


def fallback_subject(request: AgentRequest) -> str:
    movie = request.movie_title or "Movie"
    content = (request.requested_content or "summary").title()
    return f"{movie} — {content}"


def parse_draft(reply: str, fallback: str) -> tuple[str, str]:
    """Split the LLM's reply into (subject, body)."""
    lines = reply.strip().splitlines()
    subject = fallback
    body_lines = lines

    if lines:
        match = SUBJECT_LINE.match(lines[0])
        if match:
            subject = MARKERS.sub("", match.group(1)).strip() or fallback
            body_lines = lines[1:]

    return subject, "\n".join(body_lines).strip()


def compose_body(body: str, citations: list[Citation]) -> str:
    """The text that is actually emailed: the body plus a Sources list from metadata."""
    lines = [body, "", "Sources:"]
    lines += [f"- [{c.number}] {c.label()}" for c in citations]
    return "\n".join(lines)


def gather_evidence(
    request: AgentRequest, collection: Collection | None
) -> list[SearchResult]:
    """Find the subtitle chunks the email should be based on."""
    if request.chunk_id:
        chunk = load_chunk(request.chunk_id, collection)
        return [chunk] if chunk else []

    topic = request.query or request.original_message
    if request.movie_title and is_whole_movie_request(topic, request.movie_title):
        return sample_movie_chunks(request.movie_title, collection=collection)

    return retrieve(topic, movie_title=request.movie_title, collection=collection)


def draft_email(
    request: AgentRequest, results: list[SearchResult], generate_fn: LLMFunction
) -> EmailDraft | None:
    """Ask the LLM to write the email. Returns None if it found nothing to say."""
    reply = generate_fn(EMAIL_SYSTEM_PROMPT, build_email_user_prompt(request, results)).strip()
    if reply.startswith(NOT_FOUND_TEXT):
        return None

    subject, body = parse_draft(reply, fallback_subject(request))
    if not body:
        return None

    numbers = extract_cited_numbers(body, max_number=len(results))
    if not numbers:
        numbers = list(range(1, len(results) + 1))
    return EmailDraft(subject, body, build_citations(results, numbers))


def run_email_request(
    request: AgentRequest,
    collection: Collection | None,
    generate_fn: LLMFunction,
    send_fn: SendFunction,
) -> AgentResponse:
    """Evidence -> draft -> send -> confirmation with sources."""
    if not request.recipient_email:
        return AgentResponse(
            kind="error", text="I don't have a recipient email address.", request=request
        )

    results = gather_evidence(request, collection)
    if not results:
        return AgentResponse(
            kind="answer", text=f"{NO_RESULTS_TEXT} No email was sent.", request=request
        )

    draft = draft_email(request, results, generate_fn)
    if draft is None:
        return AgentResponse(
            kind="answer",
            text="I could not find enough in the subtitles to write that email, so nothing was sent.",
            request=request,
        )

    full_body = compose_body(draft.body, draft.citations)
    try:
        result = send_fn(request.recipient_email, draft.subject, full_body)
    except EmailSendError as error:
        return AgentResponse(
            kind="error", text=f"The email could not be sent: {error}", request=request
        )

    text = (
        f"{result.message}\n\n"
        f"To: {request.recipient_email}\n"
        f"Subject: {draft.subject}\n\n"
        f"{draft.body}"
    )
    return AgentResponse(
        kind="email_sent",
        text=text,
        sources=[f"[{c.number}] {c.label()}" for c in draft.citations],
        request=request,
    )