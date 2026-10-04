"""The main agent: route the message, check for ambiguity, then act."""

from chromadb.api.models.Collection import Collection

from agent.ambiguity import MISSING_RECIPIENT, apply_reply, check_ambiguity, load_chunk
from agent.email_flow import SendFunction, run_email_request
from agent.router import route_request
from config.settings import settings
from mcp_tools.email_client import send_email_via_mcp
from models.schemas import AgentRequest, AgentResponse
from rag.llm import generate
from rag.overview import is_whole_movie_request, sample_movie_chunks
from rag.prompt import SUMMARY_SYSTEM_PROMPT
from rag.qa import LLMFunction, answer_from_results, answer_question
from rag.vector_store import list_movie_titles

BAD_ADDRESS_QUESTION = (
    "That does not look like a valid email address (it needs an @ and a domain, "
    "like name@example.com). What email address should I use?"
)


def _looks_like_a_bad_address(reply: str) -> bool:
    """One word containing '@' or '.', like 'to2.com': an address attempt that failed."""
    token = reply.strip()
    return bool(token) and " " not in token and ("@" in token or "." in token)


def _handle_information(
    request: AgentRequest, collection: Collection | None, generate_fn: LLMFunction
) -> AgentResponse:
    question = request.query or request.original_message

    pinned_chunk = load_chunk(request.chunk_id, collection) if request.chunk_id else None
    if pinned_chunk is not None:
        # A specific scene was identified: answer from exactly that scene.
        answer = answer_from_results(question, [pinned_chunk], request.movie_title, generate_fn)
    elif request.movie_title and is_whole_movie_request(question, request.movie_title):
        # "Tell the story of <movie>": use chunks spread over the whole film,
        # with a prompt made for summaries.
        results = sample_movie_chunks(request.movie_title, collection=collection)
        answer = answer_from_results(
            question,
            results,
            request.movie_title,
            generate_fn,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
        )
    else:
        answer = answer_question(
            question,
            movie_title=request.movie_title,
            collection=collection,
            generate_fn=generate_fn,
        )

    return AgentResponse(
        kind="answer",
        text=answer.text,
        sources=[f"[{c.number}] {c.label()}" for c in answer.citations],
        request=request,
    )


def handle_message(
    message: str,
    pending: AgentRequest | None = None,
    collection: Collection | None = None,
    generate_fn: LLMFunction = generate,
    send_email_fn: SendFunction = send_email_via_mcp,
) -> AgentResponse:
    """Process one user message.

    `pending` is the request from the previous turn if the agent asked a
    clarification question and is now waiting for the answer.
    """
    known_titles = list_movie_titles(collection)
    default_recipient = settings.default_recipient_email

    request = None
    if pending is not None:
        request = apply_reply(pending, message, known_titles, default_recipient, collection)
        if (
            request is None
            and MISSING_RECIPIENT in pending.missing_fields
            and _looks_like_a_bad_address(message)
        ):
            # Keep the half-finished request and ask for the address again.
            retry = pending.model_copy(update={"clarification_question": BAD_ADDRESS_QUESTION})
            return AgentResponse(kind="clarification", text=BAD_ADDRESS_QUESTION, request=retry)

    if request is None:
        request = route_request(message, known_titles)

    request = check_ambiguity(request, collection, default_recipient)

    if request.intent == "clarification":
        return AgentResponse(
            kind="clarification",
            text=request.clarification_question or "",
            request=request,
        )

    if request.intent == "email":
        return run_email_request(request, collection, generate_fn, send_email_fn)

    return _handle_information(request, collection, generate_fn)