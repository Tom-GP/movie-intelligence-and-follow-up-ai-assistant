"""The main agent: route the message, check for ambiguity, then act."""

from chromadb.api.models.Collection import Collection

from agent.ambiguity import apply_reply, check_ambiguity, load_chunk
from agent.router import route_request
from config.settings import settings
from models.schemas import AgentRequest, AgentResponse
from rag.llm import generate
from rag.qa import LLMFunction, answer_from_results, answer_question
from rag.vector_store import list_movie_titles


def _or_not_given(value: str | None) -> str:
    return value if value else "(not given)"


def _handle_information(
    request: AgentRequest, collection: Collection | None, generate_fn: LLMFunction
) -> AgentResponse:
    question = request.query or request.original_message

    pinned_chunk = load_chunk(request.chunk_id, collection) if request.chunk_id else None
    if pinned_chunk is not None:
        # A specific scene was identified: answer from exactly that scene.
        answer = answer_from_results(
            question, [pinned_chunk], request.movie_title, generate_fn
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


def _handle_email(request: AgentRequest) -> AgentResponse:
    """Placeholder: Phase 6 will retrieve evidence and send the email via MCP."""
    text = (
        "I understood this as an email request.\n"
        f"  Recipient: {_or_not_given(request.recipient_email)}\n"
        f"  Movie: {_or_not_given(request.movie_title)}\n"
        f"  Topic: {_or_not_given(request.query)}\n"
        f"  Content type: {_or_not_given(request.requested_content)}\n"
        "Sending real emails is built in Phase 6, so nothing was sent."
    )
    return AgentResponse(kind="email_pending", text=text, request=request)


def handle_message(
    message: str,
    pending: AgentRequest | None = None,
    collection: Collection | None = None,
    generate_fn: LLMFunction = generate,
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
        return _handle_email(request)

    return _handle_information(request, collection, generate_fn)