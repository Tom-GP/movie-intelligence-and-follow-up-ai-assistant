"""The main agent: route the message, then send it to the right place."""

from chromadb.api.models.Collection import Collection

from agent.router import route_request
from models.schemas import AgentRequest, AgentResponse
from rag.llm import generate
from rag.qa import LLMFunction, answer_question
from rag.vector_store import list_movie_titles


def _or_not_given(value: str | None) -> str:
    return value if value else "(not given)"


def _handle_information(
    request: AgentRequest, collection: Collection | None, generate_fn: LLMFunction
) -> AgentResponse:
    answer = answer_question(
        request.query or request.original_message,
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
    collection: Collection | None = None,
    generate_fn: LLMFunction = generate,
) -> AgentResponse:
    """Process one user message and return the agent's response."""
    request = route_request(message, list_movie_titles(collection))

    if request.intent == "clarification":
        return AgentResponse(
            kind="clarification",
            text=request.clarification_question or "",
            request=request,
        )

    if request.intent == "email":
        return _handle_email(request)

    return _handle_information(request, collection, generate_fn)