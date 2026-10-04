"""Decides what the user wants: information, email, or clarification."""

import re

from models.schemas import AgentRequest
from rag.retriever import detect_movie_title

EMAIL_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# A message that STARTS with a request to send, such as:
# "Send me...", "Email Sarah...", "Can you email me...", "Please forward..."
EMAIL_COMMAND = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"|i\s+(?:want|need)\s+you\s+to\s+"
    r"|i(?:'d| would)\s+like\s+you\s+to\s+)?"
    r"(?:send|e-?mail|mail|forward)\b",
    re.IGNORECASE,
)

# "Summarize the scene AND SEND it to Bob"
EMAIL_AFTER_CONNECTOR = re.compile(
    r"\b(?:and|then)\s+(?:send|e-?mail|mail|forward)\b", re.IGNORECASE
)

# "send email", "send me an email", "write an email" ANYWHERE in the message
EMAIL_PHRASE = re.compile(
    r"\b(?:send|write|draft|compose)\s+(?:me\s+|us\s+)?(?:an?\s+|the\s+)?e-?mail\b",
    re.IGNORECASE,
)

# "hi", "hello there", "thanks", "help me"
GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|help)(?:\s+\w+)?[\s.!?]*$", re.IGNORECASE
)

EMPTY_MESSAGE_QUESTION = (
    "What would you like to know about the movies? You can ask about a scene, "
    "or ask me to email you a summary."
)
TOO_VAGUE_QUESTION = (
    "Could you tell me a bit more? For example: 'What does Mufasa say about the "
    "stars?' or 'Email me a summary of the interrogation scene in Orbit Seven.'"
)


def check_if_too_vague(text: str) -> str | None:
    """Return a clarification question if the message is empty or too short."""
    if not text:
        return EMPTY_MESSAGE_QUESTION
    if GREETING.match(text) or len(text.split()) < 2:
        return TOO_VAGUE_QUESTION
    return None


def is_email_request(text: str) -> bool:
    return bool(
        EMAIL_ADDRESS.search(text)
        or EMAIL_COMMAND.search(text)
        or EMAIL_AFTER_CONNECTOR.search(text)
        or EMAIL_PHRASE.search(text)
    )


def extract_recipient(text: str) -> str | None:
    """Return the first email address in the message, if any."""
    match = EMAIL_ADDRESS.search(text)
    return match.group(0) if match else None


def detect_requested_content(text: str) -> str | None:
    """What should the email contain? 'summary', 'analysis', or 'quote analysis'."""
    lowered = text.lower()
    if re.search(r"summar|recap|overview", lowered):
        return "summary"
    if "quote" in lowered and re.search(r"analy|meaning|explain", lowered):
        return "quote analysis"
    if "analy" in lowered:
        return "analysis"
    return None


def build_email_query(text: str) -> str:
    """Keep only the topic, removing the email address and the 'send me' part."""
    query = EMAIL_ADDRESS.sub("", text)
    query = EMAIL_COMMAND.sub("", query, count=1)
    query = EMAIL_PHRASE.sub("", query)
    query = re.sub(r"\s+(?:at|to|for)\b[\s.!?]*$", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+about\s+(?:it|this|that)[\s.!?]*$", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^\s*(?:me|us)\b\s*", "", query, flags=re.IGNORECASE)
    query = " ".join(query.split()).strip(" .,!?")
    return query or text


def route_request(message: str, known_titles: list[str]) -> AgentRequest:
    """Turn a user message into a structured AgentRequest."""
    text = " ".join(message.split())

    question = check_if_too_vague(text)
    if question:
        return AgentRequest(
            intent="clarification",
            original_message=message,
            clarification_question=question,
        )

    movie_title = detect_movie_title(text, known_titles)

    if is_email_request(text):
        return AgentRequest(
            intent="email",
            original_message=message,
            movie_title=movie_title,
            query=build_email_query(text),
            recipient_email=extract_recipient(text),
            requested_content=detect_requested_content(text),
        )

    return AgentRequest(
        intent="information",
        original_message=message,
        movie_title=movie_title,
        query=text,
    )