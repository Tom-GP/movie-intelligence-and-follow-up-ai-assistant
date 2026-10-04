"""Checks for missing or ambiguous information before the agent acts."""

import re
from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

from agent.router import extract_recipient
from models.schemas import AgentRequest
from rag.qa import format_timestamp
from rag.retriever import detect_movie_title
from rag.vector_store import SearchResult, get_collection, search_chunks

MAX_OPTIONS = 8  # most choices we list in one question
SEARCH_DEPTH = 15  # how many chunks we look at to find candidate movies
CLEAR_MATCH_SCORE = 0.40  # a movie must score at least this to count as relevant
AMBIGUITY_MARGIN = 0.08  # movies within this of the best score count as "tied"
MAX_REPLY_WORDS = 8  # a longer reply is treated as a brand-new question

# Names of the things we can ask about
MISSING_MOVIE = "movie_title"
MISSING_RECIPIENT = "recipient_email"
MISSING_SCENE = "scene"

QUOTE_PATTERN = re.compile(r'["“]([^"“”]{2,})["”]')  # text inside double quotes
ME_PATTERN = re.compile(r"\b(?:me|myself)\b|\bmy\s+(?:email|inbox)\b", re.IGNORECASE)


@dataclass(frozen=True)
class QuoteMatch:
    """A chunk that contains a quoted phrase."""

    chunk_id: str
    movie_title: str
    start_time: str
    end_time: str
    start_seconds: float


# ---------- Quotes ----------


def extract_quote(text: str) -> str | None:
    """Return the text inside double quotes, if any."""
    match = QUOTE_PATTERN.search(text)
    return match.group(1).strip() if match else None


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation, so 'I KNOW!' matches 'I know.'"""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def find_quote_matches(
    quote: str,
    movie_title: str | None = None,
    collection: Collection | None = None,
) -> list[QuoteMatch]:
    """Find every chunk that contains the quote as a whole phrase."""
    collection = get_collection() if collection is None else collection
    where = {"movie_title": movie_title} if movie_title else None
    data = collection.get(where=where, include=["documents", "metadatas"])

    needle = f" {normalize_text(quote)} "
    matches: list[QuoteMatch] = []
    for text, meta in zip(data["documents"], data["metadatas"]):
        if needle in f" {normalize_text(text)} ":
            matches.append(
                QuoteMatch(
                    chunk_id=str(meta["chunk_id"]),
                    movie_title=str(meta["movie_title"]),
                    start_time=str(meta["start_time"]),
                    end_time=str(meta["end_time"]),
                    start_seconds=float(meta["start_seconds"]),
                )
            )
    matches.sort(key=lambda m: (m.movie_title, m.start_seconds))
    return matches


# ---------- Which movie? ----------


def choose_candidates(
    hits: list[SearchResult],
    margin: float = AMBIGUITY_MARGIN,
    min_score: float = CLEAR_MATCH_SCORE,
) -> list[str]:
    """From search hits, return the movies that clearly match, best first."""
    best_score: dict[str, float] = {}
    for hit in hits:
        best_score[hit.movie_title] = max(best_score.get(hit.movie_title, 0.0), hit.score)

    if not best_score:
        return []
    top = max(best_score.values())
    if top < min_score:
        return []  # nothing matches clearly

    close = [(title, score) for title, score in best_score.items() if score >= top - margin]
    close.sort(key=lambda item: item[1], reverse=True)
    return [title for title, _ in close]


def find_candidate_movies(
    query: str, collection: Collection | None = None
) -> list[str]:
    hits = search_chunks(query, top_k=SEARCH_DEPTH, collection=collection)
    return choose_candidates(hits)


def load_chunk(chunk_id: str, collection: Collection | None = None) -> SearchResult | None:
    """Load one stored chunk by its ID (used once a scene has been identified)."""
    collection = get_collection() if collection is None else collection
    data = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not data["ids"]:
        return None
    meta = data["metadatas"][0]
    return SearchResult(
        chunk_id=str(meta["chunk_id"]),
        movie_id=str(meta["movie_id"]),
        movie_title=str(meta["movie_title"]),
        start_time=str(meta["start_time"]),
        end_time=str(meta["end_time"]),
        subtitle_start_index=int(meta["subtitle_start_index"]),
        subtitle_end_index=int(meta["subtitle_end_index"]),
        text=data["documents"][0],
        score=1.0,
    )


# ---------- Building clarification requests ----------


def _ask(
    request: AgentRequest,
    question: str,
    missing: list[str],
    options: list[str] | None = None,
) -> AgentRequest:
    """Turn a request into a clarification request that remembers what it was."""
    return request.model_copy(
        update={
            "intent": "clarification",
            "pending_intent": request.intent,
            "clarification_question": question,
            "missing_fields": missing,
            "options": options or [],
        }
    )


def _choice_question(intro: str, labels: list[str], footer: str) -> str:
    lines = [intro]
    lines += [f"{number}. {label}" for number, label in enumerate(labels, start=1)]
    lines.append(footer)
    return "\n".join(lines)


def _unique_titles(matches: list[QuoteMatch]) -> list[str]:
    titles: list[str] = []
    for match in matches:
        if match.movie_title not in titles:
            titles.append(match.movie_title)
    return titles


def _ask_which_movie_for_quote(
    request: AgentRequest, matches: list[QuoteMatch]
) -> AgentRequest:
    titles = _unique_titles(matches)
    shown = titles[:MAX_OPTIONS]
    labels = []
    for title in shown:
        count = sum(1 for m in matches if m.movie_title == title)
        labels.append(f"{title} ({count} scene{'s' if count != 1 else ''})")

    footer = "Reply with a number or a movie title."
    if len(titles) > len(shown):
        footer = f"Showing {len(shown)} of {len(titles)}. Reply with a number or type any movie title."

    intro = f"I found that quote in {len(titles)} movies. Which one do you mean?"
    return _ask(request, _choice_question(intro, labels, footer), [MISSING_MOVIE], shown)


def _ask_which_scene(request: AgentRequest, matches: list[QuoteMatch]) -> AgentRequest:
    scenes = matches[:MAX_OPTIONS]
    labels = [
        f"{format_timestamp(m.start_time)}–{format_timestamp(m.end_time)}" for m in scenes
    ]
    intro = f"I found that quote in {len(matches)} scenes of {matches[0].movie_title}. Which one do you mean?"
    request = request.model_copy(update={"movie_title": matches[0].movie_title})
    return _ask(request, _choice_question(intro, labels, "Reply with a number."), [MISSING_SCENE], labels)


def _missing_info_question(
    request: AgentRequest, movie_missing: bool, recipient_missing: bool
) -> str:
    if movie_missing and recipient_missing:
        return (
            "Which movie and scene would you like me to use, "
            "and what email address should I send it to?"
        )
    if movie_missing:
        topic = request.query
        return f'Which movie should I use for "{topic}"?' if topic else "Which movie do you mean?"

    what = request.requested_content or "email"
    hint = ""
    if ME_PATTERN.search(request.original_message):
        hint = ' (Tip: set DEFAULT_RECIPIENT_EMAIL in your .env file so I know what "me" means.)'
    return f"What email address should I send the {what} to?{hint}"


# ---------- The main check ----------


def check_ambiguity(
    request: AgentRequest,
    collection: Collection | None = None,
    default_recipient: str | None = None,
) -> AgentRequest:
    """Return the request unchanged (or completed), or a clarification request."""
    if request.intent == "clarification":
        return request

    # 1. A quoted phrase: does it match several movies or several scenes?
    quote = extract_quote(request.original_message)
    if quote and not request.chunk_id:
        matches = find_quote_matches(quote, request.movie_title, collection)
        if len(_unique_titles(matches)) > 1:
            return _ask_which_movie_for_quote(request, matches)
        if len(matches) > 1:
            return _ask_which_scene(request, matches)
        if len(matches) == 1:  # exactly one scene: pin it
            request = request.model_copy(
                update={"chunk_id": matches[0].chunk_id, "movie_title": matches[0].movie_title}
            )

    # 2. Which movie?
    movie_missing = False
    if request.movie_title is None:
        candidates = find_candidate_movies(request.query or request.original_message, collection)
        if len(candidates) > 1:
            question = _choice_question(
                "I found relevant scenes in more than one movie. Which movie do you mean?",
                candidates[:MAX_OPTIONS],
                "Reply with a number or a movie title.",
            )
            return _ask(request, question, [MISSING_MOVIE], candidates[:MAX_OPTIONS])
        if len(candidates) == 1:
            request = request.model_copy(update={"movie_title": candidates[0]})
        elif request.intent == "email":
            movie_missing = True

    # 3. Who gets the email?
    recipient_missing = False
    if request.intent == "email" and not request.recipient_email:
        if default_recipient and ME_PATTERN.search(request.original_message):
            request = request.model_copy(update={"recipient_email": default_recipient})
        else:
            recipient_missing = True

    # 4. Ask for everything that is still missing, in one question.
    if movie_missing or recipient_missing:
        missing = []
        if movie_missing:
            missing.append(MISSING_MOVIE)
        if recipient_missing:
            missing.append(MISSING_RECIPIENT)
        return _ask(request, _missing_info_question(request, movie_missing, recipient_missing), missing)

    return request


# ---------- Reading the user's answer ----------


def parse_choice_number(text: str) -> int | None:
    stripped = text.strip().rstrip(".!")
    return int(stripped) if stripped.isdigit() else None


def pick_option(text: str, options: list[str]) -> str | None:
    """Match a reply like '2' or 'orbit seven' to one of the offered options."""
    number = parse_choice_number(text)
    if number is not None and 1 <= number <= len(options):
        return options[number - 1]
    for option in options:
        if text.strip().lower() == option.lower():
            return option
    return None


def apply_reply(
    pending: AgentRequest,
    reply: str,
    known_titles: list[str],
    default_recipient: str | None = None,
    collection: Collection | None = None,
) -> AgentRequest | None:
    """Fill in what the user's reply provides.

    Returns the updated request, or None if the reply doesn't answer our question
    (then the agent treats the reply as a brand-new message).
    """
    if pending.intent != "clarification" or not pending.pending_intent:
        return None

    text = " ".join(reply.split())
    short_reply = len(text.split()) <= MAX_REPLY_WORDS
    updates: dict[str, str] = {}

    if MISSING_RECIPIENT in pending.missing_fields:
        address = extract_recipient(text)
        if address:
            updates["recipient_email"] = address.lower()
        elif short_reply and default_recipient and ME_PATTERN.search(text):
            updates["recipient_email"] = default_recipient

    if MISSING_MOVIE in pending.missing_fields and short_reply:
        title = pick_option(text, pending.options) or detect_movie_title(text, known_titles)
        if title:
            updates["movie_title"] = title

    if MISSING_SCENE in pending.missing_fields and short_reply:
        number = parse_choice_number(text)
        quote = extract_quote(pending.original_message)
        scenes = find_quote_matches(quote, pending.movie_title, collection)[:MAX_OPTIONS] if quote else []
        if number is not None and 1 <= number <= len(scenes):
            updates["chunk_id"] = scenes[number - 1].chunk_id
            updates["movie_title"] = scenes[number - 1].movie_title

    if not updates:
        return None

    return pending.model_copy(
        update={
            **updates,
            "intent": pending.pending_intent,
            "pending_intent": None,
            "clarification_question": None,
            "missing_fields": [],
            "options": [],
        }
    )