"""Streamlit user interface for the Movie Intelligence Assistant.

Run from the project root:
    streamlit run app.py
"""

import traceback
from pathlib import Path

import streamlit as st

from agent.agent import handle_message
from config.settings import settings
from ingestion.ingest import ingest_file
from ingestion.new_files import find_new_subtitle_files
from mcp_tools.email_client import EmailSendError, get_email_status
from rag.embeddings import get_embedding_model
from rag.llm import LLMError
from rag.vector_store import add_chunks, count_chunks, list_movie_titles

st.set_page_config(page_title="Movie Intelligence Assistant", page_icon="🎬")

EXAMPLES = [
    "Why did Tomas shut down the oxygen recycler?",
    'Who says "I know"?',
    "Tell the story of Lion King in short",
    "Email me a summary of Orbit Seven",
    "Send me a summary of the Dictator at you@example.com",
]


# ---------- remembered between reruns ----------

if "messages" not in st.session_state:
    st.session_state.messages = []  # the chat history
if "pending" not in st.session_state:
    st.session_state.pending = None  # a question the agent asked and is waiting on


# ---------- slow things, done once and remembered ----------


@st.cache_resource(show_spinner=False)
def preload_search_model() -> bool:
    """Load the embedding model once per Streamlit process (it is the slow part)."""
    get_embedding_model()
    return True


@st.cache_data(ttl=60, show_spinner=False)
def load_index_status() -> tuple[int, int, list[str]]:
    titles = list_movie_titles()
    return len(titles), count_chunks(), titles


@st.cache_data(ttl=300, show_spinner=False)
def load_email_status() -> str:
    try:
        return get_email_status()
    except EmailSendError as error:
        return f"unavailable: {error}"


def llm_status() -> str:
    provider = settings.llm_provider.lower()
    keys = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}
    if not settings.llm_model or not keys.get(provider):
        return "not configured ⚠️"
    return f"{provider} / {settings.llm_model} ✅"


# ---------- indexing new subtitle files from the page ----------


def run_indexing(paths: list[Path]) -> None:
    """Index only the new subtitle files, inside this running app."""
    try:
        with st.spinner("Indexing... (this can take a minute)"):
            preload_search_model()
            chunks = []
            for path in paths:
                chunks.extend(ingest_file(path))
            add_chunks(chunks)
    except Exception as error:  # a broken .srt file should not crash the page
        traceback.print_exc()
        st.error(f"Indexing failed: {error}")
        return
    st.cache_data.clear()
    st.rerun()


# ---------- the sidebar ----------


def render_sidebar() -> int:
    """Draw the sidebar. Returns the number of indexed chunks."""
    movies, chunks, indexed_titles = load_index_status()
    new_files = find_new_subtitle_files(settings.subtitles_dir, indexed_titles)
    email_status = load_email_status()
    email_icon = "✅" if email_status.startswith(("ready", "dry-run")) else "⚠️"

    with st.sidebar:
        st.header("Status")
        st.metric("Movies indexed", movies)
        st.metric("Vector chunks", chunks)
        rag_text = "Ready ✅" if chunks else "No data ⚠️"
        st.write(f"**RAG status:** {rag_text}")
        st.write(f"**LLM:** {llm_status()}")
        st.write(f"**MCP Email:** {email_icon} {email_status}")

        if new_files:
            names = ", ".join(path.name for path in new_files)
            st.warning(f"Not indexed yet: {names}")
        if st.button(
            "Index new subtitle files",
            disabled=not new_files,
            help="Adds subtitle files that are in data/subtitles but not in the database yet.",
        ):
            run_indexing(new_files)

        if st.button("Refresh status"):
            st.cache_data.clear()
            st.rerun()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

        with st.expander("Example messages"):
            for example in EXAMPLES:
                st.markdown(f"- {example}")
    return chunks


# ---------- showing messages ----------


def with_line_breaks(text: str) -> str:
    """Markdown ignores single line breaks; two spaces before one keeps it."""
    return text.replace("\n", "  \n")


def assistant_message(kind: str, text: str, sources: list[str] | None = None) -> dict:
    return {"role": "assistant", "kind": kind, "text": text, "sources": sources or []}


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["text"])
            return

        kind = message["kind"]
        if kind == "error":
            st.error(message["text"])
        elif kind == "email_sent":
            first_line, _, rest = message["text"].partition("\n")
            st.success(first_line)
            if rest.strip():
                st.markdown(with_line_breaks(rest))
        else:
            st.markdown(with_line_breaks(message["text"]))

        if message["sources"]:
            st.markdown("**Sources**")
            st.markdown("\n".join(f"- {source}" for source in message["sources"]))


# ---------- talking to the agent ----------


def ask_agent(text: str) -> dict:
    """Send the message to the agent and turn its response into a chat message."""
    try:
        response = handle_message(text, pending=st.session_state.pending)
    except LLMError as error:
        st.session_state.pending = None
        return assistant_message("error", f"LLM setup problem: {error}")
    except Exception as error:  # keep the page alive whatever happens
        traceback.print_exc()  # details go to the terminal
        st.session_state.pending = None
        return assistant_message("error", f"Something went wrong: {error}")

    # If the agent asked a question, remember the half-finished request.
    st.session_state.pending = response.request if response.kind == "clarification" else None
    return assistant_message(response.kind, response.text, response.sources)


# ---------- the page ----------

st.title("🎬 Movie Intelligence Assistant")
st.caption(
    "Answers come only from the subtitle files, with sources. "
    "Emails are sent through an MCP email tool."
)

chunk_count = render_sidebar()
if chunk_count == 0:
    st.warning("No movies are indexed yet. Run: python -m scripts.ingest_movies --reset")

for past_message in st.session_state.messages:
    render_message(past_message)

user_text = st.chat_input("Ask about a movie, or ask me to email you a summary...")

# The page is already visible here. Loading the model takes a while only the
# first time after Streamlit starts.
with st.spinner("Loading the search model (only the first time after Streamlit starts)..."):
    preload_search_model()

if user_text:
    user_message = {"role": "user", "text": user_text}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.spinner("Thinking..."):
        reply = ask_agent(user_text)
    st.session_state.messages.append(reply)
    render_message(reply)