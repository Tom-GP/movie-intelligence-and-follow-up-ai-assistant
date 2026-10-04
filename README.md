# Movie Intelligence & Follow-up Assistant

An AI assistant that reads movie subtitle files (`.srt`) and:

1. **Answers questions** about movies, with exact citations (movie title + timestamp range).
2. **Sends emails** (scene summaries, quote analysis) through an MCP email tool.
3. **Asks instead of guessing** when the movie, scene, quote, or recipient is unclear.
4. Has a simple **Streamlit** web page with a status sidebar.

The key design rule: the assistant never invents movie facts or sources. Every
answer is built from retrieved subtitle text, and every citation is copied from
stored metadata, never written by the language model.

## How it works

```
 .srt files
     |  parse -> clean -> time-aware chunks (30-45 s, with timestamps)
     v
 Vector database (Chroma): text + embedding + {movie, start, end}
     ^
     |  hybrid search: meaning (embeddings) + exact words (keywords)
     |
 User --> Streamlit / terminal --> Agent
                                    |-- router (rules): information / email / clarification
                                    |-- ambiguity checks: movie? scene? quote? recipient?
                                    |-- information --> RAG answer + citations
                                    '-- email --> RAG draft --> MCP client --> MCP email server --> SMTP
```

- **Chunking** keeps subtitle boundaries and exact timestamps (target 30–45 seconds per chunk).
- **Citations** are numbered excerpts `[1]`, `[2]`: the model only says *which*
  excerpt supports a claim, and the code looks up the movie and times.
- **The router** is made of explicit rules, so it is predictable and testable.
- **The MCP email server** is the only code that knows the email password.

## Requirements

- Python 3.11 or newer
- An LLM API key (OpenAI or Anthropic)
- For real emails: a Gmail account with 2-Step Verification and an App Password

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # macOS / Linux

# 2. Install the packages
pip install -r requirements.txt

# 3. Create your settings file, then open .env and fill it in
copy .env.example .env              # Windows
cp .env.example .env                # macOS / Linux

# 4. Put your .srt files in data/subtitles/ (sample movies are included),
#    then build the search index
python -m scripts.ingest_movies --reset
```

The file name becomes the movie title: `big_fish.srt` is shown as "Big Fish".

## Run it

| What | Command |
|---|---|
| Web page | `streamlit run app.py` |
| Chat in the terminal | `python -m scripts.chat` |
| One question | `python -m scripts.ask "your question" --debug` |
| Scripted demo (no real email) | `python -m scripts.demo` |
| Agent intent inspector | `python -m scripts.agent_demo "your query"` |
| Manual search test | `python -m scripts.search_demo "search query"` |
| Add new movies | copy `.srt` files to `data/subtitles/`, then `python -m scripts.ingest_movies` (or use the "Index new subtitle files" button in the web page) |
| Rebuild everything | `python -m scripts.ingest_movies --reset` |
| All tests | `python -m pytest` |
| Test the email server alone | `python -m scripts.test_mcp_server` |

There is **no separate step to start the MCP server**: the agent starts it
automatically when it needs to send an email.

Use `--reset` when you change the chunk size, the cleaner, or remove a movie.
After indexing from the terminal, restart Streamlit.

## Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | `openai` or `anthropic`, and the exact model name (here, gpt-4o-mini) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Key for your provider |
| `EMBEDDING_MODEL_NAME` | Local model that turns text into vectors (here, all-MiniLM-L6-v2) |
| `HF_HUB_OFFLINE` | `1` = skip network checks once the model is downloaded (faster start) |
| `VECTOR_DB_PATH`, `COLLECTION_NAME` | Where the database is stored |
| `SUBTITLES_DIR` | Folder with the `.srt` files |
| `TOP_K` | How many chunks the search keeps per question (here, 8)|
| `CHUNK_MIN_SECONDS`, `CHUNK_MAX_SECONDS` | Chunk length (here, 30-45 seconds) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM` | Email account (used only by the MCP email server) (here, smtp.gmail.com, 587, your.address@gmail.com, your-16-char-app-password, your.address@gmail.com) |
| `DEFAULT_RECIPIENT_EMAIL` | What "email me" means |
| `EMAIL_DRY_RUN` | `true` = emails are checked and previewed but **not sent**; set `false` to really send |

Never commit `.env`. It is listed in `.gitignore`.

### Gmail App Password (for real emails)

A normal Gmail password does not work for programs. Turn on 2-Step Verification
in your Google account, create an App Password at
myaccount.google.com/apppasswords, and put the 16 characters (no spaces) in
`SMTP_PASSWORD`. Set `EMAIL_DRY_RUN=false` and run
`python -m scripts.test_mcp_server --real your.address@gmail.com` to send a test.

## Example queries

Using movie subtitles in `data/subtitles/`:

| Type this | What happens |
|---|---|
| `Why did Tomas shut down the oxygen recycler?` | Answer from the scene, with `Orbit Seven` and its timestamp range |
| `In Dust and Thunder, who is Abel Crane?` | The movie in the message limits the search |
| `Who says "I know"?` | The quote is in several movies, so the assistant asks which one |
| `Summarize Orbit Seven` / `Tell the story of Orbit Seven in short` | A summary built from chunks spread over the whole movie |
| `Email me a summary of Orbit Seven` | Writes an email, sends it through the MCP tool, shows a preview and sources |
| `Send the scene to John` | Asks which movie and scene, and for an email address |
| `What is the capital of France?` | "Could not find" (no made-up answer) |

## Demo script (about 5 minutes)

1. Start with `streamlit run app.py`. Point at the **sidebar**: movies, chunks, RAG status, LLM, MCP Email.
2. Ask `Why did Tomas shut down the oxygen recycler?`. Show the answer and the **Sources** line, and say that the sources come from stored metadata, not from the model.
3. Ask `Who says "I know"?`. The assistant **asks which movie**. Reply with a number or a title and show the cited answer.
4. Ask `What is the capital of France?`. It admits it cannot find an answer.
5. Ask `Email me a summary of Orbit Seven`. Show the confirmation, the email text, and the sources.
6. Ask `Send the scene to John`. Show the clarification question instead of a guess.

For a recorded run without the web page: `python -m scripts.demo`.

## Project layout

```
app.py                    Streamlit web interface
config/settings.py        Loads every setting from .env
ingestion/                .srt parsing, cleaning, time-aware chunking, new file finder
rag/                      Embeddings, vector store, hybrid search, prompts, QA with citations
agent/                    Router, ambiguity checks, email workflow, main agent
mcp_tools/                MCP email server and agent-side client
models/schemas.py         Pydantic models (AgentRequest, AgentResponse)
scripts/                  CLI tools (ingest, ask, chat, demo, agent_demo, search_demo, test_mcp_server)
tests/                    Automated unit test suite (pytest)
data/subtitles/           Input .srt files
data/chroma_db/           The generated vector search index (not committed)
.streamlit/config.toml    Streamlit configuration settings
```

## Testing

- `python -m pytest` runs the unit tests. No LLM key is needed because tests use mock LLMs and mock email senders.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `streamlit` not recognized | Activate the virtual environment (`.venv`) |
| "No movies are indexed yet" | `python -m scripts.ingest_movies --reset` |
| "LLM setup problem" | Fill in `LLM_PROVIDER`, `LLM_MODEL`, and the matching key in `.env`, then restart |
| Email says `DRY RUN` | Set `EMAIL_DRY_RUN=false` in `.env` to send real emails |
| Email login rejected | Use a Gmail App Password, not your normal password |
| New movie not showing | Index it (button in the page, or the ingest command) |
| Email tool "Connection closed" | Run `pip install "mcp<2"` |
| Settings changes ignored | Settings are read at start-up; restart Streamlit |

## Limitations

- Subtitles rarely name the speaker, so "who said X?" is answered with "the subtitles do not name the speaker" unless the line is labelled.
- The assistant does not remember earlier answers: "send email about it" cannot refer to a previous reply. Name the movie in each request.
- Songs and chants in subtitles can distract the search.
- Subtitle files are usually copyrighted. Use them only for personal learning, and do not publish them (`.gitignore` excludes them).