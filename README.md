# Text-to-SQL Agent

Ask questions about any SQLite database in plain English. Upload a `.db` file,
and the agent writes a read-only SQL query, runs it, retries with the error
message if it fails, and answers in plain language with a results table and CSV
download.

It is a [LangGraph](https://langchain-ai.github.io/langgraph/) agent behind a
FastAPI backend, with a Streamlit client.

```
Streamlit (app.py) -> FastAPI (api.py) -> generate_sql -> execute_sql -> summarize
                                              ^-------- retry on error ---'
```

## Safety

- Only a single `SELECT` is allowed (checked with sqlglot) and `LIMIT` is capped.
- Databases are opened read-only, so your file is never modified.
- Uploads are validated (extension, SQLite header, size) and stored under a random name.

## Privacy

The LLM is hosted by Groq. The schema, a few sample rows per table
(`retrieval.sample_limit`, set to `0` to disable) and up to 20 result rows are
sent to it. Do not upload data you cannot share with a third party.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync
echo "GROQ_API_KEY=your-key" > .env
```

Check that `llm.model_name` in `utils/config.yaml` is a model currently available on Groq.

## Run

```bash
uv run uvicorn api:app --reload      # backend, port 8000
uv run streamlit run app.py          # UI
```

Set `API_BASE_URL` if the backend is not at `http://127.0.0.1:8000`. Without an
upload, the UI uses the sample database `db/noshow.db`.

## Test

```bash
uv run pytest
```

Tests use a fake LLM and do not need an API key.

## Configuration

See `utils/config.yaml` for the model, schema retrieval, guardrails, upload limits and
optional currency post-processing of result tables (off by default).
