# Multi-Agent Research Assistant

A runnable local research workspace with a web interface, progress tracking, source library, cancellation, and Markdown report downloads. LangGraph runs the collect → summarize → review → write workflow. In AI mode, three AutoGen agents perform the research, critique and writing stages.

## Start the application

Use **Python 3.11 or newer (tested with Python 3.12)**. The macOS system Python may be older; check `python3 --version` first.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn research_assistant.app:app --host 127.0.0.1 --port 8002
```

Open **http://127.0.0.1:8002/**. The offline sample works immediately without credentials or network access. Alternatively, run `./run.sh`; set `PYTHON_BIN=/path/to/python3.12` if needed, and `PORT=8002` to override the port. The script creates a local virtual environment and installs requirements before launching.

## Choose a research mode

| Mode | Sources | Processing | Credentials |
| --- | --- | --- | --- |
| Offline sample | Bundled, clearly fictional examples | Deterministic source extracts | None |
| Live arXiv extracts | Current arXiv API search | Collects source abstract extracts; no AI synthesis | None |
| AI research team | Current arXiv API search | AutoGen researcher, critic and writer | `OPENAI_API_KEY` |

The offline sample demonstrates the workflow and interface; it does not research the entered topic. Live extractive mode researches the topic without a paid model. AI mode sends the topic, retrieved abstracts and intermediate text to OpenAI and incurs provider usage.

For AI mode, copy `.env.example` to `.env`, fill in `OPENAI_API_KEY`, and restart the server. `OPENAI_MODEL` defaults to `gpt-4o-mini`. Never commit `.env`.

## Command-line reports

```sh
python -m research_assistant "RAG evaluation" --mode demo --output reports/sample.md
python -m research_assistant "language model uncertainty" --mode extractive --sources 3 --output reports/uncertainty.md
python -m research_assistant "retrieval augmented generation" --mode ai --output reports/rag.md
```

`--mode demo` is the default so a fresh checkout runs without provider setup. Each report includes its processing mode, inline source IDs and canonical source records.

## API

- `GET /health` — process health and active job count.
- `GET /api/config` — whether AI credentials are configured; never returns secrets.
- `POST /api/jobs` — JSON `{ "topic": "RAG evaluation", "mode": "demo", "limit": 3 }`; returns a job ID.
- `GET /api/jobs/{id}` — stage, status, source records, report or error.
- `DELETE /api/jobs/{id}` — cancel an active job.
- `GET /api/jobs/{id}/report` — download completed Markdown.

Interactive API documentation is available at `/docs`.

## Verification

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests exercise the actual LangGraph workflow, citation rejection, source parsing, offline reports, API validation, downloads, cancellation and provider failures without calling paid services. Live arXiv can be checked using the extractive CLI command above. AI reports require a valid key and are not covered by a paid end-to-end call in this workspace.

## Operational limits

This is a single-process local application. Bind to loopback and run one Uvicorn worker. The server retains at most 64 jobs in memory, allows three active jobs and cancels jobs after five minutes. Download reports you want to keep: history expires on restart or when the buffer fills. Public deployment needs authentication and persistent job storage.

The arXiv client uses a single connection, spaces queries by at least three seconds, retries transient failures, and caches results for 15 minutes. Search uses up to 25 topic words and returns at most eight abstracts. Abstracts are normalized for whitespace and bounded to 6,000 characters per source. It does not read full papers, search the entire web or verify peer review status. Citation validation checks that IDs exist; it cannot establish that a claim is supported. Human review of source material remains necessary. See the [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html), [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview), and [AutoGen documentation](https://microsoft.github.io/autogen/stable/index.html).
