# Multi-Agent Research Assistant

A research prototype that retrieves arXiv abstracts, then uses a LangGraph workflow to coordinate three AutoGen agents: a researcher, a critic, and a report writer. The generated Markdown includes source IDs and a list of retrieved records.

## Run

Use Python 3.10+, create a virtual environment, and install `pip install -r requirements.txt`. Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and run:

```sh
python -m research_assistant "retrieval augmented generation evaluation" --output reports/rag.md
```

The workflow queries arXiv and sends abstracts and the topic to OpenAI. Calls can incur costs. It searches abstracts only, not full papers or the whole web; arXiv includes work that may not be peer reviewed. Citations and claims require human verification. Source text is treated as untrusted, but prompt injection and hallucinations cannot be eliminated. No report is sent or published automatically.
