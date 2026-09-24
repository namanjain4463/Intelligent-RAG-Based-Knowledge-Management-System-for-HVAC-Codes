# V2 GraphRAG setup

This branch runs the current ReAct agent through `agent.py` and the Streamlit UI in `bot.py`. The agent chooses among `CypherSearch`, `VectorSearch`, and `HybridSearch`; answers are grounded in the canonical source blocks and citations are bound to evidence IDs before section/page rendering.

## Local prerequisites

- Python virtual environment with `requirements-runtime.txt` installed. Use the existing repository virtual environment when available.
- OpenAI API access for `gpt-5.6-luna` and `text-embedding-3-large`.
- A populated Neo4j Aura database with the validated v2 graph and the `hvac_passage_embeddings` vector index. This branch does not create a new Aura instance when the app starts.
- A local `.env` copied from `.env.example`, with your own API key, Aura URI, username, password, and database name. Never commit `.env`.

Run the UI with `python -m streamlit run bot.py` from the repository root. Each submitted question starts an isolated ReAct session and can incur API charges.

## Committed data and local-only outputs

The committed `v2_output/document.json` is the repaired canonical structural corpus used to assemble complete section evidence. `v2_output/semantic/requirements.parquet` supplies semantic records, review flags, and exception fallbacks; review flagged records are not production graph requirements. The structural Parquet files, golden cases, and compact audit summaries are included for inspection.

Local embedding caches, query vectors, API response traces, stateless checkpoints, and expanded benchmark artifacts are intentionally ignored. The production embeddings and vector index live in Aura; the runtime generates query embeddings as needed. The original PDF remains tracked because it was already part of this repository.

The current graph has `Document`, `Chapter`, `Section`, and `Requirement` nodes. Real numbered Sections are used as authoritative evidence; `section:unassigned` is excluded from production retrieval. The legacy LangChain files describe an earlier graph and are not used by `bot.py`. For copyable Windows PowerShell installation and launch commands, see the current [README](README.md).
