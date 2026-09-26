# HVAC Codes GraphRAG Assistant

A clone-and-run GraphRAG + ReAct proof of concept for questions about `HVAC-Codes.pdf`. It is a research/portfolio demo, not a production service. The current v2 runtime uses a ReAct agent to choose among three retrieval tools plus the GeneralInfo conversation tool, reads the validated graph in Neo4j Aura, and assembles answer evidence from the canonical structural corpus.

The Streamlit interface is `bot.py`; the runtime is `v2_ingestion/react_runtime.py`. The older ingestion and agent files remain in the repository for historical context, but `bot.py` uses the v2 runtime through `agent.py`.

## Current interface and validation

The HVAC Assistant provides chat, section search, references, and a PDF reader.
An optional answer inspector displays actual tool actions and section relationships.
Its offline reference-removal experiment shows which claims lose citation coverage;
it does not claim that removing a citation changes factual truth. No additional API
calls are made by the inspector. Technical details stay outside the main chat view.

See [Operations](OPERATIONS.md) for existing-environment launch, resource limits,
checkpoints, and optional hosting considerations. See [Evaluation](evals/README.md) for the
120-case live comparison and its limitations. Automated support checks deliberately
abstain when uncertain; they are not expert compliance certification.

## How it works

```text
Question
  -> GPT-5.6 Luna ReAct agent
  -> CypherSearch, VectorSearch, and/or HybridSearch
  -> read-only Neo4j Aura results
  -> canonical section evidence (including applicable lists and exceptions)
  -> structured claims with exact source quotes
  -> deterministic quote/number checks + separate semantic support review
  -> supported answer or abstention
  -> deterministic Section/page citations
```

- `CypherSearch` accepts bounded, read-only Cypher against the current graph.
- `VectorSearch` queries the existing `hvac_passage_embeddings` index using `text-embedding-3-large`.
- `HybridSearch` fuses lexical, full-text, and vector rankings by reciprocal rank, then expands canonical evidence.

The graph uses `Document`, `Chapter`, `Section`, and `Requirement` nodes, with `CONTAINS` and `STATES` relationships. The agent may call more than one tool; there is no fixed section-number router in the final runtime. It cites supplied evidence IDs such as `[E1]`, which the application renders as a Section/page citation using the local evidence map. The special `section:unassigned` ownership is excluded from authoritative retrieval.

## Run locally (Windows PowerShell)

You need Python, an OpenAI API key with access to `gpt-5.6-luna` and `text-embedding-3-large`, and credentials for the **already populated v2 Neo4j Aura database**. The app does not create or populate an Aura database on startup. The Aura graph must already contain the production Requirements and the online `hvac_passage_embeddings` vector index. You do not need a local Neo4j server.

From a fresh clone of this branch:

```powershell
git clone --branch v2-graphrag https://github.com/namanjain4463/Intelligent-RAG-Based-Knowledge-Management-System-for-HVAC-Codes.git
cd Intelligent-RAG-Based-Knowledge-Management-System-for-HVAC-Codes
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

The runtime dependency file pins the tested direct dependencies for chat and local tests. Full PDF re-ingestion additionally requires the Docling stack in `v2_requirements.txt`; use a separate environment for that workflow.

If you already have the repository's virtual environment, use that environment instead of creating or replacing it. Run the remaining commands from the repository root.

Create a private `.env` **only if one does not already exist**:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set at least these values in `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
NEO4J_URI=neo4j+s://your-aura-host.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-aura-password
NEO4J_DATABASE=your-aura-database-name
```

Use the actual database name shown in Aura (for the validated project instance, `c3834e4b`). Keep `.env` local; it is ignored by Git. Do not put credentials in source files or commit them.

Launch the app:

```powershell
.\venv\Scripts\python.exe -m streamlit run bot.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Submitting a question makes OpenAI API calls and read-only Aura queries, which can incur charges. To stop the app, press `Ctrl+C` in the terminal.

For a single terminal question instead of the UI:

```powershell
.\venv\Scripts\python.exe agent.py
```

The app expects `v2_output/document.json` and `v2_output/semantic/requirements.parquet` in the repository; they are committed on this branch. It uses the embeddings already stored in Aura, so no local corpus embedding cache is needed to launch.

## Repository layout

| Path | Purpose |
| --- | --- |
| `bot.py` | Streamlit chat interface |
| `agent.py` | Thin CLI/UI entry point for the v2 runtime |
| `v2_ingestion/react_runtime.py` | ReAct session, retrieval tools, evidence ledger, citation binding |
| `v2_ingestion/` | Validated structural and semantic pipeline, audit, and benchmark code |
| `v2_output/document.json` | Canonical structural corpus used for complete section evidence |
| `v2_output/semantic/requirements.parquet` | Semantic requirement records and review metadata |
| `tests_v2/` | v2 regression tests |
| `HVAC-Codes.pdf` | Source document |
| `V2_SETUP.md` | Additional v2 setup and data notes |

To run the local regression tests without querying Aura or OpenAI:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests_v2
```

## Scope and limitations

The production retrieval corpus excludes review-flagged, quarantined, and `section:unassigned` Requirements. Tables and other source material are not a substitute for checking the original PDF. Answers are limited to evidence retrieved from the validated corpus and should be checked against the cited source pages before using them for design or code-compliance decisions.

This branch does not rebuild the graph or vector index when the UI launches. The legacy `requirements.txt` is retained for historical workflows; use `requirements-runtime.txt` for v2 chat and offline tests. Local credentials, virtual environments, caches, logs, checkpoints, and generated benchmark outputs should stay out of Git; see `.gitignore`.

## Validation and next improvements

The regression suite regenerates audit CSV/JSON fixtures from the committed PDF
and corpus in a temporary directory. It does not require private audit files,
OpenAI calls, or Aura. PDF audit regeneration may take several minutes.

The saved 20-question benchmark reports 17 grounded correct answers, 18 completed
answers, and 18 citation-valid answers. These are historical measurements, not a
new evaluation of the current revision. Citation binding verifies evidence IDs,
not whether every claim follows from its source.

See [the v2 review and improvement plan](V2_REVIEW.md) for findings, fixes, and
acceptance criteria. `TECHNICAL_ARCHITECTURE.md` and `TECHNICAL_ROADMAP.md` describe
the legacy implementation and should not be used as v2 performance evidence.

For deployment, provision database credentials restricted to reads. Driver read
routing and application query validation are not database authorization. Generated
Cypher has a literal terminal row limit, rejects UNION, and uses a 15-second
transaction timeout; a row limit does not bound the size of an individual value.

## Engineering contribution and scope

This project combines a ReAct tool loop (observe a result, then choose another
retrieval action or answer) with a hierarchical document graph in Aura and
canonical section expansion. ReAct refers to the agent pattern, not React.js;
the UI uses Streamlit. Some questions need only one retrieval action.

The distinctive demonstrator is an inspectable answer pipeline: recorded tool
actions, real section-parent links, claim-to-reference dependencies, and an
offline reference-removal experiment. Together with resumable tool-call pairing
and measured retrieval ablations, this is an engineering contribution, not a claim
of a new research algorithm. Parent links displayed by the inspector come from
the local document structure; they do not pretend a database traversal happened.

For a stronger research claim, collect independently reviewed questions requiring
parent conditions or exceptions and compare identical retrieval with and without
section expansion. The existing ranking ablation does not isolate this graph effect.
No production hosting is required. Clone, configure your own credentials and
populated Aura database, and run locally. Cloning does not provision the database.

## Chat and GeneralInfo

The chat uses a fixed bottom composer, a single pending-response slot, and public
activity events from actual runtime stages. Completed activity can be expanded;
private model reasoning and opaque checkpoint content are never displayed.
The implementation follows the [Responses function-calling contract](https://developers.openai.com/api/docs/guides/function-calling).

`GeneralInfo` handles greetings, identity/help, and an authored set of basic HVAC
concept explanations (HVAC, heat pumps, ventilation, refrigerant, thermostats,
filters, ducts, and air conditioners). It is deliberately bounded, not a general
world-knowledge chatbot. Whole-request matching prevents a concept keyword from
authorizing an unrelated or regulatory answer. Code questions still use retrieval.
The tool is exposed to the agent; recognized conversation shortcuts also invoke
it directly without model/database calls, recorded as deterministic shortcuts.

A three-turn browser check covered a heat-pump explanation, a cited Section 303.7
answer, and a Section 303.6 follow-up that abstained. Loading and completed layouts
were inspected. The shared live-test ledger reached approximately $0.64434 against
the existing $1 cap; no new evaluation budget was assumed.
