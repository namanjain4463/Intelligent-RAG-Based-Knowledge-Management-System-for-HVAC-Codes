# Running the proof of concept

The Streamlit app now returns structured claims, checks exact quotes and numbers,
and runs a separate support/qualification review. Failure at any step yields an
abstention. A successful automated check is not an expert compliance decision.

## Configuration

Install `requirements-runtime.txt` with Python 3.11. Set credentials in a local
`.env`, or set `HVAC_ENV_FILE` to an existing private environment file before
starting Python. `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `VECTOR_INDEX_NAME`, and
`VECTOR_DIMENSION` are read centrally before clients are created.

For an existing legacy environment on Windows:

```powershell
.\scripts\start.ps1 -EnvFile "C:\path\to\final_ai\.env"
```

The launcher selects `hvac_passage_embeddings` / 3072 for v2 without changing the
original `.env`. Launch from the environment where runtime dependencies are
installed. `python -m streamlit run bot.py` remains supported with a correct `.env`.

## Database access (optional hosting considerations)

Production hosting is outside this project’s scope. If you later share a hosted
instance, use an Aura identity restricted to reads. The supplied
account inspected during this work has an administrative role; it was not modified.
Use the Aura console's supported user/role controls for the instance/tier to
provision an application reader. Do not grant schema/admin/write permissions.
Application validation and driver read routing are not database authorization.

Preflight checks connectivity, configured vector-index state/dimensions, and the
Document source hash. Local PDF bytes must match the canonical recorded hash.
This checks source identity, not complete graph/corpus transformation equivalence;
rebuilds from the same PDF should also retain a reviewed ingestion manifest.
Generated Cypher rejects writes (including INSERT), procedure calls, UNION,
variable-length traversals, and collection/range/reduction functions. All database
reads have timeouts and bounded result bytes/rows. Resource limits reduce abuse;
restrict database privileges and capacity separately.

## Checkpoints and privacy

Checkpoints preserve encrypted reasoning and exact tool-call/output pairs, but
questions, quotes, and answers are ordinary local JSON. Keep them private. Final
validated results are persisted so completed sessions can be returned without
reissuing paid model calls. Source hashes prevent resuming against a different PDF.

```powershell
python -m scripts.checkpoint_cleanup --days 7          # dry run
python -m scripts.checkpoint_cleanup --days 7 --delete # explicit local cleanup
```

No automatic background deletion or scheduled task is installed. Service clients
are closed after requests. The browser retains its own conversation until cleared;
New conversation clears that browser session, not existing disk checkpoints.

## API behavior

Structured Responses output uses `text.format` with strict JSON Schema, and
`store=false`. Checkpoint replay retains opaque reasoning items. Request/context,
output, question-length, and retrieval-payload limits are enforced. Network
requests use explicit timeouts and no automatic paid retries. Historical replay
tests use explicit legacy text mode; the UI, CLI and evaluators use strict mode.

References: [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Neo4j managed transactions](https://neo4j.com/docs/python-manual/current/transactions/),
[GPT-5.6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[embedding pricing](https://developers.openai.com/api/docs/models/text-embedding-3-large).
