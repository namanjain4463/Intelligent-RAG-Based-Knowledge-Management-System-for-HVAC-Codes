# V2 review and improvement plan

Reviewed branch: `v2-graphrag`, starting at `732d9d9` (2026-09-24).

## Assessment

This is a credible research/portfolio prototype with substantially better evidence
handling than a basic PDF chatbot. It is not yet demonstrated to be a dependable
code-compliance decision system. Its strongest features are the canonical source
corpus, preservation of lists and exceptions, deterministic evidence IDs, exclusion
of review-flagged production requirements, and resumable stateless tool sessions.

The current graph primarily provides document hierarchy and section-to-requirement
links. Describe it as hierarchical GraphRAG; the older documents' equipment/location
relationship model describes v1 and should not be presented as the v2 schema.

## What the evidence actually shows

- The saved `v2_output/react/phase8_final_react_benchmark_summary.md` reports 17/20
  grounded correct, 18/20 completed, and 18/20 citation-valid answers. Expected
  evidence and section coverage are both 20/20: retrieving the expected section
  therefore did not guarantee a correct answer. This small historical benchmark
  was manually reviewed and was not rerun against live services in this review.
- `v2_output/semantic/golden_metrics.json` reports 90% value recall across 50
  extraction cases, versus 100% predicate, reference, and section recall. Numeric
  extraction deserves special attention.
- The original fresh-clone offline run executed 53 tests and failed with one test
  failure plus one class setup error because audit artifacts were absent.
- Regenerating those artifacts from the committed PDF and repaired corpus exposed
  stale assertions: 739 legacy source headings, not 584, and 441 reconciliation
  candidates, not the historical 643. The two page-174 clauses are classified as
  normative text and linked with full token coverage. Tests now check that behavior.
- Preservation as an UnparsedBlock accounts for source text; it does not prove that
  the application retrieves or interprets it correctly. Current reconciliation
  still classifies 71 candidate lines as genuine parser omissions, with explicit
  preservation. Do not equate zero silent drops with perfect usable extraction.

## Fixes in this review

1. **Final-tool synthesis:** allow one answer-only model request after the last
   permitted retrieval. Previously the loop exited before it could use that result.
   A model requesting another tool is rejected, and resume at the limit is covered.
2. **Cypher boundaries:** reject fake limits inside quoted text, nonterminal limits,
   limit arithmetic, and UNION. Generated queries receive a 15-second transaction
   timeout. These checks are deliberately conservative; they are not a Cypher parser
   or a replacement for database privileges.
3. **Reproducible tests:** rebuild audit fixtures in temporary storage from committed
   inputs, retain zero-loss and provenance assertions, and add current-denominator
   accounting without relabeling historical 643-line metrics as current evidence.
4. **Setup and CI:** fix the clone branch, add tested direct runtime dependency pins,
   mark old architecture documents as historical, and add offline GitHub Actions
   tests scoped to pushes and pull requests targeting `v2-graphrag`.

5. **Portable source provenance:** the first clean Linux CI run exposed an absolute
   Windows source path in semantic ingestion. Resolve the PDF from the current
   checkout and verify its recorded SHA-256 before using it. Semantic tests now
   consume regenerated reconciliation artifacts, rather than silently omitting them.

## Highest-value next work

| Priority | Improvement and implementation | Acceptance evidence |
| --- | --- | --- |
| P1 | Build a held-out evaluation set of at least 100 questions covering values/units, exceptions, tables/footnotes, multi-section rules, ambiguous requests, and unanswerable questions. Save source spans and acceptable abstentions, plus per-case outputs. Keep tuning cases separate. | Report claim correctness, exception completeness, retrieval recall@k, unsupported-claim rate, abstention quality, p50/p95 latency, and actual usage/cost. Compare vector-only, current hybrid, and graph-assisted variants on the same cases. |
| P1 | Replace citation-presence confidence with claim-level validation. Generate structured claims with evidence IDs and source quotes; validate IDs/quotes deterministically, check numeric units and qualifiers, and reject or abstain on unsupported claims. Evaluate semantic support with expert labels; matching a quote alone is insufficient. | Adversarial tests where a real E1 supports a different claim must fail. Measure false acceptance and false rejection. |
| P1 | Close no-evidence policy gaps. `_likely_regulatory_question()` examines only the current question and has capability/inventory exemptions. Mixed requests and follow-ups such as "Does that apply here?" can bypass it. Use explicit answer types and conversation-aware grounding checks; require retrieved evidence for code claims regardless of prompt keywords. | Tests for mixed capability/code requests, unsupported inventory counts, and pronoun follow-ups. Ordinary greetings remain usable. |
| P1 | Use a restricted database account, reviewed query templates or a stricter query subset, bounded response bytes, input/context/output budgets, and explicit API timeouts. A small row limit can still return a huge `collect()` or trigger expensive traversal. | Tests for aggregate payloads, expensive queries, timeouts, and budget exhaustion; verify database privileges separately. |
| P2 | Improve hybrid ranking. `hybrid_search()` currently concatenates lexical, full-text, and vector candidates, truncating at top_k; lexical hits can consume every slot. Evaluate reciprocal-rank fusion followed by a small reranking step, deduplicating by source identity. Batch section/ancestor lookups to reduce repeated Aura round trips. | Better clause recall and answer correctness at equal cost, with an ablation showing whether graph expansion helps. |
| P2 | Make sources inspectable: return structured answer/evidence data to Streamlit, provide expandable exact excerpts and PDF page links, and show document edition/jurisdiction/coverage verified from source metadata. Bind graph and local corpus versions with hashes during preflight. | Clicking a citation displays the exact quoted source; stale corpus/graph combinations are detected. |
| P2 | Separate runtime evidence assembly from benchmark helpers, centralize model/index configuration, and add checkpoint retention and explicit client lifecycle management. `.env.example` model settings currently do not control the hard-coded runtime constants. | Configuration tests, predictable cleanup, and no change to checkpoint call pairing or evidence IDs. |

## Verification boundaries

Local regression tests use fake model/database clients. No paid OpenAI request,
Aura graph mutation, corpus regeneration, or live answer-quality benchmark was
performed. The dependency file pins direct packages observed in the local Python
3.11 environment; it is not a complete transitive lockfile. A clean Linux CI installation succeeded; the first test run exposed the source-path
issue described above. Consult the commit checks for the final CI result. This review did not visually exercise
the Streamlit interface because the changes concern runtime boundaries and tests.

Neo4j documents that driver read routing is not an access-control guarantee and
supports transaction timeouts through `unit_of_work`:
[Neo4j managed transactions](https://neo4j.com/docs/python-manual/current/transactions/).
