# Evaluation protocol

`cases.json` freezes 140 source-anchored candidate cases: 20 development and 120
held-out, including 100 single-section questions, 10 multi-section comparisons,
and 10 ambiguous/out-of-scope/mixed requests. Development and held-out expected
sections are disjoint. Questions were generated from canonical section titles and
source features; they are **not an independently expert-labeled benchmark**.

Do not tune prompts/rank weights on these held-out cases and then call the same
scores unbiased validation. Use a newly collected expert test set for publication.
Exact section numbers make many questions easier than real user requests.

Run from the repository root (requires the populated Aura instance):

```powershell
python -m scripts.evaluate --env-file "path/to/.env" --cap-usd 1 --answer-limit 30
python -m scripts.evaluate_answers --env-file "path/to/.env" --cap-usd 1
```

Both commands share `v2_output/eval_live/spend.json`. The second reuses the first
run's frozen candidate rankings to compare synthesis against identical retrieval
results. It performs up to 360 answer checks using four threads. **Do not run two
processes against the same ledger simultaneously.** Thread reservations are locked;
this is not a distributed billing system. A process crash leaves reservations
charged conservatively. Never remove the ledger to evade an authorized cap.

The first run records section recall@5. All three variants expand the same local
canonical evidence; this is a ranking ablation, not an independent proof that graph
traversal improves answer correctness. The second run records answer dispositions,
automated support verdicts, input/output usage, and p50/p95 latency. Charges are
conservative estimates using verified model pricing; the provider invoice remains
the billing authority. Dataset metadata and quoted source text are public corpus
material; credentials are never written into outputs.

## Human review

Review each grounded answer against every cited source passage and the PDF. Mark
claim correctness, scope, units, missing conditions/exceptions, and whether an
abstention was justified. Keep at least two reviewer labels and adjudicate
agreements/disagreements. Until labels exist, expert correctness, unsupported-claim
rate, and exception completeness are **null**, not 0 or 100%. Automated model
verdicts are a separate signal and cannot fill those fields.

Live per-case outputs and spend reservations are ignored. Compact aggregate
summaries in this directory describe the actual run and its limitations.

## Measured run: 2026-09-24

| Mode | Mean section recall@5 | Source-checked answers | Abstain/clarify | Negative abstention | Answer p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 79.1% | 46/120 | 74/120 | 90% | 4.66 / 11.41 s |
| Legacy concatenation | 28.6% | 14/120 | 106/120 | 100% | 2.59 / 6.08 s |
| Rank fusion | 79.1% | 40/120 | 80/120 | 100% | 3.94 / 9.11 s |

Recall averages the 110 source-backed cases; negative abstention uses 10 cases.
All 360 answer checks completed without service errors. The shared ledger total
is **$0.635521**, including setup/smoke calls and initial answer probes, under the
authorized $1 cap. This is conservative accounting, not an invoice. Answer latency
excludes retrieval and uses four concurrent workers. These are fixed-retrieval
synthesis checks, not 360 full ReAct trajectories.

Fusion repairs lexical crowd-out but does not outperform vector retrieval here.
The stricter checker still abstains often. Source-checked means the automated
checks passed; expert correctness remains unmeasured. Before deployment, prioritize
expert review of false acceptance and false rejection rather than loosening checks
to increase this answer count. A section-number metadata validation bug was fixed
before the full synthesis comparison; consequently treat this set as an engineering
regression set, not an untouched publication-quality test set.
