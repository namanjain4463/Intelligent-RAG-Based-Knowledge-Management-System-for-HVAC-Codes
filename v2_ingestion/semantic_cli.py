"""Run Phase 2A without touching Neo4j."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .semantic_evaluation import write_golden_artifacts
from .semantic_extractor import extract_requirements
from .semantic_input import build_semantic_input
from .semantic_writer import write_semantic_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build grounded Phase 2A semantic outputs")
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    spans, input_errors, input_summary = build_semantic_input(
        args.document, args.audit_dir / "provenance_reconciliation.csv", args.audit_dir / "reconciliation_unparsed_blocks.json"
    )
    requirements, extraction_errors, extraction_summary = extract_requirements(spans)
    errors = [*input_errors, *extraction_errors]
    summary = {"phase": "2A", "neo4j_write_performed": False, "input": input_summary, "extraction": extraction_summary, "semantic_error_count": len(errors)}
    paths = write_semantic_outputs(args.output_dir, spans, requirements, errors, summary)
    paths.update(write_golden_artifacts(args.output_dir, spans, requirements))
    print(json.dumps({"counts": {"input_spans": len(spans), "requirements": len(requirements), "semantic_errors": len(errors), "golden_cases": json.loads((args.output_dir / 'golden_metrics.json').read_text(encoding='utf-8')).get('gold_case_count', 0)}, "artifacts": {k: str(v.resolve()) for k, v in paths.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
