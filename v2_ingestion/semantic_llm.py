"""Optional schema-constrained LLM boundary for ambiguous clauses.

No provider or network call is made here.  A local caller may use
``build_prompt`` with its own model and pass the JSON response through
``validate_response`` before handing the validated objects to the extractor.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from .semantic_models import Requirement, SemanticInputSpan


def build_prompt(span: SemanticInputSpan, clause: str) -> str:
    schema = Requirement.model_json_schema()
    return (
        "Extract only atomic HVAC-code requirements grounded in the supplied clause. "
        "Do not infer entities, values, conditions, exceptions, or references. "
        "Return a JSON array matching this Pydantic schema; use only the permitted "
        "predicates and preserve the exact evidence source text. If the clause is "
        "not sufficiently grounded, return an empty array.\n\n"
        f"Section: {span.section_id}\nPage: {span.page_no}\n"
        f"Exact source span: {span.source_text}\nClause: {clause}\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


def validate_response(payload: Any, span: SemanticInputSpan) -> list[Requirement]:
    """Validate provider JSON and enforce source grounding at the adapter boundary."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if payload is None:
        return []
    candidates = payload if isinstance(payload, list) else [payload]
    requirements = TypeAdapter(list[Requirement]).validate_python(candidates)
    validated: list[Requirement] = []
    for requirement in requirements:
        if requirement.section_id != span.section_id or requirement.evidence.source_text != span.source_text:
            raise ValueError("LLM output is not grounded to the exact semantic input span")
        if requirement.evidence.page_no != span.page_no or requirement.evidence.source_sha256 != span.source_sha256:
            raise ValueError("LLM output changed page or source-file provenance")
        validated.append(requirement)
    return validated
