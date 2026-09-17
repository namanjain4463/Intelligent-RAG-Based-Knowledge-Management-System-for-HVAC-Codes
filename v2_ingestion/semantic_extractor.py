"""Grounded Phase 2A requirement extraction.

The default extractor is deterministic and conservative.  An optional LLM
adapter can be supplied for ambiguous clauses, but it must return validated
``Requirement`` objects and cannot replace the source evidence.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, Callable, Iterable

from .semantic_models import (
    Condition,
    EntityRef,
    ExceptionRef,
    ReferenceRef,
    Requirement,
    RequirementEvidence,
    SemanticError,
    SemanticInputSpan,
)
from .semantic_preparser import annotate, cue_span, predicate_for, split_clauses


def _id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(x or "") for x in parts)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _entity(mention: str, span_id: str, role: str) -> EntityRef | None:
    mention = re.sub(r"^\s*(?:\d+\.|[a-z]\.)\s*", "", mention).strip(" ,:")
    if len(mention) < 2 or not re.search(r"[A-Za-z]", mention):
        return None
    return EntityRef(entity_id=_id("entity", span_id, role, mention), mention=mention, entity_type="unspecified")


def _condition(clause: str, span_id: str) -> list[Condition]:
    conditions: list[Condition] = []
    for match in re.finditer(r"\b(?:where|when|if|unless|provided\s+that|except\s+when)\b.*", clause, re.I):
        text = match.group(0).strip(" .")
        if text:
            conditions.append(Condition(condition_id=_id("condition", span_id, text), text=text, condition_type="source_clause", source_start=match.start(), source_end=match.end()))
    return conditions


def _references(clause: str, span_id: str) -> list[ReferenceRef]:
    refs: list[ReferenceRef] = []
    patterns = [
        ("section", re.compile(r"\bSections?\s+[0-9]{3,4}(?:\.[0-9]+)*(?:\([0-9]+\))?(?:\s+(?:through|to|and)\s+[0-9]{3,4}(?:\.[0-9]+)*)?", re.I)),
        ("table", re.compile(r"\bTables?\s+[0-9]+(?:\.[0-9]+)*(?:\([0-9]+\))?", re.I)),
        ("chapter", re.compile(r"\bChapters?\s+[0-9]{1,2}(?:\s+(?:through|to|and)\s+[0-9]{1,2})?", re.I)),
        ("equation", re.compile(r"\b(?:Equation|Eq\.?)\s*[0-9]+\s*[-–]\s*[0-9]+", re.I)),
        ("external_standard", re.compile(r"\b(?:ASHRAE|NFPA|ANSI|ASTM|UL|NSF|IMC|IECC|ICC|CSA|ARI|AHRI|AGA|ISO|SMACNA|OSHA|ACGIH)(?:\s+[A-Z0-9][A-Z0-9./-]*)?", re.I)),
    ]
    for kind, regex in patterns:
        for match in regex.finditer(clause):
            text = match.group(0)
            refs.append(ReferenceRef(reference_id=_id("reference", span_id, kind, text), reference_type=kind, target=text, text=text, source_start=match.start(), source_end=match.end()))
    unique: OrderedDict[str, ReferenceRef] = OrderedDict((r.reference_id, r) for r in refs)
    return list(unique.values())


def _value(clause: str) -> tuple[str | None, str | None]:
    annotations = annotate(clause)
    if not annotations.values:
        return None, None
    first = annotations.values[0]
    return str(first.get("text") or "") or None, first.get("unit")


def _requirement_for(span: SemanticInputSpan, clause: str, ordinal: int, method: str = "deterministic") -> Requirement | None:
    predicate = predicate_for(clause)
    if predicate is None:
        return None
    cue_start, cue_end = cue_span(clause)
    subject = _entity(clause[:cue_start] if cue_start is not None else clause, span.input_span_id, f"subject:{ordinal}")
    tail = clause[cue_end:] if cue_end is not None else clause
    tail = re.split(r"\b(?:where|when|if|unless|provided\s+that|except\s+when)\b", tail, maxsplit=1, flags=re.I)[0]
    object_ref = _entity(tail, span.input_span_id, f"object:{ordinal}")
    value_text, unit = _value(clause)
    conditions = _condition(clause, span.input_span_id)
    exceptions: list[ExceptionRef] = []
    if span.source_type == "exception" or re.search(r"\b(?:exception|except|unless|provided\s+that)\b", clause, re.I):
        exceptions.append(ExceptionRef(exception_id=_id("exception", span.input_span_id, ordinal), text=span.source_text, exception_block_id=span.exception_block_id))
    references = _references(clause, span.input_span_id)
    evidence = RequirementEvidence(
        evidence_id=_id("evidence", span.input_span_id, ordinal), input_span_id=span.input_span_id,
        source_block_id=span.source_block_id, source_text=span.source_text, page_no=span.page_no,
        source_pdf_text=span.source_pdf_text,
        section_id=span.section_id, bbox=span.bbox, char_start=span.char_start, char_end=span.char_end,
        provenance_status=span.provenance_status, extraction_method=method,
        source_sha256=span.source_sha256, pdf_page_text_sha256=span.pdf_page_text_sha256,
    )
    review_reason = None
    needs_review = False
    if subject is None or object_ref is None:
        needs_review = True
        review_reason = "deterministic extraction left a subject or object unresolved"
    return Requirement(
        requirement_id=_id("requirement", span.document_id, span.input_span_id, ordinal, predicate, clause),
        document_id=span.document_id, section_id=span.section_id, predicate=predicate,
        subject=subject, object=object_ref, value_text=value_text, unit=unit,
        conditions=conditions, exceptions=exceptions, references=references, evidence=evidence,
        confidence=0.82 if not needs_review else 0.68, needs_review=needs_review, review_reason=review_reason,
    )


def extract_requirements(
    spans: Iterable[SemanticInputSpan],
    llm: Callable[[SemanticInputSpan, str], Requirement | list[Requirement] | None] | None = None,
) -> tuple[list[Requirement], list[SemanticError], dict[str, Any]]:
    requirements: list[Requirement] = []
    errors: list[SemanticError] = []
    seen: set[str] = set()
    input_count = 0
    table_excluded = 0
    for span in spans:
        input_count += 1
        if span.source_type in {"table_reference", "equation_reference"}:
            continue
        if not span.normative:
            continue
        clauses = split_clauses(span.source_text)
        for ordinal, clause in enumerate(clauses):
            req = _requirement_for(span, clause, ordinal)
            if req is None:
                if llm is not None and re.search(r"\b(?:shall|must|required|prohibited|permitted|minimum|maximum)\b", clause, re.I):
                    try:
                        candidate = llm(span, clause)
                    except Exception as exc:  # pragma: no cover - adapter boundary
                        errors.append(SemanticError(error_id=_id("semantic-error", span.input_span_id, ordinal, "llm"), code="LLM_FAILURE", message=str(exc), severity="quarantine", input_span_id=span.input_span_id, section_id=span.section_id, page_no=span.page_no, source_text=span.source_text))
                        candidate = None
                    candidates = candidate if isinstance(candidate, list) else ([candidate] if candidate else [])
                    for item in candidates:
                        if item.evidence.source_text != span.source_text or item.section_id != span.section_id:
                            errors.append(SemanticError(error_id=_id("semantic-error", span.input_span_id, ordinal, "llm-grounding"), code="LLM_GROUNDING_FAILURE", message="LLM output did not preserve exact section/source evidence", severity="quarantine", input_span_id=span.input_span_id, section_id=span.section_id, page_no=span.page_no, source_text=span.source_text))
                            continue
                        requirements.append(item)
                continue
            if req.requirement_id not in seen:
                seen.add(req.requirement_id)
                requirements.append(req)
    summary = {
        "input_span_count": input_count,
        "requirement_count": len(requirements),
        "needs_review_count": sum(r.needs_review for r in requirements),
        "table_derived_requirement_count": table_excluded,
        "semantic_error_count": len(errors),
        "predicate_counts": {predicate: sum(r.predicate == predicate for r in requirements) for predicate in sorted({r.predicate for r in requirements})},
        "section_count": len({r.section_id for r in requirements}),
    }
    return requirements, errors, summary


def related_records(requirements: Iterable[Requirement]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    entities: dict[str, dict[str, Any]] = {}
    conditions: dict[str, dict[str, Any]] = {}
    exceptions: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for req in requirements:
        for entity in [req.subject, req.object]:
            if entity:
                row = entity.model_dump(mode="json"); row["requirement_id"] = req.requirement_id; entities.setdefault(entity.entity_id, row)
        for condition in req.conditions:
            row = condition.model_dump(mode="json"); row["requirement_id"] = req.requirement_id; conditions.setdefault(condition.condition_id, row)
        for exception in req.exceptions:
            row = exception.model_dump(mode="json"); row["requirement_id"] = req.requirement_id; exceptions.setdefault(exception.exception_id, row)
        row = req.evidence.model_dump(mode="json"); row["requirement_id"] = req.requirement_id; evidence.setdefault(req.evidence.evidence_id, row)
    return list(entities.values()), list(conditions.values()), list(exceptions.values()), list(evidence.values())
