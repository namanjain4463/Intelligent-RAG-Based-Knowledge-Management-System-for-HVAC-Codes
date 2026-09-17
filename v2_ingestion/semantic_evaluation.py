"""Reproducible, field-separated golden evaluation for Phase 2A."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .semantic_models import Requirement, SemanticInputSpan
from .semantic_preparser import annotate, predicate_for


TARGET_PREDICATES = ("PROHIBITED_IN", "MINIMUM", "MAXIMUM", "REQUIRES_CLEARANCE", "REQUIRES_DEVICE", "MUST_COMPLY_WITH", "PERMITTED_IN", "REQUIRES")


def _case(span: SemanticInputSpan, predicate: str, label: str, expected_ref_type: str | None = None) -> dict[str, Any]:
    annotations = annotate(span.source_text)
    values = [str(x.get("text")) for x in annotations.values[:3]]
    refs = []
    if expected_ref_type == "section": refs = [x["text"] for x in annotations.section_refs]
    if expected_ref_type == "table": refs = [x["text"] for x in annotations.table_refs]
    if expected_ref_type == "chapter": refs = [x["text"] for x in annotations.chapter_refs]
    if expected_ref_type == "equation": refs = [x["text"] for x in annotations.equation_refs]
    if expected_ref_type == "external_standard": refs = [x["text"] for x in annotations.external_standard_refs]
    return {"gold_id": f"gold:{len(label)}:{span.input_span_id}:{label}", "label": label, "section_id": span.section_id, "page_no": span.page_no, "input_span_id": span.input_span_id, "source_text": span.source_text, "expected_predicate": predicate, "expected_values": values, "expected_references": refs, "expected_reference_type": expected_ref_type}


def build_golden_cases(spans: list[SemanticInputSpan], minimum: int = 50) -> list[dict[str, Any]]:
    """Select a diverse, source-backed benchmark from the audited input.

    The cases are persisted with their exact source text and labels so a human
    can review or replace individual labels without changing extraction code.
    """
    selected: list[dict[str, Any]] = []; used: set[str] = set()
    priorities = [
        ("303.3", "PROHIBITED_IN", "section_303_3_prohibition", None),
        ("303.3", "REQUIRES", "section_303_3_exception", None),
        ("403.3.1.1", "MINIMUM", "chapter_4_ventilation_minimum", None),
        ("403.3.1.1", "MAXIMUM", "chapter_4_ventilation_maximum", None),
    ]
    for number, predicate, label, ref_type in priorities:
        for span in spans:
            if span.section_id.endswith(number) and span.input_span_id not in used and predicate_for(span.source_text) == predicate:
                selected.append(_case(span, predicate, label, ref_type)); used.add(span.input_span_id); break
    cue_labels = [
        ("PROHIBITED_IN", "prohibition"), ("PERMITTED_IN", "permission"), ("REQUIRES", "mandatory"),
        ("MINIMUM", "minimum"), ("MAXIMUM", "maximum"), ("REQUIRES_CLEARANCE", "clearance"),
        ("REQUIRES_DEVICE", "device"), ("MUST_COMPLY_WITH", "compliance"),
    ]
    for predicate, label in cue_labels:
        for span in spans:
            if span.input_span_id in used or predicate_for(span.source_text) != predicate:
                continue
            selected.append(_case(span, predicate, label)); used.add(span.input_span_id); break
    ref_specs = [("section", "section_reference"), ("table", "table_reference"), ("chapter", "chapter_reference"), ("equation", "equation_reference"), ("external_standard", "external_standard_reference")]
    for ref_type, label in ref_specs:
        for span in spans:
            ann = annotate(span.source_text)
            matches = getattr(ann, f"{ref_type}_refs", [])
            if span.input_span_id not in used and matches and predicate_for(span.source_text):
                selected.append(_case(span, predicate_for(span.source_text) or "REQUIRES", label, ref_type)); used.add(span.input_span_id); break
    # Fill across sections/pages to avoid a chapter-4-only benchmark.
    for span in sorted(spans, key=lambda s: (s.page_no, s.order, s.input_span_id)):
        if len(selected) >= minimum: break
        predicate = predicate_for(span.source_text)
        if predicate and span.input_span_id not in used:
            selected.append(_case(span, predicate, f"diverse_{len(selected)+1}")); used.add(span.input_span_id)
    return selected


def evaluate(requirements: list[Requirement], cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_span: dict[str, list[Requirement]] = {}
    for req in requirements:
        by_span.setdefault(req.evidence.input_span_id, []).append(req)
    predicate_hits = value_hits = reference_hits = section_hits = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        found = by_span.get(case["input_span_id"], [])
        pred_hit = any(req.predicate == case["expected_predicate"] for req in found)
        value_hit = any(not case["expected_values"] or any(value.casefold() in (req.value_text or "").casefold() for value in case["expected_values"]) for req in found)
        ref_hit = any(not case["expected_references"] or any(expected.casefold() in json.dumps([r.model_dump(mode="json") for r in req.references], ensure_ascii=False).casefold() for expected in case["expected_references"]) for req in found)
        section_hit = bool(found) and all(req.section_id == case["section_id"] for req in found)
        predicate_hits += pred_hit; value_hits += value_hit; reference_hits += ref_hit; section_hits += section_hit
        rows.append({"gold_id": case["gold_id"], "predicate_hit": pred_hit, "value_hit": value_hit, "reference_hit": ref_hit, "section_hit": section_hit, "production_count": len(found)})
    total = len(cases) or 1
    return {"gold_case_count": len(cases), "predicate_recall": predicate_hits / total, "value_recall": value_hits / total, "reference_recall": reference_hits / total, "section_recall": section_hits / total, "case_results": rows}


def write_golden_artifacts(output_dir: Path, spans: list[SemanticInputSpan], requirements: list[Requirement]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_golden_cases(spans, minimum=50)
    metrics = evaluate(requirements, cases)
    cases_path = output_dir / "golden_cases.json"
    metrics_path = output_dir / "golden_metrics.json"
    cases_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"golden_cases": cases_path, "golden_metrics": metrics_path}
