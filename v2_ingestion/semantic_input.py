"""Build the canonical, provenance-checked semantic input view.

This stage consumes Phase 1 artifacts.  It does not mutate ``document.json``
and it intentionally omits table-derived content from production semantic
input while retaining table references in section prose.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import fitz

from .semantic_models import SemanticError, SemanticInputSpan


EXCLUDED_CLASSES = {"Commentary", "UserNote", "Insights", "NavigationArtifact", "SourceArtifact"}
EXCLUDED_LABELS = {"page_header", "page_footer", "header", "footer", "navigation"}


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\ufffd", " ").replace("\u00a0", " ")
    value = re.sub(r"[^\w%./+-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.casefold().split())


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _is_table_residue(record: dict[str, Any]) -> bool:
    table_id = record.get("table_id")
    parent = str(record.get("parent_target_id") or "")
    reason = str(record.get("reason") or "").casefold()
    return bool(table_id or parent.startswith("table:") or "table source" in reason or "tablefootnote" in reason)


def _section_maps(document: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str | None], dict[str, str | None]]:
    sections = {str(s.get("id")): s for s in document.get("sections") or []}
    block_to_section: dict[str, str | None] = {}
    block_to_chapter: dict[str, str | None] = {}
    for section in sections.values():
        sid = str(section.get("id"))
        cid = section.get("chapter_id")
        for block in section.get("blocks") or []:
            block_to_section[str(block.get("id"))] = sid
            block_to_chapter[str(block.get("id"))] = cid
            for item in _walk_items(block):
                block_to_section[str(item.get("id"))] = sid
                block_to_chapter[str(item.get("id"))] = cid
    return sections, block_to_section, block_to_chapter


def _walk_items(block: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in block.get("items") or []:
        yield item
        if item.get("block_type") == "list":
            yield from _walk_items(item)


def _load_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            pages.append({"page_no": page_no, "text": page.get_text("text") or ""})
    return pages


def _repair_page(text: str, pages: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any] | None, str | None]:
    """Recover page identity only when the source text is sufficiently unique."""
    key = _norm(text)
    if not key:
        return None, None, "empty source text"
    tokens = key.split()
    probes = [" ".join(tokens[: min(18, len(tokens))]), " ".join(tokens[-min(18, len(tokens)):])]
    candidates: list[dict[str, Any]] = []
    for page in pages:
        page_key = _norm(page["text"])
        if all(probe and probe in page_key for probe in probes if probe):
            candidates.append(page)
    if len(candidates) != 1:
        return None, None, f"page match ambiguous ({len(candidates)} candidates)"
    return candidates[0]["page_no"], None, "unique normalized PDF-text page match"


def _provenance(block: dict[str, Any], pages: list[dict[str, Any]], failures: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str, str | None]:
    raw = dict(block.get("provenance") or {})
    if raw.get("page_no") and block.get("text"):
        return raw, "canonical", None
    failure = failures.get(str(block.get("id")))
    if not block.get("text") and failure:
        block_text = failure.get("raw_text") or ""
    else:
        block_text = block.get("text") or (failure or {}).get("raw_text") or ""
    page_no, bbox, evidence = _repair_page(block_text, pages)
    if page_no is None:
        return None, "", evidence
    raw["page_no"] = page_no
    if bbox:
        raw["bbox"] = bbox
    raw["pdf_text"] = raw.get("pdf_text") or block_text
    return raw, "reconciled", evidence


def _failure_lookup(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map Docling refs and ids to missing-provenance diagnostics."""
    result: dict[str, dict[str, Any]] = {}
    for failure in document.get("failures") or []:
        if failure.get("code") != "MISSING_PROVENANCE":
            continue
        ref = str(failure.get("item_ref") or "")
        if ref:
            result[ref] = failure
    return result


def _record_from_block(
    block: dict[str, Any],
    *,
    section: dict[str, Any],
    chapter_id: str | None,
    source_type: str,
    pages: list[dict[str, Any]],
    failures: dict[str, dict[str, Any]],
    source_sha256: str,
    exception_block_id: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    text = str(block.get("text") or "").strip()
    if not text:
        return None, "empty source text"
    provenance, status, evidence = _provenance(block, pages, failures)
    if provenance is None or not provenance.get("page_no"):
        return None, evidence or "missing page provenance"
    if not section.get("id"):
        return None, "missing section id"
    record = {
        "input_span_id": _stable_id("semantic-span", section.get("id"), block.get("id"), source_type, text),
        "document_id": "",
        "chapter_id": chapter_id,
        "section_id": str(section["id"]),
        "source_block_id": block.get("id"),
        "source_type": source_type,
        "source_text": text,
        "source_pdf_text": provenance.get("pdf_text") or text,
        "page_no": int(provenance["page_no"]),
        "bbox": provenance.get("bbox"),
        "char_start": provenance.get("char_start"),
        "char_end": provenance.get("char_end"),
        "provenance_status": str(block.get("_provenance_status") or status),
        "provenance_evidence": evidence or "canonical structural provenance",
        "order": int(block.get("order") or 0),
        "normative": True,
        "exception_block_id": exception_block_id,
        "source_sha256": provenance.get("source_sha256") or source_sha256,
        "pdf_page_text_sha256": provenance.get("pdf_page_text_sha256"),
    }
    return record, None


def resolve_source_pdf(document_path: Path, document: dict[str, Any]) -> Path:
    """Prefer the relocated corpus PDF, verifying its recorded content hash."""
    recorded = str(document.get("source_file") or "HVAC-Codes.pdf")
    filename = PureWindowsPath(recorded).name
    candidates = [document_path.parent.parent / filename, Path(recorded)]
    expected = str(document.get("source_sha256") or "").casefold()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if expected and hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            continue
        return candidate
    raise FileNotFoundError(
        "No source PDF matching the corpus provenance was found. "
        f"Place {filename} beside the v2_output directory."
    )


def build_semantic_input(
    document_path: Path,
    reconciliation_path: Path,
    reconciliation_unparsed_path: Path,
) -> tuple[list[SemanticInputSpan], list[SemanticError], dict[str, Any]]:
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document_id = str(document.get("id") or document_path.stem)
    source_sha256 = str(document.get("source_sha256") or "")
    pages = _load_pages(resolve_source_pdf(document_path, document))
    sections, block_to_section, block_to_chapter = _section_maps(document)
    failures = _failure_lookup(document)
    spans: list[dict[str, Any]] = []
    errors: list[SemanticError] = []
    seen: set[tuple[str, int, str]] = set()

    def add(block: dict[str, Any], section: dict[str, Any], chapter_id: str | None, source_type: str, exception_id: str | None = None) -> None:
        text = str(block.get("text") or "").strip()
        if not text:
            return
        if block.get("content_class") in EXCLUDED_CLASSES or str(block.get("source_label") or "").casefold() in EXCLUDED_LABELS:
            return
        record, reason = _record_from_block(
            block, section=section, chapter_id=chapter_id, source_type=source_type,
            pages=pages, failures=failures, source_sha256=source_sha256, exception_block_id=exception_id,
        )
        if record is None:
            error_id = _stable_id("semantic-error", block.get("id"), reason)
            errors.append(SemanticError(
                error_id=error_id, code="MISSING_RELIABLE_PROVENANCE", message=reason or "missing reliable provenance",
                severity="quarantine", input_span_id=str(block.get("id") or "") or None,
                section_id=str(section.get("id") or "") or None, page_no=None, source_text=text,
                details={"source_block_id": block.get("id"), "source_type": source_type},
            ))
            return
        record["document_id"] = document_id
        key = (record["section_id"], record["page_no"], _norm(record["source_text"]))
        if key not in seen:
            seen.add(key)
            spans.append(record)

    for section in document.get("sections") or []:
        sid = str(section.get("id") or "")
        if sid not in sections:
            continue
        chapter_id = section.get("chapter_id")
        for block in section.get("blocks") or []:
            btype = block.get("block_type")
            if btype == "exception":
                add(block, section, chapter_id, "exception", str(block.get("id")))
                for item in block.get("items") or []:
                    add(item, section, chapter_id, "exception", str(block.get("id")))
                continue
            if btype == "list":
                for item in _walk_items(block):
                    if item.get("block_type") == "list":
                        continue
                    if item.get("content_class") not in EXCLUDED_CLASSES:
                        add(item, section, chapter_id, "list_item")
                continue
            if block.get("content_class") in EXCLUDED_CLASSES:
                continue
            if btype == "heading":
                continue
            if btype == "table_ref":
                add(block, section, chapter_id, "table_reference")
            elif btype == "equation_ref":
                add(block, section, chapter_id, "equation_reference")
            elif btype in {"paragraph", "reference", "unknown", "unparsed"}:
                add(block, section, chapter_id, "section_prose" if btype != "unparsed" else "unparsed_normative")

    # The structural parser stores equations as first-class objects rather than
    # emitting equation-reference blocks in every owning section.  Add a
    # provenance-bearing reference view here; equations remain excluded from
    # production requirement extraction.
    for equation in document.get("equations") or []:
        sid = str(equation.get("section_id") or "")
        if sid in sections:
            add(equation, sections[sid], sections[sid].get("chapter_id"), "equation_reference")
        else:
            errors.append(SemanticError(
                error_id=_stable_id("semantic-error", equation.get("id"), "missing-section"),
                code="MISSING_SECTION_CONTEXT", message="equation reference has no reliable owning section",
                severity="quarantine", input_span_id=equation.get("id"), page_no=(equation.get("provenance") or {}).get("page_no"),
                source_text=equation.get("text"), details={"equation_number": equation.get("equation_number")},
            ))

    if reconciliation_unparsed_path.exists():
        preserved = json.loads(reconciliation_unparsed_path.read_text(encoding="utf-8"))
        for record in preserved:
            if _is_table_residue(record):
                continue
            parent = str(record.get("parent_target_id") or "")
            sid = parent if parent.startswith("section:") else block_to_section.get(parent)
            if not sid or sid not in sections:
                errors.append(SemanticError(
                    error_id=_stable_id("semantic-error", record.get("id"), "missing-section"),
                    code="MISSING_SECTION_CONTEXT", message="reconciled normative span has no reliable section context",
                    severity="quarantine", input_span_id=record.get("id"), page_no=record.get("page_no"), source_text=record.get("text"),
                    details={"parent_target_id": parent},
                ))
                continue
            block = {
                "id": record.get("id"), "text": record.get("text"), "order": 10_000_000 + len(spans),
                "provenance": {"page_no": record.get("page_no"), "bbox": record.get("bbox"), "source_sha256": source_sha256, "pdf_text": record.get("text")},
                "content_class": "NormativeSectionContent",
                "_provenance_status": "reconciled",
            }
            add(block, sections[sid], sections[sid].get("chapter_id"), "unparsed_normative")

    # Account explicitly for every Phase 1 missing-provenance failure.  A
    # failure is resolved only when its normalized text maps to one existing
    # semantic span; repeated short markers and empty Docling groups remain
    # quarantined rather than being guessed into a section.
    missing_failures = [f for f in document.get("failures") or [] if f.get("code") == "MISSING_PROVENANCE"]
    resolved_failure_refs: list[str] = []
    quarantined_failure_refs: list[str] = []
    for failure in missing_failures:
        ref = str(failure.get("item_ref") or "")
        raw_text = str(failure.get("raw_text") or "").strip()
        matches = [span for span in spans if raw_text and _norm(span.get("source_text")) == _norm(raw_text)]
        unique_matches = {(span.get("input_span_id"), span.get("section_id"), span.get("page_no")) for span in matches}
        if len(unique_matches) == 1:
            resolved_failure_refs.append(ref)
            continue
        quarantined_failure_refs.append(ref)
        errors.append(SemanticError(
            error_id=_stable_id("semantic-error", ref, "missing-provenance"),
            code="MISSING_RELIABLE_PROVENANCE",
            message="Phase 1 missing-provenance failure was not uniquely reconciled for semantic use",
            severity="quarantine", page_no=failure.get("page_no"), source_text=raw_text or None,
            details={"failure_item_ref": ref, "candidate_count": len(unique_matches), "raw_text_empty": not bool(raw_text)},
        ))

    spans.sort(key=lambda x: (x["page_no"], x["order"], x["input_span_id"]))
    model_spans = [SemanticInputSpan.model_validate(row) for row in spans]
    summary = {
        "document_id": document_id,
        "input_span_count": len(model_spans),
        "quarantined_span_count": len(errors),
        "canonical_provenance_count": sum(s.provenance_status == "canonical" for s in model_spans),
        "reconciled_provenance_count": sum(s.provenance_status == "reconciled" for s in model_spans),
        "source_type_counts": {key: sum(s.source_type == key for s in model_spans) for key in sorted({s.source_type for s in model_spans})},
        "table_derived_requirement_input_count": 0,
        "missing_provenance_failure_count": sum(1 for f in document.get("failures") or [] if f.get("code") == "MISSING_PROVENANCE"),
        "missing_provenance_failure_resolved_count": len(resolved_failure_refs),
        "missing_provenance_failure_quarantined_count": len(quarantined_failure_refs),
        "missing_provenance_failure_quarantine_refs": quarantined_failure_refs,
    }
    return model_spans, errors, summary
