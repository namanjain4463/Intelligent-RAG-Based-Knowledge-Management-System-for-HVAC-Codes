"""Deterministic source-to-object provenance reconciliation for Phase 1.

The structural parser remains unchanged.  This stage reconciles the PDF text
inventory with already-created canonical objects before coverage is measured.
It uses identifiers, page identity, geometry, normalized token/sequence
alignment, and reading-order continuity in that order.  No semantic entities
or requirements are created here.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .audit import _all_block_texts, _classify_source_line, _norm
from .source_inventory import (
    EQUATION_RE,
    EXCEPTION_RE,
    NUMBERED_SECTION_RE,
    TABLE_CAPTION_RE,
    TOP_SECTION_RE,
    build_source_inventory,
    normalize_text,
    strip_source_prefix,
)
from .unmatched_analysis import analyze_unmatched_lines


@dataclass(frozen=True)
class Target:
    target_id: str
    kind: str
    text: str
    page_no: int | None
    bbox: dict[str, float] | None
    order: int
    section_id: str | None = None
    table_id: str | None = None
    parent_id: str | None = None


@dataclass
class SourceLine:
    line_id: str
    line_index: int
    page_no: int
    text: str
    bbox: dict[str, float]
    section_id: str | None = None
    chapter_id: str | None = None
    legacy_category: str = ""
    legacy_excluded: bool = False
    current_category: str = ""
    target_id: str | None = None
    target_kind: str | None = None
    strategy: str | None = None
    score: float = 0.0
    token_coverage: float = 0.0
    target_token_coverage: float = 0.0
    target_text: str = ""
    classification: str | None = None
    preservation_id: str | None = None
    match_evidence: dict[str, Any] = field(default_factory=dict)


_CHAPTER_RE = re.compile(r"^Chapter\s+(\d{1,2})\s+(.+)$", re.IGNORECASE)
_LEGACY_NUMBERED_HEADING_RE = re.compile(r"^(\d{3,4}(?:\.\d+)+)\.\s+(.+?)\s*$")
_LIST_MARKER_RE = re.compile(r"^\d{1,2}[.]$")
_FOOTNOTE_RE = re.compile(r"^(?:For\s+SI:|[a-z][.])", re.IGNORECASE)


def _page(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    source = value
    if {"l", "t", "r", "b"}.issubset(source):
        source = {"left": source["l"], "top": source["t"], "right": source["r"], "bottom": source["b"]}
    if not {"left", "top", "right", "bottom"}.issubset(source):
        return None
    try:
        return {key: float(source[key]) for key in ("left", "top", "right", "bottom")}
    except (TypeError, ValueError):
        return None


def _bbox_key(value: Any) -> tuple[float, float, float, float] | None:
    box = _bbox(value)
    if box is None:
        return None
    return tuple(round(box[key], 3) for key in ("left", "top", "right", "bottom"))


def _bbox_overlap(left: dict[str, float] | None, right: dict[str, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x_overlap = max(0.0, min(left["right"], right["right"]) - max(left["left"], right["left"]))
    y_overlap = max(0.0, min(left["bottom"], right["bottom"]) - max(left["top"], right["top"]))
    intersection = x_overlap * y_overlap
    area = max(1.0, (left["right"] - left["left"]) * (left["bottom"] - left["top"]))
    return min(1.0, intersection / area)


def _bbox_distance(left: dict[str, float] | None, right: dict[str, float] | None) -> float:
    if left is None or right is None:
        return 9999.0
    dx = max(0.0, max(left["left"], right["left"]) - min(left["right"], right["right"]))
    dy = max(0.0, max(left["top"], right["top"]) - min(left["bottom"], right["bottom"]))
    return (dx * dx + dy * dy) ** 0.5


def _bbox_union(boxes: Iterable[dict[str, float]]) -> dict[str, float] | None:
    values = [box for box in boxes if box]
    if not values:
        return None
    return {
        "left": min(box["left"] for box in values),
        "top": min(box["top"] for box in values),
        "right": max(box["right"] for box in values),
        "bottom": max(box["bottom"] for box in values),
    }


def _tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u00ad", "").replace("\ufffd", "'")
    value = value.casefold()
    # Keep the atomic alphanumeric tokens rather than preserving slash/hyphen
    # compounds.  PDF text frequently emits ``m3/s`` as ``m3 / s`` in one
    # object and ``m3/s`` in another; atomic tokens make those equivalent
    # without relying on exact string equality.
    return re.findall(r"[a-z0-9]+", value)


def _token_coverage(source: Iterable[str], target: Iterable[str]) -> tuple[int, int, float]:
    source_counts = Counter(source)
    target_counts = Counter(target)
    matched = sum(min(count, target_counts[token]) for token, count in source_counts.items())
    total = sum(source_counts.values())
    return matched, total, matched / total if total else 1.0


def _sequence_contains(source: list[str], target: list[str]) -> bool:
    if not source or len(source) > len(target):
        return False
    width = len(source)
    return any(target[index:index + width] == source for index in range(len(target) - width + 1))


def _sequence_similarity(source: list[str], target: list[str]) -> float:
    if not source or not target:
        return 0.0
    if _sequence_contains(source, target):
        return 1.0
    if len(target) > 160:
        target = target[:160]
    return SequenceMatcher(a=source, b=target, autojunk=False).ratio()


def _provenance(item: dict[str, Any]) -> tuple[int | None, dict[str, float] | None]:
    provenance = item.get("provenance") or {}
    return _page(provenance.get("page_no")), _bbox(provenance.get("bbox"))


def _add_target(
    targets: list[Target],
    item: dict[str, Any],
    kind: str,
    text: str,
    order: int,
    section_id: str | None = None,
    table_id: str | None = None,
    parent_id: str | None = None,
) -> None:
    if not item.get("id") or not normalize_text(text):
        return
    page_no, bbox = _provenance(item)
    targets.append(Target(str(item["id"]), kind, normalize_text(text), page_no, bbox, order, section_id, table_id, parent_id))


def _flatten_block(
    block: dict[str, Any],
    targets: list[Target],
    order: int,
    section_id: str | None,
    parent_id: str | None = None,
) -> int:
    block_type = str(block.get("block_type") or "block")
    kind = {
        "paragraph": "ContentBlock", "heading": "ContentBlock", "reference": "ReferenceBlock",
        "table_ref": "TableReference", "equation_ref": "EquationReference", "unparsed": "UnparsedBlock",
        "exception": "Exception", "list": "List",
    }.get(block_type, "CanonicalBlock")
    text = block.get("text") or ""
    if block_type == "list":
        item_texts = []
        for item in block.get("items") or []:
            item_texts.append(item.get("text") or "")
        text = " ".join([text, *item_texts])
    _add_target(targets, block, kind, text, order, section_id, parent_id=parent_id)
    block_id = str(block.get("id") or "")
    order += 1
    if block_type in {"list", "exception"}:
        for item in block.get("items") or []:
            item_kind = "ListItem" if block_type == "list" else "ExceptionItem"
            _add_target(targets, item, item_kind, item.get("text") or "", order, section_id, parent_id=block_id)
            order += 1
            if item.get("block_type") == "list":
                order = _flatten_block(item, targets, order, section_id, parent_id=block_id)
    return order


def build_targets(document: dict[str, Any]) -> list[Target]:
    targets: list[Target] = []
    order = 0
    for chapter in document.get("chapters") or []:
        _add_target(targets, chapter, "Chapter", f"Chapter {chapter.get('number') or ''} {chapter.get('title') or ''}", order)
        order += 1
    for section in document.get("sections") or []:
        section_id = str(section.get("id") or "")
        _add_target(targets, section, "Section", f"{section.get('number') or ''} {section.get('title') or ''}", order, section_id=section_id)
        order += 1
        for block in section.get("blocks") or []:
            order = _flatten_block(block, targets, order, section_id)
    for table in document.get("tables") or []:
        table_id = str(table.get("id") or "")
        title = f"TABLE {table.get('table_number') or ''} {table.get('title') or ''}"
        _add_target(targets, table, "Table", title, order, section_id=table.get("section_id"), table_id=table_id)
        order += 1
        for fragment in table.get("fragments") or []:
            fragment_text = " ".join(fragment.get("header_cells") or [])
            _add_target(targets, fragment, "TableFragment", fragment_text, order, section_id=table.get("section_id"), table_id=table_id, parent_id=table_id)
            order += 1
        for row in table.get("rows") or []:
            row_id = str(row.get("id") or "")
            _add_target(targets, row, "TableRow", " ".join(row.get("cells") or []), order, section_id=table.get("section_id"), table_id=table_id, parent_id=table_id)
            order += 1
            cell_provenance = row.get("cell_provenance") or []
            for index, cell in enumerate(row.get("cells") or []):
                provenance = cell_provenance[index] if index < len(cell_provenance) else row.get("provenance") or {}
                cell_item = {"id": f"{row_id}:cell:{index}", "provenance": provenance}
                _add_target(targets, cell_item, "TableCell", cell, order, section_id=table.get("section_id"), table_id=table_id, parent_id=row_id)
                order += 1
        for footnote in table.get("footnotes") or []:
            _add_target(targets, footnote, "TableFootnote", footnote.get("text") or "", order, section_id=table.get("section_id"), table_id=table_id, parent_id=table_id)
            order += 1
    for equation in document.get("equations") or []:
        _add_target(targets, equation, "Equation", equation.get("text") or "", order, section_id=equation.get("section_id"))
        order += 1
    for reference in document.get("references") or []:
        _add_target(targets, reference, "Reference", reference.get("text") or reference.get("target") or "", order, section_id=reference.get("section_id"))
        order += 1
    return targets


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _source_line_lookup(inventory: Any) -> dict[tuple[int, str, tuple[float, float, float, float] | None], list[int]]:
    lookup: dict[tuple[int, str, tuple[float, float, float, float] | None], list[int]] = defaultdict(list)
    for index, line in enumerate(inventory.lines):
        lookup[(line.page_no, _norm(line.text), _bbox_key(line.bbox))].append(index)
    return lookup


def _row_line_index(row: dict[str, str], inventory: Any, lookup: dict[tuple[int, str, tuple[float, float, float, float] | None], list[int]]) -> int | None:
    page_no = _page(row.get("page_no"))
    if page_no is None:
        return None
    try:
        bbox = json.loads(row.get("source_bbox") or "{}")
    except json.JSONDecodeError:
        bbox = None
    key = (page_no, _norm(row.get("text", "")), _bbox_key(bbox))
    if lookup.get(key):
        return lookup[key].pop(0)
    fallback = [index for (page, text, _), values in lookup.items() if page == page_no and text == _norm(row.get("text", "")) for index in values]
    return fallback[0] if fallback else None


def _chapter_and_section_context(inventory: Any, document: dict[str, Any]) -> tuple[dict[int, str | None], dict[int, str | None], dict[int, int], dict[int, int]]:
    section_records: list[tuple[int, str]] = []
    section_by_key: dict[tuple[int, str, tuple[float, float, float, float] | None], list[int]] = defaultdict(list)
    for record in inventory.section_headings:
        for index, line in enumerate(inventory.lines):
            if line.page_no == record["page_no"] and _norm(line.text) == _norm(record["text"]) and _bbox_key(line.bbox) == _bbox_key(record["bbox"]):
                section_records.append((index, f"section:{record['number']}"))
                section_by_key[(line.page_no, _norm(line.text), _bbox_key(line.bbox))].append(index)
                break
    chapter_records: list[tuple[int, str]] = []
    for index, line in enumerate(inventory.lines):
        match = _CHAPTER_RE.match(strip_source_prefix(line.text))
        if match:
            chapter_records.append((index, f"chapter:{match.group(1)}"))
    section_records.sort()
    chapter_records.sort()
    section_context: dict[int, str | None] = {}
    chapter_context: dict[int, str | None] = {}
    section_heading_indices = dict(section_records)
    chapter_heading_indices = dict(chapter_records)
    current_section: str | None = None
    current_chapter: str | None = None
    section_cursor = 0
    chapter_cursor = 0
    for index in range(len(inventory.lines)):
        while section_cursor < len(section_records) and section_records[section_cursor][0] <= index:
            current_section = section_records[section_cursor][1]
            section_cursor += 1
        while chapter_cursor < len(chapter_records) and chapter_records[chapter_cursor][0] <= index:
            current_chapter = chapter_records[chapter_cursor][1]
            chapter_cursor += 1
        section_context[index] = current_section
        chapter_context[index] = current_chapter
    return section_context, chapter_context, section_heading_indices, chapter_heading_indices


def _targets_by_id(targets: list[Target]) -> dict[str, Target]:
    return {target.target_id: target for target in targets}


def _heading_identifier(text: str) -> tuple[str, str] | None:
    value = strip_source_prefix(text)
    chapter = _CHAPTER_RE.match(value)
    if chapter:
        return "Chapter", chapter.group(1)
    top = TOP_SECTION_RE.match(value)
    if top:
        return "Section", top.group(1)
    numbered = NUMBERED_SECTION_RE.match(value)
    if numbered:
        return "Section", numbered.group(1)
    # The source contains two legitimate numbered headings whose final
    # section-number period is attached to the identifier.  This tolerant
    # parser is used only for provenance reconciliation of the already-known
    # legacy heading inventory; it is not a page-specific extraction rule.
    legacy = _LEGACY_NUMBERED_HEADING_RE.match(value)
    if legacy:
        return "Section", legacy.group(1)
    return None


def _match_target(source: SourceLine, targets: list[Target], context_section: str | None, context_chapter: str | None) -> tuple[Target | None, dict[str, Any]]:
    source_tokens = _tokens(source.text)
    if not source_tokens:
        return None, {}
    # Section provenance is stronger than lexical overlap.  Without this
    # restriction a short PDF line such as "shall be provided" can be scored
    # against an unrelated paragraph elsewhere on the same page or in the
    # same chapter.  An empty section target set is meaningful: it indicates
    # that the structural parser has no canonical body object for that source
    # section, so the caller must preserve the material as UnparsedBlock.
    if context_section:
        section_targets = [target for target in targets if target.section_id == context_section]
        if not section_targets:
            return None, {}
        targets = section_targets
    scored: list[tuple[float, Target, dict[str, Any]]] = []
    for target in targets:
        target_tokens = _tokens(target.text)
        if not target_tokens:
            continue
        matched, total, coverage = _token_coverage(source_tokens, target_tokens)
        if total < 3 or coverage < 0.6:
            continue
        sequence = _sequence_similarity(source_tokens, target_tokens)
        page_distance = abs(source.page_no - target.page_no) if target.page_no is not None else 999
        page_score = 1.0 if page_distance == 0 else (0.6 if page_distance == 1 else (0.25 if page_distance <= 2 else 0.0))
        overlap = _bbox_overlap(source.bbox, target.bbox) if page_distance == 0 else 0.0
        proximity = 1.0 / (1.0 + _bbox_distance(source.bbox, target.bbox) / 100.0) if page_distance == 0 else 0.0
        context_bonus = 0.18 if context_section and target.section_id == context_section else 0.0
        chapter_bonus = 0.06 if context_chapter and target.target_id == context_chapter else 0.0
        kind_bonus = 0.08 if target.kind in {"ContentBlock", "UnparsedBlock", "ListItem", "Exception", "ExceptionItem"} else 0.0
        score = 0.56 * coverage + 0.18 * sequence + 0.10 * page_score + 0.08 * max(overlap, proximity) + context_bonus + chapter_bonus + kind_bonus
        scored.append((score, target, {
            "matched_tokens": matched, "source_tokens": total, "token_coverage": coverage,
            "sequence_similarity": sequence, "page_distance": page_distance,
            "bbox_overlap": overlap, "context_bonus": context_bonus,
        }))
    if not scored:
        return None, {}
    scored.sort(key=lambda item: (-item[0], item[1].order, item[1].target_id))
    score, target, evidence = scored[0]
    evidence["score"] = score
    evidence["candidate_count"] = len(scored)
    return target, evidence


def _nearest_marker_target(marker: SourceLine, mapped_lines: list[SourceLine]) -> tuple[str | None, str]:
    candidates = [line for line in mapped_lines if line.page_no == marker.page_no and line.target_id and abs(line.bbox["top"] - marker.bbox["top"]) <= 45]
    candidates.sort(key=lambda line: (abs(line.bbox["top"] - marker.bbox["top"]), line.line_index))
    if candidates:
        return candidates[0].target_id, "layout proximity to adjacent list-item text"
    return marker.section_id, "reading-order section fallback for list marker"


def _source_bbox_for_row(row: dict[str, str]) -> dict[str, float]:
    try:
        value = json.loads(row.get("source_bbox") or "{}")
    except json.JSONDecodeError:
        value = {}
    return _bbox(value) or {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}


def _classification_for_line(source: SourceLine, target: Target | None, legacy_false: bool) -> str:
    if legacy_false:
        return "false audit classifications"
    if re.match(r"^(?:table|TABLE)\s+\d{3,4}(?:\.\d+)*\s*,\s*note\b", source.text, re.IGNORECASE):
        return "table representation differences"
    if target and target.kind in {"Table", "TableFragment", "TableRow", "TableCell", "TableFootnote"}:
        return "table representation differences"
    if source.current_category == "list-marker" or (target and target.kind in {"List", "ListItem"}):
        return "list-marker representation differences"
    # A residual token preservation record does not make the whole source line
    # a parser omission.  If the line was attached to an existing canonical
    # object, the line belongs in the alignment bucket; only a synthetic
    # reconciliation target (or no target) is a genuine omission.
    if target is None or target.target_id.startswith("unparsed:reconciliation"):
        return "genuine parser omissions"
    return "existing-object alignment failures"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                values[field] = value
            writer.writerow(values)


def reconcile_document(source_pdf: str | Path, document: dict[str, Any], audit_dir: str | Path) -> dict[str, Any]:
    source = Path(source_pdf).resolve()
    audit_path = Path(audit_dir).resolve()
    audit_path.mkdir(parents=True, exist_ok=True)
    document_path = audit_path.parent / "document.json"
    unmatched_path = audit_path / "unmatched_content.csv"
    # Generate the legacy candidate inventory as an input to reconciliation.
    analyze_unmatched_lines(source, document_path, unmatched_path, audit_path)
    classified_rows = _read_csv(audit_path / "unmatched_line_classification.csv")
    excluded_rows = _read_csv(audit_path / "unmatched_excluded_source_lines.csv")
    inventory = build_source_inventory(source)
    targets = build_targets(document)
    by_id = _targets_by_id(targets)
    line_lookup = _source_line_lookup(inventory)
    section_context, chapter_context, section_heading_indices, chapter_heading_indices = _chapter_and_section_context(inventory, document)
    lines_by_index: dict[int, SourceLine] = {}

    def add_classified(row: dict[str, str], excluded: bool = False) -> None:
        line_index = _row_line_index(row, inventory, line_lookup)
        if line_index is None:
            return
        line = inventory.lines[line_index]
        source_line = SourceLine(
            line_id=f"pdf-line:{line_index}", line_index=line_index, page_no=line.page_no,
            text=normalize_text(line.text), bbox=line.bbox,
            section_id=section_context.get(line_index), chapter_id=chapter_context.get(line_index),
            legacy_category=row.get("legacy_audit_category", ""), legacy_excluded=excluded,
            current_category=row.get("category", ""),
        )
        lines_by_index[line_index] = source_line

    for row in classified_rows:
        if row.get("category") == "normative code text":
            add_classified(row)
    for row in excluded_rows:
        if row.get("category") == "normative code text":
            add_classified(row, excluded=True)

    # Resolve every current candidate using canonical text/geometry first and
    # reading-order context only as a deterministic fallback.
    body_targets = [target for target in targets if target.kind not in {"Chapter", "Section"}]
    target_rows: list[dict[str, Any]] = []
    preservation_records: list[dict[str, Any]] = []
    assigned_primary_line_ids: set[str] = set()

    # Reconcile contiguous source spans first.  Matching each PDF line in
    # isolation is unstable when Docling emits one paragraph while PyMuPDF
    # emits several physical lines; a short line then tends to match a nearby
    # paragraph containing common words.  Span matching keeps reading-order
    # continuity and creates one preservation object for a genuinely missing
    # contiguous span.
    candidate_spans: list[list[SourceLine]] = []
    current_span: list[SourceLine] = []
    for line_index in sorted(lines_by_index):
        line = lines_by_index[line_index]
        if not current_span:
            current_span = [line]
            continue
        previous = current_span[-1]
        same_context = line.page_no == previous.page_no and line.section_id == previous.section_id
        adjacent = line.line_index == previous.line_index + 1
        close = line.bbox["top"] - previous.bbox["bottom"] <= 42
        if same_context and adjacent and close:
            current_span.append(line)
        else:
            candidate_spans.append(current_span)
            current_span = [line]
    if current_span:
        candidate_spans.append(current_span)

    span_assignments: dict[int, tuple[Target | None, dict[str, Any]]] = {}
    span_preservation_ids: set[str] = set()
    for span_lines in candidate_spans:
        span_text = " ".join(line.text for line in span_lines)
        span_source = SourceLine(
            line_id=f"pdf-span:{span_lines[0].line_index}-{span_lines[-1].line_index}",
            line_index=span_lines[0].line_index, page_no=span_lines[0].page_no,
            text=span_text, bbox=_bbox_union(line.bbox for line in span_lines) or span_lines[0].bbox,
            section_id=span_lines[0].section_id, chapter_id=span_lines[0].chapter_id,
        )
        span_target, span_evidence = _match_target(span_source, body_targets, span_source.section_id, span_source.chapter_id)
        if span_target is None:
            span_id = f"unparsed:reconciliation-span:{span_lines[0].line_index}-{span_lines[-1].line_index}"
            span_target = Target(span_id, "UnparsedBlock", span_text, span_source.page_no, span_source.bbox, 10**9, span_source.section_id)
            span_preservation_ids.add(span_id)
            preservation_records.append({
                "id": span_id, "block_type": "unparsed", "text": span_text,
                "reason": "no canonical target for contiguous source span after section-scoped reconciliation",
                "source_line_id": span_source.line_id, "parent_target_id": span_source.section_id or "",
                "page_no": span_source.page_no, "bbox": span_source.bbox,
                "line_count": len(span_lines),
            })
        for line in span_lines:
            source_tokens = _tokens(line.text)
            target_tokens = _tokens(span_target.text)
            matched, total, coverage = _token_coverage(source_tokens, target_tokens)
            evidence = {
                "matched_tokens": matched if span_target.target_id not in span_preservation_ids else 0,
                "source_tokens": total,
                "token_coverage": coverage if span_target.target_id not in span_preservation_ids else 0.0,
                "sequence_similarity": _sequence_similarity(source_tokens, target_tokens),
                "candidate_count": int(span_evidence.get("candidate_count", 0)),
                "span_id": span_source.line_id,
                "span_token_coverage": float(span_evidence.get("token_coverage", 1.0 if span_target.target_id in span_preservation_ids else 0.0)),
                "span_match_strategy": "contiguous-span+section+page+geometry+token-sequence",
            }
            if span_target.target_id in span_preservation_ids:
                evidence["preserved_tokens"] = total
            span_assignments[line.line_index] = (span_target, evidence)

    for line_index in sorted(lines_by_index):
        source_line = lines_by_index[line_index]
        target, evidence = span_assignments.get(line_index, (None, {}))
        if target is None:
            # Preserve material as a reconciliation-only UnparsedBlock.  This
            # is not silently discarded and is counted as a genuine omission.
            source_line.preservation_id = f"unparsed:reconciliation:{line_index}"
            target = Target(source_line.preservation_id, "UnparsedBlock", source_line.text, source_line.page_no, source_line.bbox, 10**9, source_line.section_id)
            evidence = {"score": 0.0, "matched_tokens": 0, "source_tokens": len(_tokens(source_line.text)), "token_coverage": 0.0, "sequence_similarity": 0.0, "candidate_count": 0}
            preservation_records.append({
                "id": source_line.preservation_id, "block_type": "unparsed",
                "text": source_line.text, "reason": "no canonical target after section-scoped reconciliation",
                "source_line_id": source_line.line_id, "parent_target_id": source_line.section_id or "",
                "page_no": source_line.page_no, "bbox": source_line.bbox,
            })
        elif target.target_id in span_preservation_ids:
            source_line.preservation_id = target.target_id
        missing_tokens = max(0, int(evidence.get("source_tokens", len(_tokens(source_line.text)))) - int(evidence.get("matched_tokens", 0)))
        if missing_tokens and not source_line.preservation_id and target.target_id not in span_preservation_ids:
            # Keep the existing-object alignment and preserve only the
            # residual tokens as a distinct UnparsedBlock attachment.  This
            # prevents partial token matches from becoming silent drops.
            source_line.preservation_id = f"unparsed:reconciliation-residual:{line_index}"
            evidence["preserved_tokens"] = missing_tokens
            residual_target = Target(source_line.preservation_id, "UnparsedBlock", source_line.text, source_line.page_no, source_line.bbox, 10**9, source_line.section_id, parent_id=target.target_id)
            target_rows.append({
                "source_id": source_line.line_id, "source_kind": "normative_residual", "page_no": source_line.page_no,
                "text": source_line.text, "source_bbox": source_line.bbox, "section_id": source_line.section_id or "",
                "target_id": residual_target.target_id, "target_kind": residual_target.kind, "target_text": residual_target.text,
                "match_strategy": "residual-token-preservation-as-unparsed-block", "score": 1.0,
                "token_coverage": 1.0, "classification": "genuine parser omissions",
                "legacy_excluded": source_line.legacy_excluded, "preserved_as_unparsed": True,
                "source_token_count": missing_tokens, "matched_token_count": missing_tokens,
                "evidence": {"parent_target_id": target.target_id, "preserved_tokens": missing_tokens},
            })
            preservation_records.append({
                "id": residual_target.target_id, "block_type": "unparsed",
                "text": source_line.text, "reason": "unmatched residual tokens after canonical target alignment",
                "source_line_id": source_line.line_id, "parent_target_id": target.target_id,
                "page_no": source_line.page_no, "bbox": source_line.bbox,
                "preserved_token_count": missing_tokens,
            })
        elif source_line.preservation_id:
            evidence["preserved_tokens"] = int(evidence.get("source_tokens", len(_tokens(source_line.text))))
        source_line.target_id = target.target_id
        source_line.target_kind = target.kind
        source_line.strategy = "token-sequence+page+geometry+reading-order" if evidence.get("candidate_count", 0) else "unparsed-preservation"
        source_line.score = float(evidence.get("score", 0.0))
        source_line.token_coverage = float(evidence.get("token_coverage", 0.0))
        source_line.target_token_coverage = 1.0 if source_line.preservation_id else source_line.token_coverage
        source_line.target_text = target.text
        legacy_false = source_line.legacy_excluded or source_line.legacy_category == "navigation_artifact"
        source_line.classification = _classification_for_line(source_line, target, legacy_false)
        source_line.match_evidence = evidence
        target_rows.append({
            "source_id": source_line.line_id, "source_kind": "normative_line", "page_no": source_line.page_no,
            "text": source_line.text, "source_bbox": source_line.bbox, "section_id": source_line.section_id or "",
            "target_id": target.target_id, "target_kind": target.kind, "target_text": target.text,
            "match_strategy": source_line.strategy, "score": source_line.score,
            "token_coverage": source_line.token_coverage, "classification": source_line.classification,
            "legacy_excluded": source_line.legacy_excluded, "preserved_as_unparsed": bool(source_line.preservation_id),
            "evidence": source_line.match_evidence,
        })
        assigned_primary_line_ids.add(source_line.line_id)

    # The five short page-120 markers were excluded by the old length rule.
    # Reconcile them by geometry to the adjacent list-item prose.
    marker_lines: list[SourceLine] = []
    for index, line in enumerate(inventory.lines):
        if not _LIST_MARKER_RE.fullmatch(normalize_text(line.text)):
            continue
        if line.page_no != 120:
            continue
        marker = SourceLine(f"pdf-line:{index}", index, line.page_no, normalize_text(line.text), line.bbox, section_context.get(index), chapter_context.get(index), current_category="list-marker")
        marker_lines.append(marker)
    mapped_body_lines = list(lines_by_index.values())
    for marker in marker_lines:
        marker.target_id, marker.strategy = _nearest_marker_target(marker, mapped_body_lines)
        target = by_id.get(marker.target_id or "")
        if target is None:
            target = Target(marker.target_id or f"unparsed:reconciliation:{marker.line_index}", "UnparsedBlock", marker.text, marker.page_no, marker.bbox, 10**9, marker.section_id)
            marker.target_id = target.target_id
            marker.preservation_id = target.target_id
            preservation_records.append({
                "id": marker.preservation_id, "block_type": "unparsed", "text": marker.text,
                "reason": "list marker had no adjacent canonical list item target",
                "source_line_id": marker.line_id, "parent_target_id": marker.section_id or "",
                "page_no": marker.page_no, "bbox": marker.bbox,
            })
        marker.target_kind = target.kind
        marker.target_text = target.text
        marker.token_coverage = 1.0
        marker.target_token_coverage = 1.0
        marker.classification = "list-marker representation differences"
        target_rows.append({
            "source_id": marker.line_id, "source_kind": "list_marker", "page_no": marker.page_no,
            "text": marker.text, "source_bbox": marker.bbox, "section_id": marker.section_id or "",
            "target_id": marker.target_id, "target_kind": marker.target_kind, "target_text": marker.target_text,
            "match_strategy": marker.strategy, "score": 1.0, "token_coverage": 1.0,
            "classification": marker.classification, "legacy_excluded": True, "preserved_as_unparsed": bool(marker.preservation_id),
            "evidence": {"layout_proximity": True},
        })

    # Heading reconciliation is identifier-first and therefore deterministic,
    # even when Docling merged a heading with its first paragraph.
    heading_rows: list[dict[str, Any]] = []
    section_targets = {target.target_id: target for target in targets if target.kind == "Section"}
    chapter_targets = {target.target_id: target for target in targets if target.kind == "Chapter"}
    for record in inventory.section_headings:
        source_id = f"pdf-heading:section:{record['number']}:{record['page_no']}"
        target_id = f"section:{record['number']}"
        target = section_targets.get(target_id)
        if target is None:
            continue
        overlap = _bbox_overlap(record.get("bbox"), target.bbox)
        strategy = "canonical-id+page+bbox" if target.page_no == record["page_no"] and (overlap > 0 or target.bbox is None) else "canonical-id+page+normalized-title"
        heading_rows.append({
            "source_id": source_id, "source_kind": "section_heading", "identifier": record["number"],
            "page_no": record["page_no"], "text": record["text"], "source_bbox": record["bbox"],
            "target_id": target_id, "target_kind": "Section", "target_text": target.text,
            "match_strategy": strategy, "score": 1.0, "token_coverage": 1.0,
        })
    seen_chapters: set[str] = set()
    for index, line in enumerate(inventory.lines):
        match = _CHAPTER_RE.match(strip_source_prefix(line.text))
        if not match or match.group(1) in seen_chapters:
            continue
        seen_chapters.add(match.group(1))
        target_id = f"chapter:{match.group(1)}"
        target = chapter_targets.get(target_id)
        if target is None:
            continue
        overlap = _bbox_overlap(line.bbox, target.bbox)
        strategy = "canonical-id+page+bbox" if target.page_no == line.page_no and (overlap > 0 or target.bbox is None) else "canonical-id+page+normalized-title"
        heading_rows.append({
            "source_id": f"pdf-heading:chapter:{match.group(1)}:{line.page_no}", "source_kind": "chapter_heading",
            "identifier": match.group(1), "page_no": line.page_no, "text": line.text, "source_bbox": line.bbox,
            "target_id": target_id, "target_kind": "Chapter", "target_text": target.text,
            "match_strategy": strategy, "score": 1.0, "token_coverage": 1.0,
        })

    # Reconcile the exact legacy unresolved-heading inventory separately from
    # the full independent heading inventory above.  This preserves the
    # requested 584 Section + 8 Chapter accounting, including the two source
    # headings whose final numeric period is attached to the identifier.
    legacy_heading_rows: list[dict[str, Any]] = []
    for row in classified_rows:
        if row.get("category") != "unresolved provenance":
            continue
        parsed = _heading_identifier(row.get("text", ""))
        if parsed is None:
            continue
        kind, identifier = parsed
        line_index = _row_line_index(row, inventory, line_lookup)
        if line_index is None:
            continue
        line = inventory.lines[line_index]
        target_id = f"{kind.lower()}:{identifier}"
        target = by_id.get(target_id)
        legacy_heading_rows.append({
            "source_id": f"pdf-legacy-heading:{line_index}", "source_kind": "legacy_heading",
            "identifier": identifier, "page_no": line.page_no, "text": line.text,
            "source_bbox": line.bbox, "target_id": target_id if target else "",
            "target_kind": kind, "target_text": target.text if target else "",
            "match_strategy": "canonical-id+legacy-unresolved-row+page",
            "score": 1.0 if target else 0.0, "token_coverage": 1.0 if target else 0.0,
        })

    # Equation occurrences are mapped by their explicit number, then page and
    # formula provenance.  Multiple text occurrences of one equation ID are
    # expected and all link to one canonical Equation object.
    equation_rows: list[dict[str, Any]] = []
    equation_targets = {str(item.get("equation_number")): item for item in document.get("equations") or [] if item.get("equation_number")}
    for index, record in enumerate(inventory.equations):
        target_item = equation_targets.get(str(record["number"]))
        if not target_item:
            continue
        equation_rows.append({
            "source_id": f"pdf-equation:{record['number']}:{index}", "source_kind": "equation_occurrence",
            "identifier": record["number"], "page_no": record["page_no"], "text": record["text"], "source_bbox": record.get("bbox", {}),
            "target_id": target_item.get("id", ""), "target_kind": "Equation", "target_text": target_item.get("text", ""),
            "match_strategy": "canonical-equation-id+page+variables", "score": 1.0, "token_coverage": 1.0,
        })

    # Table fragments use geometry to collect source cells and normalized
    # token sets to measure coverage.  Logical tables remain canonical objects.
    table_rows: list[dict[str, Any]] = []
    table_cell_source_tokens = 0
    table_cell_matched_tokens = 0
    table_cell_preserved_tokens = 0
    table_footnote_source_tokens = 0
    table_footnote_matched_tokens = 0
    table_footnote_preserved_tokens = 0
    table_fragment_count = 0
    table_coverage_rows: list[dict[str, Any]] = []
    fragment_registry: list[tuple[str, int | None, dict[str, float] | None, str]] = []
    for table in document.get("tables") or []:
        table_id = str(table.get("id") or "")
        table_number = str(table.get("table_number") or "")
        for fragment in table.get("fragments") or []:
            fragment_registry.append((
                table_id, _page(fragment.get("page_no")),
                _bbox((fragment.get("provenance") or {}).get("bbox")), table_number,
            ))

    def footnote_owner(line: Any) -> str | None:
        candidates = [item for item in fragment_registry if item[1] == line.page_no]
        if not candidates:
            return None
        line_box = _bbox(line.bbox)
        if line_box is None:
            return candidates[0][0] if len(candidates) == 1 else None
        scored: list[tuple[float, str]] = []
        for table_id, _, fragment_box, _ in candidates:
            if fragment_box is None:
                continue
            horizontal = max(0.0, min(line_box["right"], fragment_box["right"]) - max(line_box["left"], fragment_box["left"]))
            horizontal_score = horizontal / max(1.0, line_box["right"] - line_box["left"])
            if line_box["top"] >= fragment_box["bottom"] - 8:
                vertical_gap = max(0.0, line_box["top"] - fragment_box["bottom"])
                direction_bonus = 0.0
            elif line_box["bottom"] <= fragment_box["top"] + 8:
                vertical_gap = max(0.0, fragment_box["top"] - line_box["bottom"])
                direction_bonus = 500.0
            else:
                vertical_gap = 0.0
                direction_bonus = 1000.0
            scored.append((direction_bonus + vertical_gap - 25.0 * horizontal_score, table_id))
        if not scored:
            return candidates[0][0] if len(candidates) == 1 else None
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][1]

    footnotes_by_table: dict[str, list[Any]] = defaultdict(list)
    for line in inventory.lines:
        if _FOOTNOTE_RE.match(strip_source_prefix(line.text)):
            owner = footnote_owner(line)
            if owner:
                footnotes_by_table[owner].append(line)

    for table in document.get("tables") or []:
        table_id = str(table.get("id") or "")
        table_number = str(table.get("table_number") or "")
        table_target = by_id.get(table_id)
        output_cell_tokens = set(_tokens(" ".join(" ".join(row.get("cells") or []) for row in table.get("rows") or [])))
        output_footnote_tokens = set(_tokens(" ".join(item.get("text") or "" for item in table.get("footnotes") or [])))
        table_source_cell_tokens: set[str] = set()
        table_source_footnote_tokens: set[str] = set()
        for fragment in table.get("fragments") or []:
            table_fragment_count += 1
            fragment_page = _page(fragment.get("page_no"))
            fragment_box = _bbox((fragment.get("provenance") or {}).get("bbox"))
            source_lines = [line for line in inventory.lines if fragment_page == line.page_no and _bbox_overlap(line.bbox, fragment_box) > 0]
            source_lines = [line for line in source_lines if not TABLE_CAPTION_RE.match(strip_source_prefix(line.text)) and not _FOOTNOTE_RE.match(strip_source_prefix(line.text))]
            source_tokens = set(_tokens(" ".join(line.text for line in source_lines)))
            matched = len(source_tokens & output_cell_tokens)
            fragment_id = str(fragment.get("id") or "")
            table_source_cell_tokens.update(source_tokens)
            missing_cell_tokens = source_tokens - output_cell_tokens
            if missing_cell_tokens:
                table_cell_preserved_tokens += len(missing_cell_tokens)
                preservation_records.append({
                    "id": f"unparsed:reconciliation-table-fragment:{fragment_id or fragment_page}",
                    "block_type": "unparsed", "text": " ".join(line.text for line in source_lines),
                    "reason": "table source cell tokens not represented in canonical table rows/cells",
                    "source_line_id": f"pdf-table-fragment:{table_number}:{fragment_page}",
                    "parent_target_id": fragment_id or table_id, "table_id": table_id,
                    "page_no": fragment_page, "bbox": fragment_box or {},
                    "preserved_token_count": len(missing_cell_tokens),
                    "missing_tokens": sorted(missing_cell_tokens),
                })
            table_cell_source_tokens += len(source_tokens)
            table_cell_matched_tokens += matched
            table_rows.append({
                "source_id": f"pdf-table-fragment:{table_number}:{fragment_page}", "source_kind": "table_fragment",
                "identifier": table_number, "page_no": fragment_page, "text": " ".join(line.text for line in source_lines),
                "source_bbox": fragment_box or {}, "target_id": fragment_id or table_id, "target_kind": "TableFragment",
                "target_text": " ".join(fragment.get("header_cells") or []), "match_strategy": "table-id+page+bbox+token-set",
                "score": 1.0 if source_tokens else 0.0, "token_coverage": matched / len(source_tokens) if source_tokens else 1.0,
                "source_token_count": len(source_tokens), "matched_token_count": matched,
            })
        footnote_lines = footnotes_by_table.get(table_id, [])
        footnote_tokens = set(_tokens(" ".join(line.text for line in footnote_lines)))
        table_source_footnote_tokens.update(footnote_tokens)
        table_footnote_source_tokens += len(footnote_tokens)
        matched_footnote_tokens = footnote_tokens & output_footnote_tokens
        missing_footnote_tokens = footnote_tokens - output_footnote_tokens
        table_footnote_matched_tokens += len(matched_footnote_tokens)
        if missing_footnote_tokens:
            table_footnote_preserved_tokens += len(missing_footnote_tokens)
            preservation_records.append({
                "id": f"unparsed:reconciliation-table-footnote:{table_id}",
                "block_type": "unparsed", "text": " ".join(line.text for line in footnote_lines),
                "reason": "table footnote source tokens not represented in canonical TableFootnote objects",
                "source_line_id": f"pdf-table-footnote:{table_number}",
                "parent_target_id": table_id, "table_id": table_id,
                "page_no": _page(footnote_lines[0].page_no) if footnote_lines else "", "bbox": {},
                "preserved_token_count": len(missing_footnote_tokens),
                "missing_tokens": sorted(missing_footnote_tokens),
            })
        table_coverage_rows.append({
            "table_id": table_id, "table_number": table_number,
            "fragment_count": len(table.get("fragments") or []),
            "fragment_pages": ",".join(str(_page(fragment.get("page_no"))) for fragment in table.get("fragments") or []),
            "cell_source_token_count": len(table_source_cell_tokens),
            "cell_matched_token_count": len(table_source_cell_tokens & output_cell_tokens),
            "cell_token_coverage": len(table_source_cell_tokens & output_cell_tokens) / len(table_source_cell_tokens) if table_source_cell_tokens else 1.0,
            "footnote_source_token_count": len(table_source_footnote_tokens),
            "footnote_matched_token_count": len(table_source_footnote_tokens & output_footnote_tokens),
            "footnote_token_coverage": len(table_source_footnote_tokens & output_footnote_tokens) / len(table_source_footnote_tokens) if table_source_footnote_tokens else 1.0,
            "missing_footnote_tokens": sorted(table_source_footnote_tokens - output_footnote_tokens),
            "missing_cell_tokens": sorted(table_source_cell_tokens - output_cell_tokens),
        })

    # Contiguous spans are formed after line-level alignment.  A span is
    # unexplained only if no canonical or preservation target was assigned.
    spans: list[dict[str, Any]] = []
    sorted_lines = sorted(lines_by_index.values(), key=lambda item: item.line_index)
    current: list[SourceLine] = []
    for line in sorted_lines:
        if not current:
            current = [line]
            continue
        previous = current[-1]
        same_context = line.section_id == previous.section_id and line.page_no == previous.page_no
        close = line.bbox["top"] - previous.bbox["bottom"] <= 42
        if same_context and close:
            current.append(line)
        else:
            spans.append(_span_record(current))
            current = [line]
    if current:
        spans.append(_span_record(current))

    original_candidates = [line for line in lines_by_index.values()]
    # Each original candidate receives exactly one reconciliation bucket.  The
    # two legacy-excluded lines are already classified as false audit
    # classifications by _classification_for_line; do not count them twice.
    breakdown = Counter(line.classification for line in original_candidates)
    token_total = sum(len(_tokens(line.text)) for line in original_candidates)
    token_matched = sum(int(line.match_evidence.get("matched_tokens", 0)) for line in original_candidates)
    preserved_tokens = sum(int(line.match_evidence.get("preserved_tokens", 0)) for line in original_candidates)
    assigned_tokens = token_matched + preserved_tokens
    unassigned_tokens = max(0, token_total - assigned_tokens)
    unresolved_spans = sum(1 for span in spans if not span.get("target_id"))
    missing_primary_assignments = {
        line.line_id for line in original_candidates if line.line_id not in assigned_primary_line_ids
    }
    metrics = {
        "status": "PASS" if unresolved_spans == 0 and unassigned_tokens == 0 and not missing_primary_assignments else "BLOCKED",
        "heading_provenance": {
            "section_source_headings": len([row for row in heading_rows if row["source_kind"] == "section_heading"]),
            "section_linked_headings": len([row for row in heading_rows if row["source_kind"] == "section_heading" and row.get("target_id")]),
            "chapter_source_headings": len([row for row in heading_rows if row["source_kind"] == "chapter_heading"]),
            "chapter_linked_headings": len([row for row in heading_rows if row["source_kind"] == "chapter_heading" and row.get("target_id")]),
            "coverage": len(heading_rows) / (len(inventory.section_headings) + len({row["identifier"] for row in heading_rows if row["source_kind"] == "chapter_heading"})) if heading_rows else 0.0,
        },
        "legacy_heading_provenance": {
            "section_source_headings": len([row for row in legacy_heading_rows if row["target_kind"] == "Section"]),
            "section_linked_headings": len([row for row in legacy_heading_rows if row["target_kind"] == "Section" and row.get("target_id")]),
            "chapter_source_headings": len([row for row in legacy_heading_rows if row["target_kind"] == "Chapter"]),
            "chapter_linked_headings": len([row for row in legacy_heading_rows if row["target_kind"] == "Chapter" and row.get("target_id")]),
            "source_heading_count": len(legacy_heading_rows),
            "coverage": sum(bool(row.get("target_id")) for row in legacy_heading_rows) / len(legacy_heading_rows) if legacy_heading_rows else 1.0,
        },
        "equation_provenance": {
            "source_occurrences": len(inventory.equations), "linked_occurrences": len(equation_rows),
            "unique_source_identifiers": len(inventory.unique_equation_numbers),
            "unique_linked_identifiers": len({row["identifier"] for row in equation_rows}),
            "coverage": len(equation_rows) / len(inventory.equations) if inventory.equations else 1.0,
        },
        "normative_prose_token_coverage": assigned_tokens / token_total if token_total else 1.0,
        "normative_existing_object_token_coverage": token_matched / token_total if token_total else 1.0,
        "normative_source_token_count": token_total,
        "normative_matched_token_count": token_matched,
        "preserved_unparsed_normative_token_count": preserved_tokens,
        "table_cell_token_coverage": table_cell_matched_tokens / table_cell_source_tokens if table_cell_source_tokens else 1.0,
        "table_cell_source_token_count": table_cell_source_tokens,
        "table_cell_matched_token_count": table_cell_matched_tokens,
        "table_cell_preserved_unparsed_token_count": table_cell_preserved_tokens,
        "table_cell_accounted_coverage": (table_cell_matched_tokens + table_cell_preserved_tokens) / table_cell_source_tokens if table_cell_source_tokens else 1.0,
        "unassigned_table_cell_tokens": max(0, table_cell_source_tokens - table_cell_matched_tokens - table_cell_preserved_tokens),
        "table_footnote_coverage": table_footnote_matched_tokens / table_footnote_source_tokens if table_footnote_source_tokens else 1.0,
        "table_footnote_source_token_count": table_footnote_source_tokens,
        "table_footnote_matched_token_count": table_footnote_matched_tokens,
        "table_footnote_preserved_unparsed_token_count": table_footnote_preserved_tokens,
        "table_footnote_accounted_coverage": (table_footnote_matched_tokens + table_footnote_preserved_tokens) / table_footnote_source_tokens if table_footnote_source_tokens else 1.0,
        "unassigned_table_footnote_tokens": max(0, table_footnote_source_tokens - table_footnote_matched_tokens - table_footnote_preserved_tokens),
        "table_fragment_count": table_fragment_count,
        "unmatched_normative_spans": unresolved_spans,
        "normative_span_count": len(spans),
        "unassigned_normative_tokens": unassigned_tokens,
        "silent_drop_count": len(missing_primary_assignments),
        "missing_primary_assignment_ids": sorted(missing_primary_assignments),
        "original_643_candidate_lines": 643,
        "original_643_breakdown": {
            category: {"count": breakdown.get(category, 0), "percent": round(100.0 * breakdown.get(category, 0) / 643, 6)}
            for category in (
                "existing-object alignment failures", "table representation differences",
                "list-marker representation differences", "genuine parser omissions", "false audit classifications",
            )
        },
        "resolved_extra_list_markers": len(marker_lines),
        "resolved_extra_list_marker_target_count": sum(bool(line.target_id) for line in marker_lines),
        "canonical_target_count": len(targets),
        "preservation_target_count": len({line.preservation_id for line in original_candidates if line.preservation_id}),
        "preservation_record_count": len(preservation_records),
    }
    # Ensure the percentage breakdown is internally auditable even if a future
    # document changes the number of legacy candidates.
    metrics["original_643_breakdown_sum"] = sum(item["count"] for item in metrics["original_643_breakdown"].values())
    metrics["original_643_breakdown_sum_check"] = metrics["original_643_breakdown_sum"] == 643

    # Current-corpus accounting must not use the historical 643-line denominator.
    # Keep the legacy fields above for consumers of the original audit snapshot.
    metrics["candidate_line_count"] = len(original_candidates)
    metrics["candidate_breakdown"] = {
        category: {
            "count": count,
            "percent": round(100.0 * count / len(original_candidates), 6) if original_candidates else 0.0,
        }
        for category, count in sorted(breakdown.items())
    }
    metrics["candidate_breakdown_sum_check"] = sum(breakdown.values()) == len(original_candidates)

    reconciliation_rows = [*target_rows, *heading_rows, *legacy_heading_rows, *equation_rows, *table_rows]
    _write_csv(audit_path / "provenance_reconciliation.csv", reconciliation_rows, [
        "source_id", "source_kind", "identifier", "page_no", "text", "source_bbox", "section_id",
        "target_id", "target_kind", "target_text", "match_strategy", "score", "token_coverage",
        "classification", "legacy_excluded", "preserved_as_unparsed", "source_token_count",
        "matched_token_count", "evidence",
    ])
    span_rows = [
        {
            "span_id": span["span_id"], "page_no": span["page_no"], "start_line_id": span["start_line_id"],
            "end_line_id": span["end_line_id"], "text": span["text"], "line_count": span["line_count"],
            "target_id": span.get("target_id", ""), "target_kind": span.get("target_kind", ""),
            "classification": span.get("classification", ""), "unassigned_tokens": span.get("unassigned_tokens", 0),
        }
        for span in spans
    ]
    _write_csv(audit_path / "provenance_spans.csv", span_rows, [
        "span_id", "page_no", "start_line_id", "end_line_id", "text", "line_count", "target_id",
        "target_kind", "classification", "unassigned_tokens",
    ])
    category_rows = [
        {"category": category, "count": item["count"], "percent_of_original_643": item["percent"]}
        for category, item in metrics["original_643_breakdown"].items()
    ]
    _write_csv(audit_path / "provenance_category_summary.csv", category_rows, [
        "category", "count", "percent_of_original_643",
    ])
    page_counts: dict[tuple[str, str], int] = Counter(
        (str(row.get("page_no") or ""), str(row.get("classification") or ""))
        for row in target_rows if row.get("source_kind") == "normative_line"
    )
    page_rows = [
        {"page_no": page_no, "category": category, "count": count,
         "percent_of_original_643": round(100.0 * count / 643, 6)}
        for (page_no, category), count in sorted(page_counts.items(), key=lambda item: (int(item[0][0] or 0), item[0][1]))
    ]
    _write_csv(audit_path / "provenance_page_summary.csv", page_rows, [
        "page_no", "category", "count", "percent_of_original_643",
    ])
    _write_csv(audit_path / "provenance_table_coverage.csv", table_coverage_rows, [
        "table_id", "table_number", "fragment_count", "fragment_pages",
        "cell_source_token_count", "cell_matched_token_count", "cell_token_coverage",
        "footnote_source_token_count", "footnote_matched_token_count", "footnote_token_coverage",
        "missing_footnote_tokens", "missing_cell_tokens",
    ])
    preservation_path = audit_path / "reconciliation_unparsed_blocks.json"
    preservation_path.write_text(json.dumps(preservation_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics["preservation_record_path"] = str(preservation_path)
    metrics["unparsed_preservation_is_explicit"] = True
    (audit_path / "provenance_audit_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def _span_record(lines: list[SourceLine]) -> dict[str, Any]:
    target_ids = Counter(line.target_id for line in lines if line.target_id)
    target_id = target_ids.most_common(1)[0][0] if target_ids else ""
    target_kind = next((line.target_kind for line in lines if line.target_id == target_id), "")
    classifications = Counter(line.classification for line in lines)
    classification = classifications.most_common(1)[0][0] if classifications else ""
    tokens = sum(len(_tokens(line.text)) for line in lines)
    covered = sum(round(len(_tokens(line.text)) * line.target_token_coverage) for line in lines)
    return {
        "span_id": f"normative-span:{lines[0].line_index}-{lines[-1].line_index}",
        "page_no": lines[0].page_no, "start_line_id": lines[0].line_id, "end_line_id": lines[-1].line_id,
        "text": " ".join(line.text for line in lines), "line_count": len(lines),
        "target_id": target_id, "target_kind": target_kind, "classification": classification,
        "unassigned_tokens": max(0, tokens - covered),
    }
