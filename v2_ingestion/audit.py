"""Source-to-output structural coverage audit for the v2 corpus.

The audit deliberately uses a separate PyMuPDF text/geometry inventory rather
than trusting Docling's object inventory.  It compares identifiers and source
provenance, and records unresolved content instead of treating it as absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .source_inventory import (
    EXCEPTION_RE,
    NUMBERED_SECTION_RE,
    STANDARD_RE,
    TABLE_CAPTION_RE,
    TOP_SECTION_RE,
    build_source_inventory,
    normalize_text,
    strip_source_prefix,
)


def _norm(value: Any) -> str:
    return normalize_text(str(value or "")).rstrip(".").casefold()


def _page(provenance: dict[str, Any] | None) -> int | None:
    value = (provenance or {}).get("page_no")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _all_block_texts(document: dict[str, Any]) -> dict[int, list[str]]:
    """Build output text by page without using the full page PDF text field."""

    by_page: dict[int, list[str]] = defaultdict(list)

    def add(page: int | None, text: Any) -> None:
        value = normalize_text(str(text or ""))
        if page is not None and value:
            by_page[page].append(value)

    def add_block(block: dict[str, Any], fallback_page: int | None = None) -> None:
        provenance = block.get("provenance") or {}
        page = _page(provenance) or fallback_page
        add(page, block.get("text"))
        for item in block.get("items") or []:
            add_block(item, page)

    for section in document.get("sections") or []:
        section_page = _page(section.get("provenance"))
        add(section_page, section.get("title"))
        for block in section.get("blocks") or []:
            add_block(block, section_page)
    for table in document.get("tables") or []:
        table_page = _page(table.get("provenance"))
        add(table_page, table.get("title"))
        for row in table.get("rows") or []:
            row_page = _page(row.get("provenance")) or table_page
            for cell in row.get("cells") or []:
                add(row_page, cell)
        for footnote in table.get("footnotes") or []:
            add(_page(footnote.get("provenance")) or table_page, footnote.get("text"))
    for equation in document.get("equations") or []:
        add(_page(equation.get("provenance")), equation.get("text"))
    for reference in document.get("references") or []:
        add(_page(reference.get("provenance")), reference.get("text"))
    for block in document.get("unparsed_blocks") or []:
        add(_page(block.get("provenance")), block.get("text"))
    return by_page


def _exception_blocks(inventory: Any) -> list[dict[str, Any]]:
    """Group each ``Exception:`` marker through the next structural heading."""
    return inventory.exception_blocks()


def _classify_source_line(text: str, inventory: Any, page_no: int) -> str:
    value = strip_source_prefix(text)
    upper = value.upper()
    if TOP_SECTION_RE.match(value) or NUMBERED_SECTION_RE.match(value):
        return "section_heading"
    if TABLE_CAPTION_RE.match(value):
        return "table_caption"
    if re.search(r"\b(?:Equation|Eq\.?)\s*\d+\s*[-–]\s*\d+", value, re.IGNORECASE):
        return "equation"
    if EXCEPTION_RE.match(value):
        return "exception"
    if upper.startswith("INSIGHTS"):
        return "insights"
    if upper.startswith("USER NOTE"):
        return "user_note"
    if re.match(r"^(?:For SI:|[a-z]\.)", value, re.IGNORECASE):
        return "table_footnote_candidate"
    if any(token in upper for token in ("CONTENTS", "INDEX", "GO TO", "PAGE ")) or re.fullmatch(r"\d+[.]?", value):
        return "navigation_artifact"
    if any(regex.search(value) for regex in (inventory_reference_regexes(inventory))):
        return "cross_reference"
    if upper.startswith(("SOURCE:", "HTTPS://", "HTTP://")):
        return "source_artifact"
    return "normative_or_commentary_text"


def _positive_navigation_evidence(text: str, line: Any, page_height: float = 792.0) -> bool:
    """Return true only when text and page geometry positively indicate navigation.

    Words such as ``index`` are ordinary regulatory vocabulary in this corpus.
    Navigation is therefore accepted only for explicit navigation/page-number
    text positioned in a page header or footer band.
    """
    bbox = getattr(line, "bbox", None)
    if not bbox:
        return False
    value = normalize_text(text)
    upper = value.upper()
    in_margin = float(bbox.get("top", 0.0)) <= 55.0 or float(bbox.get("bottom", 0.0)) >= page_height - 55.0
    if not in_margin:
        return False
    return bool(
        re.fullmatch(r"\d{1,4}[.]?", value)
        or re.search(r"\b(?:contents|table\s+of\s+contents|index|go\s+to|previous|next|page\s+\d+)\b", upper)
    )


def inventory_reference_regexes(inventory: Any) -> tuple[re.Pattern[str], ...]:
    # The inventory already applied the authoritative patterns; this helper
    # keeps line classification independent of the output references.
    del inventory
    from .source_inventory import CHAPTER_REF_RE, SECTION_REF_RE, TABLE_REF_RE
    return SECTION_REF_RE, TABLE_REF_RE, CHAPTER_REF_RE, STANDARD_RE


def _source_text_audit(inventory: Any, document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output_pages = _all_block_texts(document)
    output_joined = {page: " ".join(values).casefold() for page, values in output_pages.items()}
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for line in inventory.lines:
        text = normalize_text(line.text)
        category = _classify_source_line(text, inventory, line.page_no)
        positive_navigation = _positive_navigation_evidence(text, line)
        meaningful = len(_norm(text)) >= 5 and not positive_navigation
        matched = bool(text and _norm(text) in output_joined.get(line.page_no, ""))
        if meaningful:
            counts["meaningful_source_lines"] += 1
            counts["matched_source_lines"] += int(matched)
        if not matched and meaningful:
            rows.append({
                "record_type": "source_line",
                "status": "unmatched",
                "category": category,
                "page_no": line.page_no,
                "end_page_no": line.page_no,
                "identifier": "",
                "text": text,
                "reason": "No normalized output block on the same source page contains this line.",
            })

    for block in document.get("unparsed_blocks") or []:
        rows.append({
            "record_type": "output_block",
            "status": "unparsed_output",
            "category": "UnparsedBlock",
            "page_no": _page(block.get("provenance")),
            "end_page_no": _page(block.get("provenance")),
            "identifier": block.get("id", ""),
            "text": block.get("text", ""),
            "reason": block.get("reason", ""),
        })
    return rows, dict(counts)


def _section_rows(inventory: Any, document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = inventory.section_headings
    output = [section for section in document.get("sections") or [] if section.get("number")]
    by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in output:
        by_number[str(section.get("number"))].append(section)
    rows: list[dict[str, Any]] = []
    statuses = Counter()
    source_keys = set()
    for item in source:
        number = item["number"]
        source_keys.add(number)
        matches = by_number.get(number, [])
        if not matches:
            status = "missing"
            match = {}
        elif len(matches) > 1:
            status = "duplicate_output"
            match = matches[0]
        else:
            match = matches[0]
            status = "matched" if _norm(item["title"]) == _norm(match.get("title")) else "title_mismatch"
        statuses[status] += 1
        rows.append({
            "number": number, "source_kind": item["kind"], "source_title": item["title"],
            "source_page_no": item["page_no"], "source_bbox": item["bbox"],
            "output_id": match.get("id", ""), "output_title": match.get("title", ""),
            "output_page_no": _page(match.get("provenance")), "output_level": match.get("level", ""),
            "status": status,
            "details": "Output fallback provenance" if match.get("provenance", {}).get("docling_label") == "pdf_heading_fallback" else "",
        })
    for section in output:
        if str(section.get("number")) not in source_keys:
            statuses["output_only"] += 1
            rows.append({
                "number": section.get("number", ""), "source_kind": "", "source_title": "",
                "source_page_no": "", "source_bbox": {}, "output_id": section.get("id", ""),
                "output_title": section.get("title", ""), "output_page_no": _page(section.get("provenance")),
                "output_level": section.get("level", ""), "status": "output_only", "details": "",
            })
    synthetic = [section for section in document.get("sections") or [] if not section.get("number")]
    for section in synthetic:
        statuses["output_only"] += 1
        rows.append({
            "number": "", "source_kind": "", "source_title": "", "source_page_no": "", "source_bbox": {},
            "output_id": section.get("id", ""), "output_title": section.get("title", ""),
            "output_page_no": _page(section.get("provenance")), "output_level": section.get("level", ""),
            "status": "output_only", "details": "Synthetic unassigned-content section",
        })
    return rows, dict(statuses)


def _table_rows(inventory: Any, document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in inventory.table_captions:
        source_by_number[item["number"]].append(item)
    output_by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for table in document.get("tables") or []:
        number = table.get("table_number")
        if number:
            output_by_number[str(number)].append(table)
    rows: list[dict[str, Any]] = []
    statuses = Counter()
    for number, occurrences in source_by_number.items():
        matches = output_by_number.get(number, [])
        match = matches[0] if matches else {}
        fragment_pages = []
        for fragment in match.get("fragments") or []:
            if fragment.get("page_no") is not None:
                fragment_pages.append(fragment["page_no"])
        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "duplicate_output"
        elif len(set(fragment_pages)) > 1:
            status = "matched_multi_page_canonical"
        else:
            status = "matched"
        statuses[status] += 1
        rows.append({
            "table_number": number, "source_caption": occurrences[0]["caption"],
            "source_pages": sorted({item["page_no"] for item in occurrences}),
            "source_occurrence_count": len(occurrences), "output_id": match.get("id", ""),
            "output_title": match.get("title", ""), "output_page_no": _page(match.get("provenance")),
            "fragment_pages": fragment_pages, "fragment_count": len(match.get("fragments") or []),
            "row_count": len(match.get("rows") or []), "footnote_count": len(match.get("footnotes") or []),
            "status": status,
            "details": "One canonical Table with physical fragments" if status == "matched_multi_page_canonical" else "",
        })
    source_numbers = set(source_by_number)
    for table in document.get("tables") or []:
        number = table.get("table_number")
        if not number or str(number) in source_numbers:
            continue
        statuses["output_only_unnumbered"] += 1
        rows.append({
            "table_number": "", "source_caption": "", "source_pages": [], "source_occurrence_count": 0,
            "output_id": table.get("id", ""), "output_title": table.get("title", ""),
            "output_page_no": _page(table.get("provenance")),
            "fragment_pages": [fragment.get("page_no") for fragment in table.get("fragments") or []],
            "fragment_count": len(table.get("fragments") or []), "row_count": len(table.get("rows") or []),
            "footnote_count": len(table.get("footnotes") or []), "status": "output_only_unnumbered",
            "details": "Source contains a section-local tabular block without an explicit TABLE caption.",
        })
    return rows, dict(statuses)


def _equation_rows(inventory: Any, document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in inventory.equations:
        source_by_number[item["number"]].append(item)
    output_by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for equation in document.get("equations") or []:
        number = equation.get("equation_number")
        if number:
            output_by_number[str(number)].append(equation)
    rows: list[dict[str, Any]] = []
    statuses = Counter()
    for number, occurrences in source_by_number.items():
        matches = output_by_number.get(number, [])
        match = matches[0] if matches else {}
        source_vars = sorted({variable for item in occurrences for variable in item.get("variables", [])})
        output_vars = sorted(set(match.get("variables") or []))
        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "duplicate_output"
        elif not set(source_vars).issubset(output_vars):
            status = "matched_variable_coverage_gap"
        else:
            status = "matched"
        statuses[status] += 1
        rows.append({
            "equation_number": number, "source_occurrence_count": len(occurrences),
            "source_pages": sorted({item["page_no"] for item in occurrences}),
            "source_variables": source_vars, "source_text": occurrences[0]["text"],
            "output_id": match.get("id", ""), "output_page_no": _page(match.get("provenance")),
            "output_variables": output_vars, "output_text": match.get("text", ""), "status": status,
            "details": "Deterministic PDF-label fallback" if match.get("id", "").startswith("equation:") else "",
        })
    for equation in document.get("equations") or []:
        if equation.get("equation_number"):
            continue
        statuses["output_only_unnumbered"] += 1
        rows.append({
            "equation_number": "", "source_occurrence_count": 0, "source_pages": [], "source_variables": [],
            "source_text": "", "output_id": equation.get("id", ""), "output_page_no": _page(equation.get("provenance")),
            "output_variables": equation.get("variables") or [], "output_text": equation.get("text", ""),
            "status": "output_only_unnumbered", "details": "Docling formula without an explicit source Equation identifier.",
        })
    return rows, dict(statuses)


def _reference_rows(inventory: Any, document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = inventory.references
    output = document.get("references") or []
    output_keys: Counter[tuple[str, str, int | None]] = Counter(
        (str(item.get("reference_type") or "other"), _norm(item.get("target")), _page(item.get("provenance")))
        for item in output
    )
    source_keys: Counter[tuple[str, str, int | None]] = Counter(
        (str(item.get("reference_type") or "other"), _norm(item.get("target")), item.get("page_no"))
        for item in source
    )
    rows: list[dict[str, Any]] = []
    statuses = Counter()
    for index, item in enumerate(source):
        key = (item["reference_type"], _norm(item["target"]), item["page_no"])
        available = output_keys[key]
        if not available:
            status = "missing"
        elif available > 1:
            status = "duplicate_output"
        else:
            status = "matched"
        statuses[status] += 1
        rows.append({
            "source_index": index, "reference_type": item["reference_type"], "target": item["target"],
            "page_no": item["page_no"], "source_text": item["text"], "output_match_count": available,
            "status": status, "details": "Explicit PDF text cross-reference inventory",
        })
    matched_keys = set(source_keys)
    for index, item in enumerate(output):
        key = (str(item.get("reference_type") or "other"), _norm(item.get("target")), _page(item.get("provenance")))
        if key in matched_keys:
            continue
        statuses["output_only"] += 1
        rows.append({
            "source_index": "", "reference_type": item.get("reference_type", ""), "target": item.get("target", ""),
            "page_no": _page(item.get("provenance")), "source_text": "", "output_match_count": 0,
            "status": "output_only", "details": "Output hyperlink or reference without a matching explicit PDF-text occurrence.",
        })
    return rows, dict(statuses)


def _summary(
    source_pdf: Path,
    inventory: Any,
    document: dict[str, Any],
    section_status: dict[str, int],
    table_status: dict[str, int],
    equation_status: dict[str, int],
    reference_status: dict[str, int],
    exception_records: list[dict[str, Any]],
    content_counts: dict[str, int],
    unmatched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_pages: set[int] = set()
    for collection in (document.get("sections"), document.get("tables"), document.get("equations"), document.get("references")):
        for item in collection or []:
            page_no = _page(item.get("provenance"))
            if page_no is not None:
                output_pages.add(page_no)
    failures = document.get("failures") or []
    failure_codes = Counter(str(failure.get("code")) for failure in failures)
    section_total = len(inventory.section_headings)
    table_total = len(inventory.unique_table_numbers)
    equation_total = len(inventory.unique_equation_numbers)
    reference_total = len(inventory.references)
    exception_output = []
    for section in document.get("sections") or []:
        for block in section.get("blocks") or []:
            if block.get("block_type") == "exception":
                exception_output.append(block)
    exception_matches = sum(
        any(_norm(record["text"][:100]) in _norm(block.get("text")) for block in exception_output)
        for record in exception_records
    )
    matched_content = content_counts.get("matched_source_lines", 0)
    meaningful_content = content_counts.get("meaningful_source_lines", 0)
    return {
        "source": {
            "file": str(source_pdf.resolve()), "pages": inventory.pages,
            "top_level_section_heading_occurrences": len(inventory.top_sections),
            "numbered_subsection_heading_occurrences": len(inventory.numbered_sections),
            "section_heading_occurrences": section_total,
            "unique_table_caption_numbers": table_total,
            "table_caption_occurrences": len(inventory.table_captions),
            "equation_identifier_occurrences": len(inventory.equations),
            "unique_equation_identifiers": equation_total,
            "exception_block_occurrences": len(exception_records),
            "explicit_reference_occurrences": reference_total,
            "explicit_reference_unique_keys": len({(x["reference_type"], x["target"], x["page_no"]) for x in inventory.references}),
        },
        "output": {
            "chapters": len(document.get("chapters") or []), "sections": len(document.get("sections") or []),
            "tables": len(document.get("tables") or []),
            "canonical_numbered_tables": len([x for x in document.get("tables") or [] if x.get("table_number")]),
            "table_fragments": sum(len(x.get("fragments") or []) for x in document.get("tables") or []),
            "table_rows": sum(len(x.get("rows") or []) for x in document.get("tables") or []),
            "references": len(document.get("references") or []), "equations": len(document.get("equations") or []),
            "unparsed_blocks": len(document.get("unparsed_blocks") or []), "failures": len(failures),
        },
        "coverage": {
            "section_heading_recall": (section_status.get("matched", 0) + section_status.get("title_mismatch", 0)) / section_total if section_total else 1.0,
            "table_caption_recall": sum(table_status.get(key, 0) for key in ("matched", "matched_multi_page_canonical")) / table_total if table_total else 1.0,
            "equation_identifier_recall": sum(equation_status.get(key, 0) for key in ("matched", "matched_variable_coverage_gap")) / equation_total if equation_total else 1.0,
            "reference_occurrence_match_rate": (reference_status.get("matched", 0) + reference_status.get("duplicate_output", 0)) / reference_total if reference_total else 1.0,
            "reference_duplicate_output_rate": reference_status.get("duplicate_output", 0) / reference_total if reference_total else 0.0,
            "exception_block_match_rate": exception_matches / len(exception_records) if exception_records else 1.0,
            "meaningful_source_text_line_coverage": matched_content / meaningful_content if meaningful_content else 1.0,
            "source_pages_with_structural_output": len(output_pages),
            "source_page_coverage": len(output_pages) / inventory.pages if inventory.pages else 1.0,
            "duplicate_object_count": sum(status.get("duplicate_output", 0) for status in (section_status, table_status, equation_status, reference_status)),
            "split_object_count": section_status.get("duplicate_output", 0) + table_status.get("duplicate_output", 0) + equation_status.get("duplicate_output", 0),
            "stitched_multi_page_table_count": table_status.get("matched_multi_page_canonical", 0),
            "ambiguous_object_count": sum(1 for row in unmatched_rows if row.get("status") == "ambiguous"),
            "unmatched_content_record_count": len(unmatched_rows),
        },
        "status_counts": {
            "sections": section_status, "tables": table_status, "equations": equation_status,
            "references": reference_status,
        },
        "exceptions": {"source_blocks": len(exception_records), "output_exception_blocks": len(exception_output), "matched_blocks": exception_matches},
        "content_accounting": content_counts,
        "failures_by_code": dict(failure_codes),
        "failures_by_severity": dict(Counter(str(failure.get("severity")) for failure in failures)),
        "source_inventory_method": "Independent PyMuPDF page text plus line font-size/style and deterministic regex patterns; Docling output is used only for the compared output side.",
    }


def audit(source_pdf: str | Path, output_dir: str | Path) -> dict[str, Path]:
    source = Path(source_pdf).resolve()
    output = Path(output_dir).resolve()
    document_path = output / "document.json"
    document = json.loads(document_path.read_text(encoding="utf-8"))
    inventory = build_source_inventory(source)
    audit_dir = output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    section_rows, section_status = _section_rows(inventory, document)
    table_rows, table_status = _table_rows(inventory, document)
    equation_rows, equation_status = _equation_rows(inventory, document)
    reference_rows, reference_status = _reference_rows(inventory, document)
    unmatched_rows, content_counts = _source_text_audit(inventory, document)
    exception_records = _exception_blocks(inventory)
    output_exception_blocks = [
        block for section in document.get("sections") or [] for block in section.get("blocks") or []
        if block.get("block_type") == "exception"
    ]
    for record in exception_records:
        matched = next((block for block in output_exception_blocks if _norm(record["text"][:100]) in _norm(block.get("text"))), None)
        if not matched:
            unmatched_rows.append({
                "record_type": "source_exception", "status": "unmatched_exception_block", "category": "exception",
                "page_no": record["page_no"], "end_page_no": record["end_page_no"], "identifier": f"exception:{record['index']}",
                "text": record["text"], "reason": "Exception marker/block has no matching output ExceptionBlock.",
            })
    summary = _summary(source, inventory, document, section_status, table_status, equation_status, reference_status, exception_records, content_counts, unmatched_rows)

    _write_csv(audit_dir / "section_coverage.csv", section_rows, [
        "number", "source_kind", "source_title", "source_page_no", "source_bbox", "output_id", "output_title",
        "output_page_no", "output_level", "status", "details",
    ])
    _write_csv(audit_dir / "table_coverage.csv", table_rows, [
        "table_number", "source_caption", "source_pages", "source_occurrence_count", "output_id", "output_title",
        "output_page_no", "fragment_pages", "fragment_count", "row_count", "footnote_count", "status", "details",
    ])
    _write_csv(audit_dir / "equation_coverage.csv", equation_rows, [
        "equation_number", "source_occurrence_count", "source_pages", "source_variables", "source_text", "output_id",
        "output_page_no", "output_variables", "output_text", "status", "details",
    ])
    _write_csv(audit_dir / "reference_coverage.csv", reference_rows, [
        "source_index", "reference_type", "target", "page_no", "source_text", "output_match_count", "status", "details",
    ])
    _write_csv(audit_dir / "unmatched_content.csv", unmatched_rows, [
        "record_type", "status", "category", "page_no", "end_page_no", "identifier", "text", "reason",
    ])
    # Reconcile source provenance before finalizing audit metrics.  The stage
    # is intentionally local to auditing so the structural parser and all
    # downstream semantic components remain unchanged.
    from .provenance_reconciliation import reconcile_document
    reconciliation_metrics = reconcile_document(source, document, audit_dir)
    summary["provenance_reconciliation"] = reconciliation_metrics
    summary_path = audit_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "section_coverage": audit_dir / "section_coverage.csv", "table_coverage": audit_dir / "table_coverage.csv",
        "equation_coverage": audit_dir / "equation_coverage.csv", "reference_coverage": audit_dir / "reference_coverage.csv",
        "unmatched_content": audit_dir / "unmatched_content.csv", "audit_summary": summary_path,
        "provenance_reconciliation": audit_dir / "provenance_reconciliation.csv",
        "provenance_spans": audit_dir / "provenance_spans.csv",
        "provenance_audit_summary": audit_dir / "provenance_audit_summary.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit v2 structural output against HVAC-Codes.pdf")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = audit(args.source, args.output_dir)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
