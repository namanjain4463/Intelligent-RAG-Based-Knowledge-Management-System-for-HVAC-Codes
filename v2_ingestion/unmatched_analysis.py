"""Classify source lines that the structural audit could not line-match.

This is an audit companion to :mod:`v2_ingestion.audit`.  It does not change
the structural parser or the generated document.  The input is the existing
``unmatched_content.csv`` plus the source PDF and ``document.json``.  The
classification is deliberately conservative: a line is a blocker when it
looks like uncaptured normative code text; a line that belongs to an existing
canonical object but has unresolved exact-line provenance is reported
separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz

from .audit import _all_block_texts, _classify_source_line, _positive_navigation_evidence
from .source_inventory import (
    EXCEPTION_RE,
    NUMBERED_SECTION_RE,
    TABLE_CAPTION_RE,
    TOP_SECTION_RE,
    build_source_inventory,
    normalize_text,
    strip_source_prefix,
)


CATEGORIES = (
    "non-normative source artifact",
    "header/footer/navigation",
    "duplicated text",
    "commentary/user note",
    "malformed table residue",
    "normative code text",
    "unresolved provenance",
    "genuinely unparseable content",
)

_SECTION_NUMBER_RE = re.compile(
    r"^(?:\[[A-Z]+\]\s*)?(?:Section\s+)?(\d{3,4}(?:\.\d+)+|\d{3,4})(?=\s|\.|$)",
    re.IGNORECASE,
)
_CHAPTER_HEADING_RE = re.compile(r"^Chapter\s+(\d{1,2})\s+(.+)$", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(
    r"^(?:\[[A-Z]+\]\s*)?(?:Section\s+)?\d{3,4}(?:\.\d+)*(?:\s|\.)",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}[.]?$")
_NAVIGATION_RE = re.compile(
    r"(?:^|\b)(?:contents|table of contents|index|go to|previous|next|page\s+\d+)\b",
    re.IGNORECASE,
)
_FOOTNOTE_RE = re.compile(r"^(?:For\s+SI:|[a-z]\.)", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _norm(value: str) -> str:
    return normalize_text(value).casefold().rstrip(".")


def _page(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    keys = ("left", "top", "right", "bottom")
    if not all(key in value for key in keys):
        return None
    try:
        return {key: float(value[key]) for key in keys}
    except (TypeError, ValueError):
        return None


def _bbox_intersects(left: dict[str, float], right: dict[str, float]) -> bool:
    return not (
        left["right"] < right["left"]
        or right["right"] < left["left"]
        or left["bottom"] < right["top"]
        or right["bottom"] < left["top"]
    )


def _bbox_key(value: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(value, "bbox", None) if value is not None else None
    if not isinstance(bbox, dict):
        return None
    try:
        return tuple(round(float(bbox[key]), 3) for key in ("left", "top", "right", "bottom"))
    except (KeyError, TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                serialized[key] = value
            writer.writerow(serialized)


def _load_source_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    source_rows = [
        row for row in rows
        if row.get("record_type") == "source_line" and row.get("status") == "unmatched"
    ]
    output_rows = [row for row in rows if row.get("record_type") == "output_block"]
    return source_rows, output_rows


def _table_regions(document: dict[str, Any]) -> dict[int, list[tuple[str, dict[str, float]]]]:
    regions: dict[int, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for table in document.get("tables") or []:
        table_id = str(table.get("id") or "")
        for fragment in table.get("fragments") or []:
            page_no = _page(fragment.get("page_no"))
            bbox = _bbox((fragment.get("provenance") or {}).get("bbox"))
            if page_no is not None and bbox is not None:
                regions[page_no].append((table_id, bbox))
    return regions


def _output_page_set(document: dict[str, Any]) -> set[int]:
    pages: set[int] = set()

    def add(item: dict[str, Any]) -> None:
        page_no = _page((item.get("provenance") or {}).get("page_no"))
        if page_no is not None:
            pages.add(page_no)

    for collection in ("chapters", "sections", "tables", "equations", "references", "unparsed_blocks"):
        for item in document.get(collection) or []:
            add(item)
            if collection == "tables":
                for fragment in item.get("fragments") or []:
                    add(fragment)
    return pages


def _section_and_chapter_ids(document: dict[str, Any]) -> tuple[set[str], set[str]]:
    section_ids = {str(item.get("number")) for item in document.get("sections") or [] if item.get("number")}
    chapter_ids = {str(item.get("number")) for item in document.get("chapters") or [] if item.get("number")}
    return section_ids, chapter_ids


def _candidate_lines(inventory: Any) -> dict[tuple[int, str], list[Any]]:
    lookup: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for line in inventory.lines:
        lookup[(line.page_no, _norm(line.text))].append(line)
    return lookup


def _choose_candidate(candidates: list[Any], regions: dict[int, list[tuple[str, dict[str, float]]]]) -> Any | None:
    if not candidates:
        return None
    for candidate in candidates:
        for _, region in regions.get(candidate.page_no, []):
            if _bbox_intersects(candidate.bbox, region):
                return candidate
    return candidates[0]


def _is_in_table(candidate: Any | None, regions: dict[int, list[tuple[str, dict[str, float]]]]) -> list[str]:
    if candidate is None:
        return []
    return [table_id for table_id, region in regions.get(candidate.page_no, []) if _bbox_intersects(candidate.bbox, region)]


def _table_page_set(regions: dict[int, list[tuple[str, dict[str, float]]]]) -> set[int]:
    return set(regions)


def _is_structural_heading(text: str, section_ids: set[str], chapter_ids: set[str]) -> tuple[str, str] | None:
    value = strip_source_prefix(text)
    chapter = _CHAPTER_HEADING_RE.match(value)
    if chapter and chapter.group(1) in chapter_ids:
        return "chapter", chapter.group(1)
    if not _SECTION_HEADING_RE.match(value):
        return None
    match = _SECTION_NUMBER_RE.match(value)
    if match and match.group(1) in section_ids:
        return "section", match.group(1)
    return None


def _looks_unparseable(text: str) -> bool:
    if _CONTROL_RE.search(text):
        return True
    if any(ord(char) == 0xFFFD for char in text):
        return True
    # A line made mostly of extraction noise is not safely interpretable as
    # either a code provision or a table cell.  Normal punctuation and units
    # are intentionally not treated as noise.
    compact = text.replace(" ", "")
    if compact and sum(char.isalnum() for char in compact) / len(compact) < 0.12 and len(compact) >= 10:
        return True
    return False


def _classify(
    row: dict[str, str],
    candidate: Any | None,
    table_ids: list[str],
    page_table_exists: bool,
    page_has_output: bool,
    same_page_text_count: int,
    same_geometry_count: int,
    section_ids: set[str],
    chapter_ids: set[str],
    legacy_category: str,
    page_height: float = 792.0,
) -> tuple[str, str, bool, bool, dict[str, Any]]:
    text = normalize_text(row.get("text", ""))
    upper = text.upper()
    structural = _is_structural_heading(text, section_ids, chapter_ids)

    if _positive_navigation_evidence(text, candidate, page_height):
        return "header/footer/navigation", "PDF navigation or page furniture", False, False, {}
    if upper.startswith(("SOURCE:", "HTTPS://", "HTTP://")):
        return "non-normative source artifact", "Embedded source URL or source-note artifact", False, False, {}
    if (
        upper.startswith(("INSIGHTS", "USER NOTE", "COMMENTARY", "INFORMATIONAL"))
        or upper.startswith("ABOUT THIS CHAPTER")
        or "ABOUT THIS CHAPTER:" in upper
    ):
        return "commentary/user note", "Editorial insight or user-note material", False, False, {}
    if structural:
        kind, identifier = structural
        return (
            "unresolved provenance",
            f"{kind.title()} heading has canonical id {identifier}, but exact source-line provenance was not matched",
            False,
            True,
            {"canonical_object_id": f"{kind}:{identifier}"},
        )
    if table_ids or (page_table_exists and _FOOTNOTE_RE.match(strip_source_prefix(text))):
        return (
            "malformed table residue",
            "Source line lies inside a physical table fragment or is a table-footnote candidate",
            False,
            True,
            {"table_ids": table_ids},
        )
    if same_page_text_count > 1 and same_geometry_count > 1 and candidate is not None:
        # Repeated prose in two different code provisions is legitimate.  A
        # duplicate is only reported here when the line has identical source
        # geometry; table repetition and repeated INSIGHTS markers were handled
        # above.
        same_geometry = getattr(candidate, "bbox", None)
        if same_geometry is not None:
            return "duplicated text", "Identical source text was emitted at the same source geometry", False, True, {}
    if _looks_unparseable(text):
        return "genuinely unparseable content", "Text layer contains extraction noise that cannot be safely interpreted", False, True, {}
    # This is intentionally the final branch.  Any line that is not a known
    # artifact, commentary, table residue, duplicate, or mapped structural
    # heading is potentially normative code text and therefore blocks release.
    return "normative code text", "Meaningful source text has no canonical structural mapping", True, True, {}


def _percentage(value: int | float, denominator: int | float) -> float:
    return round((100.0 * value / denominator), 6) if denominator else 0.0


def analyze_unmatched_lines(
    source_pdf: Path,
    document_path: Path,
    unmatched_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    document = json.loads(document_path.read_text(encoding="utf-8"))
    source_rows, output_rows = _load_source_rows(unmatched_csv)
    inventory = build_source_inventory(source_pdf)
    regions = _table_regions(document)
    output_pages = _output_page_set(document)
    output_texts = _all_block_texts(document)
    output_joined = {page: " ".join(values).casefold() for page, values in output_texts.items()}
    section_ids, chapter_ids = _section_and_chapter_ids(document)
    candidates_by_key = _candidate_lines(inventory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count exact same-page source text.  This is only evidence for duplicate
    # extraction when it also occupies the same geometry; repeated provisions
    # in different sections are not duplicates.
    page_text_counts = Counter((line.page_no, _norm(line.text)) for line in inventory.lines)
    meaningful_by_page = Counter()
    with fitz.open(source_pdf) as pdf:
        page_heights = {page_no: float(page.rect.height) for page_no, page in enumerate(pdf, start=1)}
    for line in inventory.lines:
        legacy_category = _classify_source_line(line.text, inventory, line.page_no)
        if len(_norm(line.text)) >= 5 and not _positive_navigation_evidence(line.text, line, page_heights.get(line.page_no, 792.0)):
            meaningful_by_page[line.page_no] += 1

    classified: list[dict[str, Any]] = []
    for row in source_rows:
        page_no = _page(row.get("page_no"))
        text = normalize_text(row.get("text", ""))
        candidates = candidates_by_key.get((page_no or 0, _norm(text)), [])
        candidate = _choose_candidate(candidates, regions)
        candidate_geometry = _bbox_key(candidate)
        same_geometry_count = sum(1 for item in candidates if _bbox_key(item) == candidate_geometry) if candidate_geometry else 0
        table_ids = _is_in_table(candidate, regions)
        legacy_category = row.get("category", "")
        category, reason, blocker, review_required, evidence = _classify(
            row=row,
            candidate=candidate,
            table_ids=table_ids,
            page_table_exists=bool(regions.get(page_no or 0)),
            page_has_output=(page_no in output_pages if page_no is not None else False),
            same_page_text_count=page_text_counts[(page_no or 0, _norm(text))],
            same_geometry_count=same_geometry_count,
            section_ids=section_ids,
            chapter_ids=chapter_ids,
            legacy_category=legacy_category,
            page_height=page_heights.get(page_no or 0, 792.0),
        )
        bbox = getattr(candidate, "bbox", None) if candidate is not None else None
        classified.append({
            "page_no": page_no or "",
            "end_page_no": _page(row.get("end_page_no")) or page_no or "",
            "text": text,
            "category": category,
            "legacy_audit_category": legacy_category,
            "classification_reason": reason,
            "blocker": "true" if blocker else "false",
            "review_required": "true" if review_required else "false",
            "candidate_count": len(candidates),
            "same_page_text_count": page_text_counts[(page_no or 0, _norm(text))],
            "table_ids": table_ids,
            "canonical_object_ids": evidence.get("canonical_object_id", ""),
            "source_bbox": bbox or {},
            "source_font_size": getattr(candidate, "max_font_size", "") if candidate is not None else "",
            "source_flags": list(getattr(candidate, "flags", ())) if candidate is not None else [],
        })

    # The historical line audit excludes navigation-like lines from its
    # meaningful denominator. Re-check those excluded lines independently so
    # a false navigation match cannot hide normative source text.
    excluded: list[dict[str, Any]] = []
    excluded_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    source_row_keys = {
        (_page(row.get("page_no")), _norm(row.get("text", "")))
        for row in source_rows
    }
    for line in inventory.lines:
        text = normalize_text(line.text)
        legacy_category = _classify_source_line(text, inventory, line.page_no)
        candidates = candidates_by_key.get((line.page_no, _norm(text)), [])
        candidate = _choose_candidate(candidates, regions)
        short_list_marker = (
            _PAGE_NUMBER_RE.fullmatch(text)
            and len(_norm(text)) < 5
            and candidate is not None
            and candidate.bbox["top"] > 50
        )
        legacy_navigation_exclusion = (
            len(_norm(text)) >= 5
            and legacy_category == "navigation_artifact"
            and (line.page_no, _norm(text)) not in source_row_keys
        )
        if not (short_list_marker or legacy_navigation_exclusion):
            continue
        matched = bool(text and _norm(text) in output_joined.get(line.page_no, ""))
        if matched:
            continue
        candidate = _choose_candidate(candidates_by_key.get((line.page_no, _norm(text)), []), regions)
        # Pure list markers in the body are structural list syntax, not page
        # navigation. Any legacy exclusion that lacks positive page-margin
        # navigation evidence is retained as normative source text rather than
        # being rescued by a phrase-specific exception.
        if _PAGE_NUMBER_RE.fullmatch(text) and candidate is not None and candidate.bbox["top"] > 50:
            category = "unresolved provenance"
            reason = "Numeric list marker was excluded by the legacy meaningful-line threshold; its list-item provenance is unresolved"
            blocker = False
        elif not _positive_navigation_evidence(text, candidate, page_heights.get(line.page_no, 792.0)):
            category = "normative code text"
            reason = "Legacy navigation exclusion lacked positive page-margin navigation evidence"
            blocker = True
        else:
            category = "header/footer/navigation"
            reason = "Navigation-like source line excluded by the legacy meaningful-line denominator"
            blocker = False
        record = {
            "page_no": line.page_no,
            "text": text,
            "category": category,
            "legacy_audit_category": legacy_category,
            "classification_reason": reason,
            "blocker": "true" if blocker else "false",
            "source_bbox": line.bbox,
            "source_font_size": line.max_font_size,
            "source_flags": list(line.flags),
        }
        excluded.append(record)
        excluded_by_page[line.page_no].append(record)

    _write_csv(
        output_dir / "unmatched_excluded_source_lines.csv",
        excluded,
        [
            "page_no", "text", "category", "legacy_audit_category",
            "classification_reason", "blocker", "source_bbox", "source_font_size", "source_flags",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    line_fields = [
        "page_no", "end_page_no", "text", "category", "legacy_audit_category",
        "classification_reason", "blocker", "review_required", "candidate_count",
        "same_page_text_count", "table_ids", "canonical_object_ids", "source_bbox",
        "source_font_size", "source_flags",
    ]
    _write_csv(output_dir / "unmatched_line_classification.csv", classified, line_fields)

    total_unmatched = len(classified)
    total_meaningful = sum(meaningful_by_page.values())
    category_rows: list[dict[str, Any]] = []
    by_category = Counter(row["category"] for row in classified)
    excluded_by_category = Counter(row["category"] for row in excluded)
    by_category_blocker = Counter(row["category"] for row in classified if row["blocker"] == "true")
    by_category_review = Counter(row["category"] for row in classified if row["review_required"] == "true")
    by_category_pages: dict[str, set[int]] = defaultdict(set)
    for row in classified:
        if row["page_no"] != "":
            by_category_pages[row["category"]].add(int(row["page_no"]))
    for category in CATEGORIES:
        count = by_category[category]
        pages = sorted(by_category_pages[category])
        category_rows.append({
            "category": category,
            "line_count": count,
            "excluded_line_count": excluded_by_category[category],
            "all_unmatched_source_line_count": count + excluded_by_category[category],
            "percent_of_unmatched_lines": _percentage(count, total_unmatched),
            "percent_of_meaningful_source_lines": _percentage(count, total_meaningful),
            "percent_of_all_unmatched_source_lines": _percentage(count + excluded_by_category[category], total_unmatched + len(excluded)),
            "blocker_count": by_category_blocker[category],
            "excluded_blocker_count": sum(item["blocker"] == "true" for item in excluded if item["category"] == category),
            "review_required_count": by_category_review[category],
            "pages_affected": len(pages),
            "first_page": pages[0] if pages else "",
            "last_page": pages[-1] if pages else "",
        })
    _write_csv(
        output_dir / "unmatched_category_summary.csv",
        category_rows,
        [
            "category", "line_count", "percent_of_unmatched_lines",
            "percent_of_meaningful_source_lines", "excluded_line_count",
            "all_unmatched_source_line_count", "percent_of_all_unmatched_source_lines",
            "blocker_count", "excluded_blocker_count", "review_required_count",
            "pages_affected", "first_page", "last_page",
        ],
    )

    slug = {category: re.sub(r"[^a-z0-9]+", "_", category).strip("_") for category in CATEGORIES}
    page_rows: list[dict[str, Any]] = []
    classified_by_page = defaultdict(list)
    for row in classified:
        classified_by_page[int(row["page_no"])].append(row)
    for page_no in range(1, inventory.pages + 1):
        page_items = classified_by_page[page_no]
        row: dict[str, Any] = {
            "page_no": page_no,
            "meaningful_source_line_count": meaningful_by_page[page_no],
            "unmatched_line_count": len(page_items),
            "percent_of_all_unmatched_lines": _percentage(len(page_items), total_unmatched),
            "percent_of_page_meaningful_lines": _percentage(len(page_items), meaningful_by_page[page_no]),
            "blocker_count": sum(item["blocker"] == "true" for item in page_items),
            "review_required_count": sum(item["review_required"] == "true" for item in page_items),
            "legacy_excluded_unmatched_line_count": len(excluded_by_page[page_no]),
            "legacy_excluded_blocker_count": sum(item["blocker"] == "true" for item in excluded_by_page[page_no]),
            "all_unmatched_source_line_count": len(page_items) + len(excluded_by_page[page_no]),
            "all_blocker_count": sum(item["blocker"] == "true" for item in page_items) + sum(item["blocker"] == "true" for item in excluded_by_page[page_no]),
        }
        for category in CATEGORIES:
            items = [item for item in page_items if item["category"] == category]
            row[f"{slug[category]}_count"] = len(items)
            row[f"{slug[category]}_percent_of_page_meaningful"] = _percentage(len(items), meaningful_by_page[page_no])
        page_rows.append(row)
    page_fields = [
        "page_no", "meaningful_source_line_count", "unmatched_line_count",
        "percent_of_all_unmatched_lines", "percent_of_page_meaningful_lines",
        "blocker_count", "review_required_count", "legacy_excluded_unmatched_line_count",
        "legacy_excluded_blocker_count", "all_unmatched_source_line_count", "all_blocker_count",
    ]
    for category in CATEGORIES:
        page_fields.extend([f"{slug[category]}_count", f"{slug[category]}_percent_of_page_meaningful"])
    _write_csv(output_dir / "unmatched_page_summary.csv", page_rows, page_fields)

    blockers = [row for row in classified if row["blocker"] == "true"]
    excluded_blockers = [row for row in excluded if row["blocker"] == "true"]
    blocker_pages = sorted({int(row["page_no"]) for row in blockers})
    effective_blocker_pages = sorted({int(row["page_no"]) for row in [*blockers, *excluded_blockers]})
    output_unparsed_by_page = Counter(_page(row.get("page_no")) for row in output_rows)
    output_unparsed_by_page.pop(None, None)
    output_unparsed_by_reason = Counter(row.get("text", "") for row in output_rows)
    analysis = {
        "status": "BLOCKED" if blockers else "PASS",
        "blocker_rule": "Any unmatched source line classified as normative code text blocks release.",
        "source_file": str(source_pdf.resolve()),
        "source_pages": inventory.pages,
        "meaningful_source_lines": total_meaningful,
        "unmatched_source_lines": total_unmatched,
        "unmatched_meaningful_line_percent": round(100.0 * total_unmatched / total_meaningful, 6) if total_meaningful else 0.0,
        "legacy_excluded_unmatched_source_lines": len(excluded),
        "all_unmatched_source_lines_including_legacy_exclusions": total_unmatched + len(excluded),
        "classified_line_count_check": sum(by_category.values()),
        "classification_count_check_passed": sum(by_category.values()) == total_unmatched,
        "blocker_line_count": len(blockers),
        "blocker_percent_of_unmatched_lines": _percentage(len(blockers), total_unmatched),
        "blocker_percent_of_meaningful_source_lines": _percentage(len(blockers), total_meaningful),
        "additional_excluded_normative_blocker_count": len(excluded_blockers),
        "effective_blocker_line_count_including_legacy_exclusions": len(blockers) + len(excluded_blockers),
        "effective_blocker_percent_of_meaningful_source_lines": _percentage(len(blockers) + len(excluded_blockers), total_meaningful),
        "effective_blocker_percent_of_all_unmatched_source_lines": _percentage(len(blockers) + len(excluded_blockers), total_unmatched + len(excluded)),
        "blocker_pages": blocker_pages,
        "effective_blocker_pages_including_legacy_exclusions": effective_blocker_pages,
        "review_required_line_count": sum(row["review_required"] == "true" for row in classified),
        "pages_with_unmatched_lines": len({int(row["page_no"]) for row in classified}),
        "categories": {row["category"]: row for row in category_rows},
        "classification_rules": {
            "unresolved_provenance": "A source heading whose canonical section/chapter ID exists, but whose exact PDF line is not present in page-level output text.",
            "malformed_table_residue": "A source line whose PDF geometry intersects a physical table fragment, including table footnote candidates.",
            "duplicated_text": "Only exact same-page source text emitted at the same source geometry; repeated regulatory language is not called duplicate.",
            "normative_code_text": "Conservative final category for meaningful source text not accounted for by the other categories.",
            "legacy_excluded_source_lines": "The main 20.50% figure retains the existing audit denominator; excluded lines are reported separately so false navigation classification cannot hide content.",
        },
        "supplementary_output_unparsed_blocks": {
            "count": len(output_rows),
            "pages": dict(sorted(output_unparsed_by_page.items())),
            "note": "These are existing output UnparsedBlock records, not source-line classifications; they remain preserved for separate parser review.",
        },
        "known_limitations": [
            "Line-level matching is stricter than object-level inventory matching; canonical heading IDs are therefore reported as unresolved provenance rather than data loss.",
            "Table residue is identified from source PDF coordinates and current physical fragments; it does not assert that every table cell is semantically correct.",
        ],
    }
    (output_dir / "unmatched_line_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--unmatched-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze_unmatched_lines(args.source_pdf, args.document, args.unmatched_csv, args.output_dir)
    print(json.dumps({
        "status": analysis["status"],
        "meaningful_source_lines": analysis["meaningful_source_lines"],
        "unmatched_source_lines": analysis["unmatched_source_lines"],
        "blocker_line_count": analysis["blocker_line_count"],
        "blocker_pages": analysis["blocker_pages"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
