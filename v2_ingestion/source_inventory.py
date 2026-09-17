"""Independent, deterministic inventories from the PDF text layer.

This module is used by the audit and only for conservative structural
fallbacks (equation labels, styled headings, table captions, and provenance).
It does not perform semantic entity or requirement extraction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz


TOP_SECTION_RE = re.compile(r"^Section\s+(\d{3,4})\s+(.+?)\s*$", re.IGNORECASE)
NUMBERED_SECTION_RE = re.compile(r"^(\d{3,4}(?:\.\d+)+)\s+(.+?)\s*$")
TABLE_CAPTION_RE = re.compile(
    r"^(?:\[[A-Z]+\]\s*)?TABLE\s+(\d+(?:\.\d+)*(?:\(\d+\))?)(?=\s|$)\s*(.*)$"
)
EQUATION_RE = re.compile(r"\b(?:Equation|Eq\.?)\s*(\d+)\s*[-–]\s*(\d+)", re.IGNORECASE)
EXCEPTION_RE = re.compile(r"^Exceptions?\s*:\s*(.*)$", re.IGNORECASE)
SECTION_REF_RE = re.compile(r"\bSections?\s+(\d{3,4}(?:\.\d+)*(?:\(\d+\))?(?:\s+(?:through|to|and)\s+\d{3,4}(?:\.\d+)*)?)", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\bTables?\s+(\d{3,4}(?:\.\d+)*(?:\(\d+\))?(?:\s+(?:through|to|and)\s+\d{3,4}(?:\.\d+)*)?)", re.IGNORECASE)
CHAPTER_REF_RE = re.compile(r"\bChapters?\s+(\d{1,2}(?:\s+(?:through|to|and)\s+\d{1,2})?)", re.IGNORECASE)
STANDARD_RE = re.compile(
    r"\b(?:ASHRAE|NFPA|ANSI|ASTM|UL|NSF|IMC|IECC|ICC|CSA|ARI|AHRI|AGA|ISO|SMACNA|OSHA|ACGIH)"
    r"(?:\s+[A-Z0-9][A-Z0-9./-]*)?",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def strip_source_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:\[[A-Z]+\]\s*)?(?:➡\s*)*", "", normalize_text(text))


def _formula_variable_tokens(text: str) -> set[str]:
    match = EQUATION_RE.search(text)
    if not match:
        return set()
    tail = text[match.end():].split("where:", 1)[0]
    variables: set[str] = set()
    for token in re.findall(r"[A-Z][A-Za-z0-9]*", tail):
        variables.update(re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z][a-z]|$))", token))
    return variables


@dataclass(frozen=True)
class PdfLine:
    page_no: int
    text: str
    bbox: dict[str, float]
    max_font_size: float
    flags: tuple[int, ...]


@dataclass
class SourceInventory:
    source_file: str
    pages: int
    lines: list[PdfLine] = field(default_factory=list)
    top_sections: list[dict[str, Any]] = field(default_factory=list)
    numbered_sections: list[dict[str, Any]] = field(default_factory=list)
    table_captions: list[dict[str, Any]] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)

    @property
    def section_headings(self) -> list[dict[str, Any]]:
        return [*self.top_sections, *self.numbered_sections]

    @property
    def unique_table_numbers(self) -> list[str]:
        return sorted({row["number"] for row in self.table_captions}, key=_natural_number)

    @property
    def unique_equation_numbers(self) -> list[str]:
        return sorted({row["number"] for row in self.equations}, key=_natural_number)

    def table_caption_before(self, page_no: int | None, top: float | None = None) -> dict[str, Any] | None:
        candidates = []
        for row in self.table_captions:
            if page_no is None or row["page_no"] < page_no:
                candidates.append(row)
            elif row["page_no"] == page_no and top is not None and row["bbox"]["top"] <= top:
                candidates.append(row)
        return candidates[-1] if candidates else None

    def table_caption_after(self, page_no: int | None) -> dict[str, Any] | None:
        if page_no is None:
            return None
        rows = [row for row in self.table_captions if row["page_no"] > page_no]
        return rows[0] if rows else None

    def footnote_lines_for_pages(self, start_page: int, end_page: int) -> list[PdfLine]:
        """Return likely table footnote lines separately from table rows."""
        rows: list[PdfLine] = []
        for line in self.lines:
            if not start_page <= line.page_no <= end_page:
                continue
            stripped = strip_source_prefix(line.text)
            if re.match(r"^(?:For SI:|[a-z]\.)", stripped, re.IGNORECASE):
                rows.append(line)
        return rows

    def exception_blocks(self) -> list[dict[str, Any]]:
        """Group each Exception marker through the next structural heading.

        The PDF text layer is used here because Docling occasionally returns an
        Exception paragraph without its numbered child items or with the tail
        of a long item truncated.
        """
        records: list[dict[str, Any]] = []
        for index, line in enumerate(self.lines):
            marker = EXCEPTION_RE.match(strip_source_prefix(line.text))
            if not marker:
                continue
            collected = [line]
            for following in self.lines[index + 1:]:
                text = strip_source_prefix(following.text)
                if (TOP_SECTION_RE.match(text) or NUMBERED_SECTION_RE.match(text)
                        or TABLE_CAPTION_RE.match(text) or EXCEPTION_RE.match(text)
                        or text.upper().startswith(("INSIGHTS", "USER NOTE"))):
                    break
                collected.append(following)
            items: list[dict[str, Any]] = []
            current_number: str | None = None
            current_lines: list[PdfLine] = []
            for child in collected[1:]:
                child_text = strip_source_prefix(child.text)
                number_match = re.match(r"^(\d+)\.?$", child_text)
                if number_match:
                    if current_lines:
                        items.append({
                            "number": current_number, "text": normalize_text(" ".join(item.text for item in current_lines)),
                            "page_no": current_lines[0].page_no, "bbox": current_lines[0].bbox,
                        })
                    current_number = number_match.group(1)
                    current_lines = []
                    continue
                if current_number is not None:
                    current_lines.append(child)
            if current_lines:
                items.append({
                    "number": current_number, "text": normalize_text(" ".join(item.text for item in current_lines)),
                    "page_no": current_lines[0].page_no, "bbox": current_lines[0].bbox,
                })
            records.append({
                "index": len(records), "page_no": line.page_no,
                "end_page_no": collected[-1].page_no, "text": normalize_text(" ".join(item.text for item in collected)),
                "bbox": line.bbox, "items": items,
            })
        return records


def _natural_number(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts)


def _bbox(line: dict[str, Any]) -> dict[str, float]:
    values = line.get("bbox") or [0, 0, 0, 0]
    return {"left": float(values[0]), "top": float(values[1]), "right": float(values[2]), "bottom": float(values[3])}


def _line_records(source_pdf: Path) -> tuple[list[PdfLine], dict[int, str]]:
    lines: list[PdfLine] = []
    page_texts: dict[int, str] = {}
    with fitz.open(source_pdf) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            page_texts[page_no] = page.get_text("text", sort=False)
            for block in page.get_text("dict", sort=False).get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = normalize_text("".join(span.get("text", "") for span in line.get("spans", [])))
                    if not text:
                        continue
                    spans = line.get("spans", [])
                    sizes = [float(span.get("size", 0)) for span in spans]
                    flags = tuple(sorted({int(span.get("flags", 0)) for span in spans}))
                    lines.append(PdfLine(page_no, text, _bbox(line), max(sizes or [0]), flags))
    return lines, page_texts


def build_source_inventory(source_pdf: str | Path) -> SourceInventory:
    source = Path(source_pdf).resolve()
    lines, page_texts = _line_records(source)
    inventory = SourceInventory(str(source), len(page_texts), lines=lines)
    seen_heading: set[tuple[str, str]] = set()
    seen_caption: set[tuple[str, str, int]] = set()
    for line in lines:
        text = strip_source_prefix(line.text)
        top_match = TOP_SECTION_RE.match(text)
        if top_match and line.max_font_size >= 16:
            key = (top_match.group(1), normalize_text(top_match.group(2)).rstrip("."))
            if key not in seen_heading:
                seen_heading.add(key)
                inventory.top_sections.append({
                    "number": top_match.group(1), "title": key[1], "page_no": line.page_no,
                    "bbox": line.bbox, "text": text, "kind": "top_level_section",
                })
        section_match = NUMBERED_SECTION_RE.match(text)
        if section_match and line.max_font_size >= 13 and (20 in line.flags or line.max_font_size >= 16):
            key = (section_match.group(1), normalize_text(section_match.group(2)).rstrip("."))
            if key not in seen_heading:
                seen_heading.add(key)
                inventory.numbered_sections.append({
                    "number": section_match.group(1), "title": key[1], "page_no": line.page_no,
                    "bbox": line.bbox, "text": text, "kind": "numbered_section",
                })
        caption_match = TABLE_CAPTION_RE.match(text)
        if caption_match and line.max_font_size >= 12:
            number = caption_match.group(1)
            key = (number, normalize_text(text), line.page_no)
            if key not in seen_caption:
                seen_caption.add(key)
                inventory.table_captions.append({
                    "number": number, "caption": text, "page_no": line.page_no,
                    "bbox": line.bbox,
                })
        exception_match = EXCEPTION_RE.match(text)
        if exception_match:
            inventory.exceptions.append({
                "page_no": line.page_no, "text": text, "bbox": line.bbox,
            })

    for page_no, page_text in page_texts.items():
        for match in EQUATION_RE.finditer(page_text):
            equation_number = f"{match.group(1)}-{match.group(2)}"
            candidates = [candidate for candidate in lines if candidate.page_no == page_no and match.group(0).lower() in candidate.text.lower()]
            line = next((candidate for candidate in candidates if "=" in candidate.text), None) or (candidates[0] if candidates else None)
            raw_context = (
                page_text[match.start(): match.end() + 1400]
                + " " + page_texts.get(page_no + 1, "")[:1600]
            )
            boundary = re.search(r"(?:For SI:|TABLE\s+\d|INSIGHTS\s*\(|\b\d{3,4}(?:\.\d+)+\s+[A-Z])", raw_context, re.IGNORECASE)
            if boundary and boundary.start() > 0:
                raw_context = raw_context[:boundary.start()]
            context = normalize_text(raw_context)
            formula_line = line.text if line else normalize_text(page_text[max(0, match.start()): match.end() + 200])
            # Some source labels are serialized as ``Equation 4-1Vbz=...``;
            # a word boundary before the variable would miss Vbz.
            variables = sorted(
                {m.group(1) for m in re.finditer(r"(?<![A-Za-z])([A-Za-z][A-Za-z0-9]*)\s*=", context)}
                | _formula_variable_tokens(formula_line),
                key=str.casefold,
            )
            inventory.equations.append({
                "number": equation_number, "page_no": page_no, "text": formula_line,
                "context": context, "variables": variables,
                "bbox": line.bbox if line else {"left": 0.0, "top": 0.0, "right": 612.0, "bottom": 792.0},
            })

        for kind, regex in (
            ("section", SECTION_REF_RE), ("table", TABLE_REF_RE), ("chapter", CHAPTER_REF_RE), ("external_standard", STANDARD_RE)
        ):
            for match in regex.finditer(page_text):
                full = normalize_text(match.group(0))
                if kind == "table" and re.match(r"^\s*(?:\[[A-Z]+\]\s*)?TABLE\b", full):
                    continue
                inventory.references.append({
                    "reference_type": kind, "target": full, "page_no": page_no,
                    "text": full,
                })
    inventory.top_sections.sort(key=lambda row: (row["page_no"], row["bbox"]["top"]))
    inventory.numbered_sections.sort(key=lambda row: (row["page_no"], row["bbox"]["top"]))
    inventory.table_captions.sort(key=lambda row: (row["page_no"], row["bbox"]["top"]))
    inventory.equations.sort(key=lambda row: (row["page_no"], row["number"]))
    return inventory
