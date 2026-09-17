"""Docling structural parser with PyMuPDF provenance supplementation."""

from __future__ import annotations

import importlib.metadata
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

from .models import (
    BoundingBox,
    Chapter,
    ContentBlock,
    Document,
    Equation,
    ExceptionBlock,
    ExtractionFailure,
    ListBlock,
    Reference,
    Section,
    SourceProvenance,
    Table,
    TableFragment,
    TableFootnote,
    TableRow,
    UnparsedBlock,
)
from .pdf_provenance import PdfProvenanceIndex
from .source_inventory import (
    CHAPTER_REF_RE,
    EXCEPTION_RE,
    STANDARD_RE,
    TABLE_CAPTION_RE,
    build_source_inventory,
    normalize_text,
)


CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([A-Z0-9IVX-]+)\b\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*((?:\d+\.)+\d+)\s+(.+?)\s*$")
TABLE_RE = re.compile(r"^\s*(?:\[[A-Z]+\]\s*)?TABLE\s+([A-Z0-9.\-]+(?:\(\d+\))?)(?=\s|$)\s*(.*)$", re.IGNORECASE)
TOP_SECTION_RE = re.compile(r"^\s*Section\s+(\d{3,4})\s+(.+?)\s*$", re.IGNORECASE)
SECTION_REF_RE = re.compile(r"\bSections?\s+(\d{3,4}(?:\.\d+)*(?:\(\d+\))?(?:\s+(?:through|to|and)\s+\d{3,4}(?:\.\d+)*)?)", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\bTables?\s+(\d{3,4}(?:\.\d+)*(?:\(\d+\))?(?:\s+(?:through|to|and)\s+\d{3,4}(?:\.\d+)*)?)", re.IGNORECASE)


def _ref(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("$ref") or value.get("ref")
        return str(candidate) if candidate else None
    return None


def _as_text(value: Any, references: dict[str, dict[str, Any]]) -> str:
    if isinstance(value, str):
        item = references.get(value)
        if item is not None:
            return _as_text(item, references)
        return value
    if isinstance(value, dict):
        candidate = _ref(value)
        if candidate and candidate in references:
            return _as_text(references[candidate], references)
        for key in ("text", "orig", "content", "value", "name"):
            if value.get(key) is not None:
                return _as_text(value[key], references)
    if isinstance(value, list):
        return " ".join(_as_text(item, references) for item in value).strip()
    return ""


def _item_text(item: dict[str, Any], references: dict[str, dict[str, Any]]) -> str:
    for key in ("text", "orig", "content", "value"):
        value = item.get(key)
        if value is not None:
            text = _as_text(value, references).strip()
            if text:
                return text
    return ""


def _item_id(item: dict[str, Any], fallback: str) -> str:
    value = item.get("self_ref") or item.get("id") or fallback
    return str(value).replace("#/", "").replace("/", "_")


def _label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("type") or "unknown").lower()


def _first_page(provenance: list[dict[str, Any]] | None) -> int | None:
    if not provenance:
        return None
    try:
        return int(provenance[0].get("page_no"))
    except (TypeError, ValueError, AttributeError):
        return None


class DoclingStructuralParser:
    """Parse a Docling export into structural Pydantic models.

    The parser accepts an injected converter for tests.  Production conversion
    always uses Docling's standard PDF pipeline with table structure enabled,
    while PyMuPDF is restricted to supplemental provenance.
    """

    def __init__(self, source_pdf: str | Path, converter: Any | None = None):
        self.source_pdf = Path(source_pdf).resolve()
        self.pdf = PdfProvenanceIndex(self.source_pdf)
        self.converter = converter
        self.failures: list[ExtractionFailure] = []
        self.sections: list[Section] = []
        self.chapters: list[Chapter] = []
        self.tables: list[Table] = []
        self.equations: list[Equation] = []
        self.references: list[Reference] = []
        self.unparsed_blocks: list[UnparsedBlock] = []
        self.source_inventory = build_source_inventory(self.source_pdf)
        self._section_by_id: dict[str, Section] = {}
        self._current_section: Section | None = None
        self._current_chapter: Chapter | None = None
        self._seen_table_refs: set[str] = set()
        self._seen_reference_targets: set[tuple[int | None, str]] = set()

    @property
    def parser_version(self) -> str:
        return "2.0.0"

    def _base_document(self, export: dict[str, Any] | None = None) -> Document:
        try:
            docling_version = importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            docling_version = "not-installed"
        return Document(
            id=f"document:{self.pdf.source_sha256[:16]}",
            title=self.source_pdf.stem,
            source_file=str(self.source_pdf),
            source_sha256=self.pdf.source_sha256,
            parser_name="docling-structural-v2",
            parser_version=f"{self.parser_version}; docling={docling_version}",
            pages=[self.pdf.pages[n] for n in sorted(self.pdf.pages)],
            docling_export=export,
        )

    def _failure(
        self,
        severity: str,
        code: str,
        message: str,
        item: dict[str, Any] | None = None,
        item_kind: str | None = None,
        raw_text: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        item = item or {}
        prov = self._provenance(item, item_kind or _label(item), raw_text or _item_text(item, {}), report_missing=False)
        self.failures.append(
            ExtractionFailure(
                severity=severity, code=code, message=message,
                item_kind=item_kind or _label(item), item_ref=item.get("self_ref") or item.get("id"),
                page_no=prov.page_no if prov else _first_page(item.get("prov")),
                raw_text=raw_text, provenance=prov,
                details=details or {},
            )
        )

    def _provenance(
        self,
        item: dict[str, Any],
        kind: str,
        text: str,
        report_missing: bool = True,
        fallback_prov: list[dict[str, Any]] | None = None,
    ) -> SourceProvenance:
        raw_prov = item.get("prov") or fallback_prov or []
        page_no, bbox, pdf_text, hyperlinks = self.pdf.source_for_docling_prov(raw_prov)
        if report_missing and not raw_prov:
            self._failure(
                "error", "MISSING_PROVENANCE", f"Docling {kind} item has no page/bbox provenance.",
                item=item, item_kind=kind, raw_text=text,
            )
        charspan = raw_prov[0].get("charspan") if raw_prov and isinstance(raw_prov[0], dict) else None
        char_start = charspan[0] if isinstance(charspan, (list, tuple)) and len(charspan) > 0 else None
        char_end = charspan[1] if isinstance(charspan, (list, tuple)) and len(charspan) > 1 else None
        page = self.pdf.page(page_no)
        return SourceProvenance(
            source_file=str(self.source_pdf), source_sha256=self.pdf.source_sha256,
            page_no=page_no, bbox=bbox, char_start=char_start, char_end=char_end,
            pdf_text=pdf_text or text or None,
            pdf_page_text_sha256=page.text_sha256 if page else None,
            hyperlinks=hyperlinks,
            docling_ref=item.get("self_ref") or item.get("id"),
            docling_label=item.get("label") or item.get("type"),
            docling_provenance=raw_prov[0] if raw_prov else None,
        )

    def _synthetic_section(self, first_page: int | None) -> Section:
        if "section:unassigned" in self._section_by_id:
            return self._section_by_id["section:unassigned"]
        raw = {"prov": [{"page_no": first_page, "bbox": {"l": 0, "t": 0, "r": 0, "b": 0}}]} if first_page else {}
        section = Section(
            id="section:unassigned", title="Unassigned content", number=None, level=0,
            chapter_id=self._current_chapter.id if self._current_chapter else None,
            parent_section_id=None, order=len(self.sections), provenance=self._provenance(raw, "synthetic_section", "", report_missing=False),
        )
        self.sections.append(section)
        self._section_by_id[section.id] = section
        return section

    def _section_for_content(self, item: dict[str, Any], text: str) -> Section:
        if self._current_section is not None:
            return self._current_section
        section = self._synthetic_section(_first_page(item.get("prov")))
        self._failure("warning", "UNASSIGNED_CONTENT", "Content appeared before a recognized section heading.", item, raw_text=text)
        self._current_section = section
        return section

    def _make_section(self, item: dict[str, Any], number: str, title: str) -> Section:
        level = number.count(".") + 1
        parent = None
        for candidate in reversed(self.sections):
            if candidate.number and candidate.level < level:
                parent = candidate
                break
        section = Section(
            id=f"section:{number}", number=number, title=title.rstrip("."), level=level,
            chapter_id=self._current_chapter.id if self._current_chapter else None,
            parent_section_id=parent.id if parent else None, order=len(self.sections),
            provenance=self._provenance(item, "section_header", title),
        )
        if section.id in self._section_by_id:
            self._failure("error", "DUPLICATE_SECTION_ID", f"Duplicate section identifier {section.id}.", item, raw_text=title)
            section.id = f"{section.id}:{len(self.sections)}"
        self.sections.append(section)
        self._section_by_id[section.id] = section
        self._current_section = section
        if self._current_chapter:
            self._current_chapter.section_ids.append(section.id)
        return section

    def _make_chapter(self, item: dict[str, Any], number: str, title: str) -> Chapter:
        chapter = Chapter(
            id=f"chapter:{number}", number=number, title=title.rstrip("."), order=len(self.chapters),
            provenance=self._provenance(item, "chapter_header", title),
        )
        if any(existing.id == chapter.id for existing in self.chapters):
            self._failure("error", "DUPLICATE_CHAPTER_ID", f"Duplicate chapter identifier {chapter.id}.", item, raw_text=title)
            chapter.id = f"{chapter.id}:{len(self.chapters)}"
        self.chapters.append(chapter)
        self._current_chapter = chapter
        self._current_section = None
        return chapter

    def _append_block(self, section: Section, block: ContentBlock | ListBlock | ExceptionBlock | UnparsedBlock) -> None:
        section.blocks.append(block)

    def _content_class(self, text: str, label: str = "") -> str:
        upper = text.strip().upper()
        if upper.startswith("USER NOTE"):
            return "UserNote"
        if upper.startswith("INSIGHTS"):
            return "Insights"
        if upper.startswith("SOURCE:") or "CODES.ICCSAFE.ORG" in upper:
            return "SourceArtifact"
        if label.lower() in {"commentary", "footnote", "caption"}:
            return "Commentary"
        return "NormativeSectionContent"

    def _content_block(self, item: dict[str, Any], text: str, order: int, block_type: str = "paragraph", **kwargs: Any) -> ContentBlock:
        return ContentBlock(
            id=f"block:{_item_id(item, str(order))}", block_type=block_type, text=text, order=order,
            provenance=self._provenance(item, _label(item), text), style=item.get("style"),
            content_class=kwargs.pop("content_class", self._content_class(text, _label(item))), **kwargs,
        )

    def _unparsed_block(self, item: dict[str, Any], text: str, order: int, reason: str) -> UnparsedBlock:
        block = UnparsedBlock(
            id=f"unparsed:{_item_id(item, str(order))}", text=text, order=order,
            source_label=_label(item), reason=reason, provenance=self._provenance(item, _label(item), text),
        )
        self.unparsed_blocks.append(block)
        self._failure("warning", "UNPARSED_BLOCK", reason, item, item_kind=_label(item), raw_text=text)
        return block

    def _group_children(self, item: dict[str, Any], references: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        children: list[dict[str, Any]] = []
        for child in item.get("children", []) or item.get("items", []) or []:
            ref = _ref(child)
            if ref and ref in references:
                children.append(references[ref])
            elif isinstance(child, dict):
                children.append(child)
        return children

    def _make_list(self, item: dict[str, Any], references: dict[str, dict[str, Any]], order: int) -> ListBlock:
        children = self._group_children(item, references)
        items: list[ContentBlock | ListBlock] = []
        for child_order, child in enumerate(children):
            child_label = _label(child)
            if child_label in {"list", "ordered_list", "unordered_list"}:
                items.append(self._make_list(child, references, child_order))
                continue
            text = _item_text(child, references)
            if not text:
                self._failure("error", "EMPTY_LIST_ITEM", "List item contains no text.", child, item_kind="list_item")
                continue
            items.append(self._content_block(child, text, child_order, block_type="paragraph"))
        if not children:
            self._failure("error", "EMPTY_LIST", "List group contains no resolvable children.", item, item_kind="list")
        return ListBlock(
            id=f"list:{_item_id(item, str(order))}", ordered=item.get("ordered"), items=items, order=order,
            provenance=self._provenance(
                item, "list", " ".join(
                    child.text if isinstance(child, ContentBlock)
                    else " ".join(item.text for item in child.items if isinstance(item, ContentBlock))
                    for child in items
                ),
            ),
        )

    def _make_exception(self, item: dict[str, Any], text: str, order: int) -> ExceptionBlock:
        return ExceptionBlock(
            id=f"exception:{_item_id(item, str(order))}", text=text, order=order,
            provenance=self._provenance(item, "exception", text),
        )

    def _is_exception(self, label: str, text: str) -> bool:
        return "exception" in label or bool(re.match(r"^\s*exceptions?\s*[:.]", text, re.IGNORECASE))

    def _is_heading(self, label: str, text: str) -> bool:
        return label in {"section_header", "heading", "title", "chapter"} or bool(
            CHAPTER_RE.match(text) or TOP_SECTION_RE.match(text) or SECTION_RE.match(text)
        )

    def _table_caption(self, item: dict[str, Any], references: dict[str, dict[str, Any]]) -> str | None:
        for key in ("captions", "caption", "title"):
            value = item.get(key)
            if value:
                text = _as_text(value, references).strip()
                if text and TABLE_RE.match(text):
                    return text
        return None

    @staticmethod
    def _table_number(title: str | None) -> str | None:
        match = TABLE_RE.match(title or "")
        return match.group(1) if match else None

    def _make_table(
        self,
        item: dict[str, Any],
        references: dict[str, dict[str, Any]],
        order: int,
        section: Section | None = None,
        caption_override: str | None = None,
    ) -> Table:
        table_id = f"table:{_item_id(item, str(order))}"
        raw_data = item.get("data") if isinstance(item.get("data"), dict) else item
        cells = raw_data.get("table_cells") or raw_data.get("cells") or []
        caption = self._table_caption(item, references) or caption_override
        table_prov = self._provenance(item, "table", caption or _item_text(item, references))
        parsed: dict[tuple[int, int], tuple[str, SourceProvenance, bool]] = {}
        for cell_order, cell in enumerate(cells):
            if not isinstance(cell, dict):
                self._failure("error", "INVALID_TABLE_CELL", "Table cell is not an object.", item, item_kind="table_cell")
                continue
            row = cell.get("start_row_offset_idx", cell.get("row_idx", cell.get("row", 0)))
            col = cell.get("start_col_offset_idx", cell.get("col_idx", cell.get("col", cell_order)))
            try:
                row, col = int(row), int(col)
            except (TypeError, ValueError):
                self._failure("error", "INVALID_TABLE_COORDINATE", "Table cell row/column index is not numeric.", cell, item_kind="table_cell")
                continue
            text = _item_text(cell, references)
            cell_item = dict(cell)
            if not cell_item.get("prov") and cell.get("bbox") and item.get("prov"):
                cell_item["prov"] = [dict(item["prov"][0], bbox=cell["bbox"])]
            cell_prov = self._provenance(cell_item, "table_cell", text, fallback_prov=item.get("prov"))
            parsed[(row, col)] = (text, cell_prov, bool(cell.get("column_header") or cell.get("row_header")))
        num_rows = raw_data.get("num_rows")
        num_cols = raw_data.get("num_cols")
        if num_rows is None and parsed:
            num_rows = max(row for row, _ in parsed) + 1
        if num_cols is None and parsed:
            num_cols = max(col for _, col in parsed) + 1
        if not parsed:
            self._failure("error", "EMPTY_TABLE", "Docling table has no table cells.", item, item_kind="table")
        rows: list[TableRow] = []
        for row_idx in range(int(num_rows or 0)):
            row_cells: list[str] = []
            row_provenance: list[SourceProvenance] = []
            header = False
            for col_idx in range(int(num_cols or 0)):
                value, prov, is_header = parsed.get((row_idx, col_idx), ("", table_prov, False))
                row_cells.append(value)
                row_provenance.append(prov)
                header = header or is_header
            row_prov = row_provenance[0] if row_provenance else table_prov
            rows.append(TableRow(
                id=f"{table_id}:row:{row_idx}", table_id=table_id, row_index=row_idx,
                cells=row_cells, cell_provenance=row_provenance, provenance=row_prov, is_header=header,
            ))
        footnotes: list[TableFootnote] = []
        for footnote_order, footnote in enumerate(item.get("footnotes", []) or []):
            footnote_item = references.get(_ref(footnote), footnote if isinstance(footnote, dict) else {})
            text = _as_text(footnote_item, references)
            if text:
                footnotes.append(TableFootnote(
                    id=f"{table_id}:footnote:{footnote_order}", text=text, order=footnote_order,
                    provenance=self._provenance(footnote_item, "table_footnote", text, fallback_prov=item.get("prov")),
                ))
            else:
                self._failure("warning", "EMPTY_TABLE_FOOTNOTE", "Table footnote could not be resolved.", item, item_kind="table_footnote")
        return Table(
            id=table_id, table_number=self._table_number(caption), title=caption,
            section_id=section.id if section else (self._current_section.id if self._current_section else None),
            rows=rows, footnotes=footnotes, num_rows=num_rows, num_cols=num_cols,
            provenance=table_prov, docling_markdown=item.get("markdown") or item.get("exported_markdown"),
        )

    def _make_equation(self, item: dict[str, Any], text: str, order: int) -> Equation:
        latex = item.get("latex") or item.get("formula") or item.get("math")
        equation_number = item.get("equation_number") or item.get("number")
        variables = sorted({match.group(1) for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9]*)\s*=", text)})
        return Equation(
            id=f"equation:{equation_number or _item_id(item, str(order))}", equation_number=str(equation_number) if equation_number else None,
            text=text, latex=str(latex) if latex else None, variables=variables,
            section_id=self._current_section.id if self._current_section else None, order=order,
            provenance=self._provenance(item, "equation", text),
        )

    def _make_reference(
        self,
        item: dict[str, Any],
        text: str,
        target: str,
        order: int,
        provenance: SourceProvenance | None = None,
        reference_type: str = "hyperlink",
    ) -> Reference:
        return Reference(
            id=f"reference:{len(self.references)}", text=text or target, target=target,
            reference_type=reference_type,
            section_id=self._current_section.id if self._current_section else None, order=order,
            provenance=provenance or self._provenance(item, "reference", text or target),
        )

    def _add_reference_for_links(self, item: dict[str, Any], text: str, prov: SourceProvenance, order: int) -> None:
        targets = list(prov.hyperlinks)
        for explicit_target in (item.get("target"), item.get("uri"), item.get("url")):
            if explicit_target:
                targets.append(str(explicit_target))
        for target in targets:
            key = (prov.page_no, target)
            if key in self._seen_reference_targets:
                continue
            self._seen_reference_targets.add(key)
            reference = self._make_reference(item, text, target, order, prov)
            self.references.append(reference)
            if self._current_section and reference.id not in self._current_section.reference_ids:
                self._current_section.reference_ids.append(reference.id)

    def _items_in_order(self, export: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        references: dict[str, dict[str, Any]] = {}
        for key in ("texts", "tables", "groups", "pictures", "key_value_items", "formulas"):
            for item in export.get(key, []) or []:
                if isinstance(item, dict):
                    item_ref = item.get("self_ref") or item.get("id")
                    if item_ref:
                        references[str(item_ref)] = item
        ordered: list[dict[str, Any]] = []
        for entry in export.get("body", []) or []:
            ref = _ref(entry)
            if ref and ref in references:
                ordered.append(references[ref])
            elif isinstance(entry, dict):
                ordered.append(entry)
        if not ordered:
            ordered = [item for item in references.values()]
        seen = {item.get("self_ref") or item.get("id") for item in ordered}
        for item in references.values():
            item_ref = item.get("self_ref") or item.get("id")
            if item_ref not in seen and _label(item) == "table":
                ordered.append(item)
        return ordered, references

    def parse_export(self, export: dict[str, Any]) -> Document:
        """Parse an already-exported Docling document; useful for deterministic tests."""

        self.failures = []
        self.sections = []
        self.chapters = []
        self.tables = []
        self.equations = []
        self.references = []
        self.unparsed_blocks = []
        self._section_by_id = {}
        self._current_section = None
        self._current_chapter = None
        self._seen_reference_targets = set()
        ordered, references = self._items_in_order(export)
        table_titles_by_page: dict[int, list[str]] = {}
        for candidate in ordered:
            candidate_text = _item_text(candidate, references)
            if TABLE_RE.match(candidate_text):
                page_no = _first_page(candidate.get("prov"))
                if page_no is not None:
                    table_titles_by_page.setdefault(page_no, []).append(candidate_text.rstrip("."))
        for order, item in enumerate(ordered):
            label = _label(item)
            text = _item_text(item, references)
            chapter_match = CHAPTER_RE.match(text)
            top_section_match = TOP_SECTION_RE.match(text)
            section_match = SECTION_RE.match(text)
            if chapter_match and self._is_heading(label, text):
                self._make_chapter(item, chapter_match.group(1), chapter_match.group(2))
                continue
            if top_section_match and label in {"section_header", "heading", "title", "chapter"}:
                self._make_section(item, top_section_match.group(1), top_section_match.group(2))
                continue
            if section_match and self._is_heading(label, text):
                self._make_section(item, section_match.group(1), section_match.group(2))
                continue
            if label == "table" or "table" in label:
                table_page = _first_page(item.get("prov"))
                table_prov = self._provenance(item, "table", text, report_missing=False)
                caption_override = table_titles_by_page.get(table_page, [None])[0] if table_page else None
                table_match = TABLE_RE.match(caption_override or "")
                table_section = self._section_for_table(
                    table_page, table_prov.bbox, table_match.group(1) if table_match else None,
                )
                table = self._make_table(item, references, order, section=table_section, caption_override=caption_override)
                self.tables.append(table)
                self._seen_table_refs.add(item.get("self_ref") or item.get("id") or table.id)
                if table_section:
                    table_section.table_ids.append(table.id)
                    self._append_block(table_section, self._content_block(item, table.title or "", order, "table_ref", table_id=table.id))
                continue
            if label in {"list", "ordered_list", "unordered_list"} or "list" in label and item.get("children"):
                section = self._section_for_content(item, text)
                self._append_block(section, self._make_list(item, references, order))
                continue
            if label in {"formula", "equation", "math"} or item.get("latex") or item.get("formula"):
                section = self._section_for_content(item, text)
                equation = self._make_equation(item, text, order)
                self.equations.append(equation)
                section.equation_ids.append(equation.id)
                self._append_block(section, self._content_block(item, text, order, "equation_ref", equation_id=equation.id))
                continue
            if self._is_exception(label, text):
                section = self._section_for_content(item, text)
                self._append_block(section, self._make_exception(item, text, order))
                continue
            if label in {"reference", "citation", "footnote"}:
                section = self._section_for_content(item, text)
                prov = self._provenance(item, label, text)
                target = (prov.hyperlinks[0] if prov.hyperlinks else item.get("target") or item.get("uri") or None)
                if target:
                    reference = self._make_reference(item, text, str(target), order, prov)
                    self.references.append(reference)
                    section.reference_ids.append(reference.id)
                    self._append_block(section, self._content_block(item, text, order, "reference", reference_id=reference.id))
                else:
                    self._append_block(section, self._content_block(item, text, order, "reference"))
                continue
            section = self._section_for_content(item, text)
            if label == "list_item":
                block = ListBlock(
                    id=f"list:{_item_id(item, str(order))}", items=[self._content_block(item, text, 0)], order=order,
                    provenance=self._provenance(item, label, text),
                )
            elif label in {"text", "paragraph", "section_header", "heading", "title", "chapter", "caption"}:
                block = self._content_block(item, text, order, "heading" if self._is_heading(label, text) else "paragraph")
            else:
                block = self._unparsed_block(item, text, order, f"Unsupported Docling label: {label}")
            self._append_block(section, block)
            if isinstance(block, ContentBlock):
                self._add_reference_for_links(item, text, block.provenance, order)
        self._supplement_pdf_headings()
        self._merge_pdf_exceptions()
        self._reclassify_mixed_blocks()
        self._merge_pdf_equations()
        self._merge_pdf_references()
        self._canonicalize_tables()
        self._add_page_level_links(references)
        document = self._base_document(export)
        document.chapters = self.chapters
        document.sections = self.sections
        document.tables = self.tables
        document.equations = self.equations
        document.references = self.references
        document.unparsed_blocks = self.unparsed_blocks
        document.failures = self.failures
        return document

    def _page_item(self, page_no: int, bbox: dict[str, float], label: str, text: str) -> dict[str, Any]:
        return {
            "label": label,
            "text": text,
            "prov": [{
                "page_no": page_no,
                "bbox": {"l": bbox.get("left", 0), "t": bbox.get("top", 0), "r": bbox.get("right", 612), "b": bbox.get("bottom", 792)},
            }],
        }

    def _nearest_section_id(self, page_no: int | None, top: float | None = None) -> str | None:
        if page_no is None:
            return self._current_section.id if self._current_section else None
        candidates = []
        for section in self.sections:
            section_page = section.provenance.page_no
            if section_page is None or section_page > page_no:
                continue
            if section_page < page_no or top is None:
                candidates.append(section)
            elif section.provenance.bbox is None or section.provenance.bbox.top <= top:
                candidates.append(section)
        return candidates[-1].id if candidates else None

    def _merge_pdf_equations(self) -> None:
        """Add deterministic equation records when Docling did not classify formulas."""
        existing: dict[str, Equation] = {}
        for equation in self.equations:
            if equation.equation_number:
                existing[equation.equation_number] = equation
        source_by_number: dict[str, list[dict[str, Any]]] = {}
        for source in self.source_inventory.equations:
            source_by_number.setdefault(source["number"], []).append(source)
        for number, source_items in source_by_number.items():
            source = next((item for item in source_items if "=" in item["text"]), source_items[0])
            source_variables = sorted({variable for item in source_items for variable in item["variables"]}, key=str.casefold)
            if number in existing:
                existing[number].variables = sorted(set(existing[number].variables) | set(source_variables), key=str.casefold)
                if "=" in source["text"] and "=" not in existing[number].text:
                    existing[number].text = source["text"]
                continue
            item = self._page_item(source["page_no"], source["bbox"], "pdf_equation_fallback", source["text"])
            provenance = self._provenance(item, "equation", source["text"], report_missing=False)
            equation = Equation(
                id=f"equation:{number}", equation_number=number, text=source["text"],
                latex=None, variables=source_variables, section_id=self._nearest_section_id(source["page_no"]),
                order=len(self.equations), provenance=provenance,
            )
            self.equations.append(equation)
            existing[number] = equation
        self.equations.sort(key=lambda equation: (equation.provenance.page_no or 10**9, equation.equation_number or equation.id))
        for order, equation in enumerate(self.equations):
            equation.order = order

    def _merge_pdf_exceptions(self) -> None:
        """Repair or add complete Exception blocks from the PDF text layer."""
        existing: list[tuple[Section, ExceptionBlock]] = []
        for section in self.sections:
            for block in section.blocks:
                if isinstance(block, ExceptionBlock):
                    existing.append((section, block))
        used: set[str] = set()
        for source in self.source_inventory.exception_blocks():
            source_signature = self._exception_signature(source["text"])
            candidate = next(
                (
                    pair for pair in existing
                    if pair[1].id not in used
                    and self._exception_signature(pair[1].text)[:12] == source_signature[:12]
                ),
                None,
            )
            if candidate:
                section, block = candidate
                used.add(block.id)
                block.text = source["text"]
                block.provenance = self._provenance(
                    self._page_item(source["page_no"], source["bbox"], "pdf_exception_fallback", source["text"]),
                    "exception", source["text"], report_missing=False,
                )
                block.items = [self._exception_item(source_item, source["index"], item_index) for item_index, source_item in enumerate(source["items"])]
                continue
            section_id = self._nearest_section_id(source["page_no"], source["bbox"].get("top"))
            section = next((item for item in self.sections if item.id == section_id), None) or self._synthetic_section(source["page_no"])
            block = ExceptionBlock(
                id=f"exception:pdf:{source['index']}", text=source["text"], order=len(section.blocks),
                items=[self._exception_item(source_item, source["index"], item_index) for item_index, source_item in enumerate(source["items"])],
                provenance=self._provenance(
                    self._page_item(source["page_no"], source["bbox"], "pdf_exception_fallback", source["text"]),
                    "exception", source["text"], report_missing=False,
                ),
            )
            section.blocks.append(block)
            existing.append((section, block))
            used.add(block.id)

        # A Docling exception marker that cannot be aligned to a source block
        # is preserved as an explicit unsupported block rather than silently
        # remaining as a duplicate semantic ExceptionBlock.
        for section, block in existing:
            if block.id in used:
                continue
            replacement = UnparsedBlock(
                id=f"unparsed:{block.id}", text=block.text, order=block.order,
                source_label="exception", reason="Docling exception block could not be aligned to an independent PDF exception inventory.",
                provenance=block.provenance,
            )
            section.blocks = [replacement if item.id == block.id else item for item in section.blocks]
            self.unparsed_blocks.append(replacement)
            self._failure("warning", "UNMATCHED_DOCLING_EXCEPTION", replacement.reason, item_kind="exception", raw_text=block.text)

    @staticmethod
    def _exception_signature(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", normalize_text(text).casefold())

    def _exception_item(self, source: dict[str, Any], exception_index: int, item_index: int) -> ContentBlock:
        item = self._page_item(source["page_no"], source["bbox"], "exception_item_fallback", source["text"])
        return self._content_block(
            item, source["text"], item_index, "paragraph",
        )

    def _reclassify_mixed_blocks(self) -> None:
        """Preserve Docling blocks that visibly contain multiple source records.

        A paragraph containing a later section heading or an INSIGHTS marker
        cannot safely be treated as one normative provision.  It remains in
        the corpus as an UnparsedBlock and is counted by the audit.
        """
        for section in self.sections:
            replaced: list[Any] = []
            for block in section.blocks:
                if not isinstance(block, ContentBlock) or block.block_type not in {"paragraph", "heading"}:
                    replaced.append(block)
                    continue
                text = normalize_text(block.text)
                if not self._contains_embedded_structural_marker(text):
                    replaced.append(block)
                    continue
                replacement = UnparsedBlock(
                    id=f"unparsed:{block.id}", text=block.text, order=block.order,
                    source_label=block.provenance.docling_label,
                    reason="Docling block contains multiple structural records and was not safely split.",
                    provenance=block.provenance,
                )
                replaced.append(replacement)
                self.unparsed_blocks.append(replacement)
                self._failure("warning", "MIXED_SOURCE_BLOCK", replacement.reason, item_kind="content", raw_text=block.text)
            section.blocks = replaced

    @staticmethod
    def _contains_embedded_structural_marker(text: str) -> bool:
        if re.search(r"\bINSIGHTS\s*\(", text, re.IGNORECASE) and not text.lstrip().upper().startswith("INSIGHTS"):
            return True
        for match in re.finditer(r"(?:\[[A-Z]+\]\s*)?\d{3,4}(?:\.\d+)+\s+[A-Z][a-z]", text):
            if match.start() == 0:
                continue
            prefix = text[:match.start()].rstrip().casefold()
            if re.search(r"(?:sections?|tables?|chapters?)\s*$", prefix):
                continue
            return True
        return False

    def _merge_pdf_references(self) -> None:
        """Retain explicit code cross-references separately from hyperlinks."""
        existing = {(reference.reference_type, reference.target, reference.provenance.page_no) for reference in self.references}
        for source in self.source_inventory.references:
            key = (source["reference_type"], source["target"], source["page_no"])
            if key in existing:
                continue
            item = self._page_item(source["page_no"], {"left": 0, "top": 0, "right": 612, "bottom": 792}, "pdf_reference_fallback", source["text"])
            provenance = self._provenance(item, "reference", source["text"], report_missing=False)
            reference = self._make_reference(
                item, source["text"], source["target"], len(self.references), provenance,
                reference_type=source["reference_type"],
            )
            reference.section_id = self._nearest_section_id(source["page_no"])
            self.references.append(reference)
            existing.add(key)

    def _supplement_pdf_headings(self) -> None:
        """Add only source-styled headings absent from Docling, with audit provenance."""
        existing_numbers = {section.number for section in self.sections if section.number}
        for source in self.source_inventory.section_headings:
            if source["number"] in existing_numbers:
                continue
            item = self._page_item(source["page_no"], source["bbox"], "pdf_heading_fallback", source["text"])
            self._make_section(item, source["number"], source["title"])
            existing_numbers.add(source["number"])
        self.sections.sort(key=lambda section: (
            section.provenance.page_no if section.provenance.page_no is not None else 10**9,
            section.provenance.bbox.top if section.provenance.bbox else 10**9,
            section.order,
        ))
        chapters_by_page = sorted(self.chapters, key=lambda chapter: chapter.provenance.page_no or 0)
        for order, section in enumerate(self.sections):
            section.order = order
            if section.provenance.page_no is not None:
                candidates = [chapter for chapter in chapters_by_page if (chapter.provenance.page_no or 0) <= section.provenance.page_no]
                if candidates:
                    section.chapter_id = candidates[-1].id
            section.parent_section_id = None
            for candidate in reversed(self.sections[:order]):
                if candidate.chapter_id == section.chapter_id and candidate.level < section.level:
                    section.parent_section_id = candidate.id
                    break
        for chapter in self.chapters:
            chapter.section_ids = [section.id for section in self.sections if section.chapter_id == chapter.id]

    @staticmethod
    def _same_section_series(section_id: str | None, table_number: str | None) -> bool:
        if not section_id or not table_number:
            return False
        section_number = section_id.removeprefix("section:")
        table_base = re.sub(r"\(\d+\)$", "", table_number)
        return section_number == table_base or section_number.startswith(table_base + ".")

    def _canonicalize_tables(self) -> None:
        """Stitch Docling page fragments into canonical numbered tables."""
        ordered_tables = list(self.tables)
        groups: dict[str, list[Table]] = {}
        unnumbered: list[Table] = []
        active_by_section: dict[str, str] = {}
        for table in ordered_tables:
            page_no = table.provenance.page_no
            top = table.provenance.bbox.top if table.provenance.bbox else None
            explicit_number = self._table_number(table.title) or table.table_number
            caption = self.source_inventory.table_caption_before(page_no, top)
            number = explicit_number
            if caption:
                # The PDF text layer is the source of truth for caption order.
                # Docling can attach the next page's caption to a continuation
                # fragment (notably 510.8 and 803.9(1)); use the caption that
                # actually precedes this fragment whenever it is compatible
                # with an unnumbered table's section context.
                if explicit_number is not None or self._same_section_series(table.section_id, caption["number"]):
                    number = caption["number"]
            if number is None:
                unnumbered.append(table)
                continue
            table.table_number = number
            source_caption = next((row for row in self.source_inventory.table_captions if row["number"] == number), None)
            if source_caption:
                table.title = source_caption["caption"]
            groups.setdefault(number, []).append(table)
            if table.section_id:
                active_by_section[table.section_id] = number

        canonical: list[Table] = []
        for number, fragments in groups.items():
            fragments.sort(key=lambda table: (table.provenance.page_no or 10**9, table.id))
            first = fragments[0]
            fragment_models: list[TableFragment] = []
            combined_rows: list[TableRow] = []
            max_cols = max((table.num_cols or 0 for table in fragments), default=0)
            for fragment in fragments:
                first_row = next((row for row in fragment.rows if any(cell.strip() for cell in row.cells)), None)
                header_cells = first_row.cells if first_row else []
                fragment_models.append(TableFragment(
                    id=f"{first.id}:fragment:{len(fragment_models)}", source_table_id=fragment.id,
                    page_no=fragment.provenance.page_no, row_count=len(fragment.rows), num_cols=fragment.num_cols,
                    header_cells=header_cells, is_continuation=bool(combined_rows), provenance=fragment.provenance,
                ))
                for row in fragment.rows:
                    if combined_rows and row.cells == combined_rows[0].cells and (row.is_header or any(row.cells)):
                        continue
                    combined_rows.append(row)
            for row_index, row in enumerate(combined_rows):
                row.table_id = f"table:{number}"
                row.id = f"table:{number}:row:{row_index}"
                row.row_index = row_index
            footnotes = []
            seen_footnotes: set[str] = set()
            for fragment in fragments:
                for footnote in fragment.footnotes:
                    if footnote.text not in seen_footnotes:
                        footnotes.append(footnote)
                        seen_footnotes.add(footnote.text)
            start_page = min((table.provenance.page_no or 10**9 for table in fragments), default=0)
            end_page = max((table.provenance.page_no or 0 for table in fragments), default=start_page)
            next_caption = self.source_inventory.table_caption_after(end_page)
            if next_caption and next_caption["page_no"] > end_page:
                end_page = next_caption["page_no"] - 1
            for line in self.source_inventory.footnote_lines_for_pages(start_page, end_page):
                if line.text in seen_footnotes:
                    continue
                item = self._page_item(line.page_no, line.bbox, "table_footnote_fallback", line.text)
                footnotes.append(TableFootnote(
                    id=f"table:{number}:footnote:{len(footnotes)}", text=line.text, order=len(footnotes),
                    provenance=self._provenance(item, "table_footnote", line.text, report_missing=False),
                ))
                seen_footnotes.add(line.text)
            pages = [table.provenance.page_no for table in fragments if table.provenance.page_no is not None]
            if pages and any(b - a > 1 for a, b in zip(sorted(pages), sorted(pages)[1:])):
                self._failure("warning", "AMBIGUOUS_TABLE_STITCH", f"Table {number} has non-adjacent physical fragments.", fragments[0].model_dump(), item_kind="table")
            canonical.append(Table(
                id=f"table:{number}", table_number=number, title=first.title, section_id=first.section_id,
                rows=combined_rows, footnotes=footnotes, num_rows=len(combined_rows), num_cols=max_cols,
                provenance=first.provenance, docling_markdown=next((table.docling_markdown for table in fragments if table.docling_markdown), None),
                fragments=fragment_models,
            ))
        for table in unnumbered:
            fragment = TableFragment(
                id=f"{table.id}:fragment:0", source_table_id=table.id, page_no=table.provenance.page_no,
                row_count=len(table.rows), num_cols=table.num_cols,
                header_cells=table.rows[0].cells if table.rows else [], provenance=table.provenance,
            )
            table.fragments = [fragment]
            canonical.append(table)
        self.tables = sorted(canonical, key=lambda table: (table.provenance.page_no or 10**9, table.id))

    def _section_for_table(
        self,
        page_no: int | None,
        bbox: BoundingBox | None,
        table_number: str | None = None,
    ) -> Section | None:
        """Associate late-exported tables with the nearest preceding section.

        Some Docling exports put tables in a separate collection rather than
        in ``body``.  In that case processing them after the text collection
        would otherwise attach every table to the last section in the file.
        Page and normalized coordinates are provenance, not semantic parsing.
        """

        if page_no is None:
            return self._current_section
        if table_number:
            prefix = table_number.split(".", 1)[0]
            same_series = [
                section for section in self.sections
                if section.provenance.page_no == page_no
                and section.number
                and (section.number == prefix or section.number.startswith(prefix + "."))
            ]
            if same_series:
                if bbox is not None:
                    following = [
                        section for section in same_series
                        if section.provenance.bbox is not None
                        and section.provenance.bbox.top >= bbox.bottom
                    ]
                    if following:
                        return following[0]
                return same_series[-1]
        candidates: list[Section] = []
        for section in self.sections:
            section_page = section.provenance.page_no
            if section_page is None or section_page > page_no:
                continue
            if section_page < page_no:
                candidates.append(section)
            elif bbox is None or section.provenance.bbox is None or section.provenance.bbox.top <= bbox.top + 1:
                candidates.append(section)
        return candidates[-1] if candidates else self._current_section

    def _add_page_level_links(self, references: dict[str, dict[str, Any]]) -> None:
        """Retain links not attached to a Docling item, without inferring meaning."""

        for page_no, page in self.pdf.pages.items():
            for target in page.hyperlinks:
                key = (page_no, target)
                if key in self._seen_reference_targets:
                    continue
                self._seen_reference_targets.add(key)
                raw = {"prov": [{"page_no": page_no, "bbox": {"l": 0, "t": 0, "r": page.width, "b": page.height}}]}
                prov = self._provenance(raw, "page_link", target, report_missing=False)
                self.references.append(self._make_reference(raw, target, target, len(self.references), prov))

    def _make_converter(self) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(do_ocr=False, do_table_structure=True, generate_page_images=False)
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def parse(self, page_range: tuple[int, int] | None = None) -> Document:
        """Run Docling and always return a document containing failures."""

        converter = self.converter or self._make_converter()
        captured_warnings: list[str] = []

        class _DoclingWarningHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                message = record.getMessage()
                lowered = message.lower()
                if record.levelno >= logging.WARNING and (
                    "matchingpostprocessor" in record.name.lower()
                    or "orphan" in lowered
                    or "fallback" in lowered
                    or "recover" in lowered
                ):
                    captured_warnings.append(message)

        warning_handler = _DoclingWarningHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(warning_handler)
        try:
            kwargs: dict[str, Any] = {"raises_on_error": False}
            if page_range is not None:
                kwargs["page_range"] = page_range
            result = converter.convert(self.source_pdf, **kwargs)
            export = result.document.export_to_dict()
            document = self.parse_export(export)
            errors = getattr(result, "errors", None) or []
            for error in errors:
                self._failure("error", "DOCLING_CONVERSION_ERROR", str(error), item_kind="conversion")
            for warning in captured_warnings:
                self._failure("warning", "DOCLING_STRUCTURAL_WARNING", warning, item_kind="docling")
            document.failures = self.failures
            return document
        except Exception as exc:  # retain a machine-readable failure artifact
            document = self._base_document(None)
            document.failures.append(ExtractionFailure(
                severity="fatal", code="DOCLING_CONVERSION_FAILED", message=str(exc), item_kind="conversion",
                details={"traceback": traceback.format_exc()},
            ))
            document.failures.extend(
                ExtractionFailure(severity="warning", code="DOCLING_STRUCTURAL_WARNING", message=warning, item_kind="docling")
                for warning in captured_warnings
            )
            return document
        finally:
            root_logger.removeHandler(warning_handler)
