"""Pydantic models for the v2 structural representation.

The models intentionally describe document structure and provenance only.  No
model in this module represents an HVAC entity, requirement, or graph node.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BoundingBox(StrictModel):
    left: float
    top: float
    right: float
    bottom: float
    coordinate_origin: str = "TOPLEFT"


class SourceProvenance(StrictModel):
    """A lossless-enough pointer back to the source PDF and Docling item."""

    source_file: str
    source_sha256: str
    page_no: int | None = None
    bbox: BoundingBox | None = None
    char_start: int | None = None
    char_end: int | None = None
    pdf_text: str | None = None
    pdf_page_text_sha256: str | None = None
    hyperlinks: list[str] = Field(default_factory=list)
    docling_ref: str | None = None
    docling_label: str | None = None
    docling_provenance: dict[str, Any] | None = None


class PageProvenance(StrictModel):
    page_no: int
    width: float
    height: float
    exact_pdf_text: str
    text_sha256: str
    hyperlinks: list[str] = Field(default_factory=list)


class ExtractionFailure(StrictModel):
    severity: Literal["warning", "error", "fatal"]
    code: str
    message: str
    item_kind: str | None = None
    item_ref: str | None = None
    page_no: int | None = None
    raw_text: str | None = None
    provenance: SourceProvenance | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ContentBlock(StrictModel):
    id: str
    block_type: Literal["paragraph", "heading", "table_ref", "equation_ref", "reference", "unknown"]
    text: str
    order: int
    provenance: SourceProvenance
    content_class: Literal[
        "NormativeSectionContent", "Commentary", "UserNote", "Insights", "SourceArtifact", "NavigationArtifact"
    ] = "NormativeSectionContent"
    style: str | None = None
    table_id: str | None = None
    equation_id: str | None = None
    reference_id: str | None = None


class ListBlock(StrictModel):
    id: str
    block_type: Literal["list"] = "list"
    ordered: bool | None = None
    items: list[ContentBlock | ListBlock] = Field(default_factory=list)
    order: int
    provenance: SourceProvenance
    content_class: Literal["NormativeSectionContent", "Commentary", "UserNote", "Insights", "UnparsedBlock"] = "NormativeSectionContent"


class ExceptionBlock(StrictModel):
    id: str
    block_type: Literal["exception"] = "exception"
    text: str
    order: int
    provenance: SourceProvenance
    items: list[ContentBlock] = Field(default_factory=list)
    content_class: Literal["Exception"] = "Exception"


class UnparsedBlock(StrictModel):
    id: str
    block_type: Literal["unparsed"] = "unparsed"
    text: str
    order: int
    source_label: str | None = None
    reason: str
    content_class: Literal["UnparsedBlock"] = "UnparsedBlock"
    provenance: SourceProvenance


class TableFootnote(StrictModel):
    id: str
    text: str
    order: int
    provenance: SourceProvenance


class TableRow(StrictModel):
    id: str
    table_id: str
    row_index: int
    cells: list[str] = Field(default_factory=list)
    cell_provenance: list[SourceProvenance] = Field(default_factory=list)
    provenance: SourceProvenance
    is_header: bool = False


class TableFragment(StrictModel):
    id: str
    source_table_id: str
    page_no: int | None = None
    row_count: int = 0
    num_cols: int | None = None
    header_cells: list[str] = Field(default_factory=list)
    is_continuation: bool = False
    provenance: SourceProvenance


class Table(StrictModel):
    id: str
    table_number: str | None = None
    title: str | None = None
    section_id: str | None = None
    rows: list[TableRow] = Field(default_factory=list)
    footnotes: list[TableFootnote] = Field(default_factory=list)
    num_rows: int | None = None
    num_cols: int | None = None
    provenance: SourceProvenance
    docling_markdown: str | None = None
    fragments: list[TableFragment] = Field(default_factory=list)


class Equation(StrictModel):
    id: str
    equation_number: str | None = None
    text: str
    latex: str | None = None
    variables: list[str] = Field(default_factory=list)
    section_id: str | None = None
    order: int
    provenance: SourceProvenance


class Reference(StrictModel):
    id: str
    text: str
    target: str | None = None
    reference_type: Literal["hyperlink", "section", "table", "chapter", "external_standard", "other"] = "hyperlink"
    section_id: str | None = None
    order: int
    provenance: SourceProvenance


class Section(StrictModel):
    id: str
    number: str | None = None
    title: str
    level: int
    chapter_id: str | None = None
    parent_section_id: str | None = None
    order: int
    blocks: list[ContentBlock | ListBlock | ExceptionBlock | UnparsedBlock] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    equation_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    provenance: SourceProvenance


class Chapter(StrictModel):
    id: str
    number: str | None = None
    title: str
    order: int
    section_ids: list[str] = Field(default_factory=list)
    provenance: SourceProvenance


class Document(StrictModel):
    id: str
    title: str
    source_file: str
    source_sha256: str
    parser_name: str
    parser_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pages: list[PageProvenance] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    unparsed_blocks: list[UnparsedBlock] = Field(default_factory=list)
    failures: list[ExtractionFailure] = Field(default_factory=list)
    docling_export: dict[str, Any] | None = None
