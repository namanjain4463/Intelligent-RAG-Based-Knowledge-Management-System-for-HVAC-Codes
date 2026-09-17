"""Strict Pydantic contracts for Phase 2A semantic extraction.

The semantic layer is deliberately separate from the structural models.  These
models carry exact source evidence and do not attempt to canonicalize domain
terminology.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


Predicate = Literal[
    "REQUIRES",
    "PROHIBITED_IN",
    "PERMITTED_IN",
    "REQUIRES_CLEARANCE",
    "REQUIRES_DEVICE",
    "MUST_COMPLY_WITH",
    "MINIMUM",
    "MAXIMUM",
]


class EntityRef(SemanticModel):
    entity_id: str
    mention: str
    entity_type: str = "unspecified"
    source_start: int | None = None
    source_end: int | None = None
    canonical_name: str | None = None


class Condition(SemanticModel):
    condition_id: str
    text: str
    condition_type: str = "unspecified"
    entities: list[EntityRef] = Field(default_factory=list)
    source_start: int | None = None
    source_end: int | None = None


class ExceptionRef(SemanticModel):
    exception_id: str
    text: str
    exception_block_id: str | None = None
    source_start: int | None = None
    source_end: int | None = None


class ReferenceRef(SemanticModel):
    reference_id: str
    reference_type: str
    target: str
    text: str
    source_start: int | None = None
    source_end: int | None = None


class RequirementEvidence(SemanticModel):
    evidence_id: str
    input_span_id: str
    source_block_id: str | None = None
    source_text: str
    source_pdf_text: str | None = None
    page_no: int
    section_id: str
    bbox: dict[str, Any] | None = None
    char_start: int | None = None
    char_end: int | None = None
    provenance_status: Literal["canonical", "reconciled"]
    extraction_method: Literal["deterministic", "llm"]
    source_sha256: str | None = None
    pdf_page_text_sha256: str | None = None


class Requirement(SemanticModel):
    requirement_id: str
    document_id: str
    section_id: str
    predicate: Predicate
    subject: EntityRef | None = None
    object: EntityRef | None = None
    value_text: str | None = None
    unit: str | None = None
    conditions: list[Condition] = Field(default_factory=list)
    exceptions: list[ExceptionRef] = Field(default_factory=list)
    references: list[ReferenceRef] = Field(default_factory=list)
    evidence: RequirementEvidence
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False
    review_reason: str | None = None


class SemanticInputSpan(SemanticModel):
    input_span_id: str
    document_id: str
    chapter_id: str | None = None
    section_id: str
    source_block_id: str | None = None
    source_type: Literal[
        "section_prose",
        "list_item",
        "exception",
        "unparsed_normative",
        "table_reference",
        "equation_reference",
        "reference_context",
    ]
    source_text: str
    source_pdf_text: str | None = None
    page_no: int
    bbox: dict[str, Any] | None = None
    char_start: int | None = None
    char_end: int | None = None
    provenance_status: Literal["canonical", "reconciled"]
    provenance_evidence: str
    order: int
    normative: bool = True
    exception_block_id: str | None = None
    source_sha256: str | None = None
    pdf_page_text_sha256: str | None = None


class SemanticError(SemanticModel):
    error_id: str
    code: str
    message: str
    severity: Literal["warning", "error", "quarantine"]
    input_span_id: str | None = None
    section_id: str | None = None
    page_no: int | None = None
    source_text: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SemanticAnnotations(SemanticModel):
    modal_terms: list[str] = Field(default_factory=list)
    values: list[dict[str, Any]] = Field(default_factory=list)
    section_refs: list[dict[str, Any]] = Field(default_factory=list)
    table_refs: list[dict[str, Any]] = Field(default_factory=list)
    chapter_refs: list[dict[str, Any]] = Field(default_factory=list)
    equation_refs: list[dict[str, Any]] = Field(default_factory=list)
    external_standard_refs: list[dict[str, Any]] = Field(default_factory=list)
    exception_markers: list[dict[str, Any]] = Field(default_factory=list)
