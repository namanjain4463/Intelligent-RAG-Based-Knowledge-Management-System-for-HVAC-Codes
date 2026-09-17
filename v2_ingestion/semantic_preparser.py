"""Deterministic annotations used before any optional model interpretation."""

from __future__ import annotations

import re
from typing import Any

from .semantic_models import SemanticAnnotations


_NUMBER = r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?"
VALUE_RE = re.compile(
    rf"(?P<raw>{_NUMBER}(?:\s*[°º-]?\s*[A-Za-z][A-Za-z0-9²³./()\- ]*)?)",
    re.IGNORECASE,
)
SECTION_REF_RE = re.compile(r"\bSections?\s+([0-9]{3,4}(?:\.[0-9]+)*(?:\([0-9]+\))?(?:\s+(?:through|to|and)\s+[0-9]{3,4}(?:\.[0-9]+)*)?)", re.I)
TABLE_REF_RE = re.compile(r"\bTables?\s+([0-9]+(?:\.[0-9]+)*(?:\([0-9]+\))?(?:\s+(?:through|to|and)\s+[0-9]+(?:\.[0-9]+)*)?)", re.I)
CHAPTER_REF_RE = re.compile(r"\bChapters?\s+([0-9]{1,2}(?:\s+(?:through|to|and)\s+[0-9]{1,2})?)", re.I)
EQUATION_REF_RE = re.compile(r"\b(?:Equation|Eq\.?)\s*([0-9]+\s*[-–]\s*[0-9]+)", re.I)
STANDARD_RE = re.compile(r"\b(?:ASHRAE|NFPA|ANSI|ASTM|UL|NSF|IMC|IECC|ICC|CSA|ARI|AHRI|AGA|ISO|SMACNA|OSHA|ACGIH)(?:\s+[A-Z0-9][A-Z0-9./-]*)?", re.I)
MODAL_RE = re.compile(r"\b(?:shall|must|may|prohibited|required|permitted|allowed|not\s+be\s+permitted|not\s+less\s+than|not\s+more\s+than|at\s+least|at\s+most|minimum|maximum)\b", re.I)
EXCEPTION_MARKER_RE = re.compile(r"\b(?:exception|exceptions|except|unless|provided\s+that)\b[^.;]*", re.I)


def _matches(regex: re.Pattern[str], text: str, key: str) -> list[dict[str, Any]]:
    return [{"text": m.group(0), "target": m.group(1) if m.lastindex else m.group(0), "start": m.start(), "end": m.end()} for m in regex.finditer(text)]


def annotate(text: str) -> SemanticAnnotations:
    values: list[dict[str, Any]] = []
    for match in VALUE_RE.finditer(text):
        raw = match.group("raw").strip()
        if not re.search(r"\d", raw):
            continue
        number_match = re.search(r"\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?", raw)
        number_text = number_match.group(0) if number_match else None
        unit = raw[number_match.end():].strip(" -") if number_match else None
        values.append({"text": raw, "number_text": number_text, "unit": unit or None, "start": match.start(), "end": match.end()})
    return SemanticAnnotations(
        modal_terms=[m.group(0) for m in MODAL_RE.finditer(text)],
        values=values,
        section_refs=_matches(SECTION_REF_RE, text, "section"),
        table_refs=_matches(TABLE_REF_RE, text, "table"),
        chapter_refs=_matches(CHAPTER_REF_RE, text, "chapter"),
        equation_refs=_matches(EQUATION_REF_RE, text, "equation"),
        external_standard_refs=[{"text": m.group(0), "target": m.group(0), "start": m.start(), "end": m.end()} for m in STANDARD_RE.finditer(text)],
        exception_markers=[{"text": m.group(0), "target": m.group(0), "start": m.start(), "end": m.end()} for m in EXCEPTION_MARKER_RE.finditer(text)],
    )


def split_clauses(text: str) -> list[str]:
    """Split only at high-confidence requirement boundaries, preserving text."""
    pattern = re.compile(
        r"\s*(?:;|(?<=\.)\s+|\band\s+(?=(?:shall|must|may|required|prohibited|permitted|allowed|a\s+minimum|a\s+maximum|not\s+(?:less|more)))|\band\s+(?=[A-Za-z][^.;]{0,120}\b(?:shall|must|may|required|prohibited|permitted|allowed)\b))\s*",
        re.I,
    )
    clauses = [part.strip() for part in pattern.split(text) if part.strip()]
    return clauses or [text.strip()]


def predicate_for(clause: str) -> str | None:
    lower = clause.casefold()
    if re.search(r"\b(?:shall\s+not|must\s+not|may\s+not|prohibited|not\s+be\s+permitted)\b", lower):
        return "PROHIBITED_IN"
    if re.search(r"\b(?:clearance|clearances)\b", lower) and re.search(r"\b(?:shall|must|required|minimum|not\s+less)\b", lower):
        return "REQUIRES_CLEARANCE"
    if re.search(r"\b(?:in\s+accordance\s+with|comply|compliance|conform)\b", lower):
        return "MUST_COMPLY_WITH"
    if re.search(r"\b(?:minimum|not\s+less\s+than|at\s+least|no\s+less\s+than)\b", lower):
        return "MINIMUM"
    if re.search(r"\b(?:maximum|not\s+more\s+than|not\s+exceed|at\s+most)\b", lower):
        return "MAXIMUM"
    if re.search(r"\b(?:permitted|allowed|may)\b", lower) and not re.search(r"\bmay\s+not\b", lower):
        return "PERMITTED_IN"
    if re.search(r"\b(?:devices?|detectors?|alarms?|dampers?|valves?|fans?|systems?|equipment|appliances?)\b", lower) and re.search(r"\b(?:shall|must|required)\b", lower):
        return "REQUIRES_DEVICE"
    if re.search(r"\b(?:shall|must|required)\b", lower):
        return "REQUIRES"
    return None


def cue_span(clause: str) -> tuple[int | None, int | None]:
    match = MODAL_RE.search(clause)
    return (match.start(), match.end()) if match else (None, None)
