"""Canonical evidence helpers shared by runtime and historical benchmarks."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'v2_output'

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def json_value(value: Any, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def page_from(provenance: Any) -> int | None:
    if isinstance(provenance, dict):
        value = provenance.get("page_no")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def section_number_from_id(section_id: Any) -> str | None:
    text = str(section_id or "")
    return text.split(":", 1)[1] if text.startswith("section:") else None


def build_exception_fallbacks() -> dict[str, str]:
    """Recover complete exception text already preserved in canonical evidence.

    Some repaired structural blocks retain only an ``Exception:`` heading while
    the production canonical evidence record contains the complete exception.
    This is a read-only evidence assembly fallback, not a parser mutation.
    """

    df = pd.read_parquet(OUTPUT / "semantic" / "requirements.parquet")
    candidates: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        if bool(getattr(row, "needs_review", False)):
            continue
        evidence = json_value(getattr(row, "evidence", None), {}) or {}
        block_id = evidence.get("source_block_id") or getattr(row, "source_block_id", None)
        source_text = norm_text(evidence.get("source_text"))
        if not source_text or not source_text.lower().startswith("exception:"):
            continue
        section_id = evidence.get("section_id") or getattr(row, "section_id", None)
        section_number = section_number_from_id(section_id)
        if not section_number or not block_id or not str(block_id).startswith("exception:"):
            continue
        candidates.setdefault(section_number, []).append(source_text)
    return {number: max(values, key=len) for number, values in candidates.items()}


def known_heading_matches(text: str, section_numbers: set[str]):
    if not text:
        return []
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?P<number>"
        + "|".join(sorted((re.escape(n) for n in section_numbers), key=len, reverse=True))
        + r")(?:\s+)(?=[A-Z\[])",
    )
    return list(pattern.finditer(text))


def trim_to_owned_span(text: str, current_number: str, section_numbers: set[str]) -> str:
    """Keep the current section's span when a repaired block crosses headings."""

    text = norm_text(text)
    if not text:
        return ""

    matches = known_heading_matches(text, section_numbers)
    other = [m for m in matches if m.group("number") != current_number]
    if other:
        text = text[: other[0].start()].strip()

    # A source block can repeat its explicit heading.  The section heading is
    # emitted separately, so retain only its body here.
    heading = re.match(
        rf"^\s*(?:\[[A-Z]{{2}}\]\s*)?{re.escape(current_number)}\s+[^.\n]{{1,180}}\.\s*",
        text,
    )
    if heading:
        text = text[heading.end() :].strip()
    return text


def allowed_block_text(
    section: dict[str, Any],
    block: dict[str, Any],
    section_numbers: set[str],
    exception_fallback: str | None,
) -> tuple[str, str] | None:
    content_class = str(block.get("content_class") or "")
    block_type = str(block.get("block_type") or "")
    text = norm_text(block.get("text"))
    if not text:
        return None
    if content_class.lower() in {"insights", "commentary", "usernote", "sourcenavigation", "sourceartifact"}:
        return None
    if "insights" in text.lower() or text.lower().startswith("user note"):
        return None
    is_exception = block_type.lower() == "exception" or content_class.lower().startswith("exception")
    if is_exception and exception_fallback and len(text) < len(exception_fallback):
        text = exception_fallback
    elif block_type.lower() == "table_ref":
        return None
    elif block_type.lower() == "unparsed":
        # A short fallback marker or a continuation of that same exception is
        # already represented by the complete canonical exception record.
        if exception_fallback and (
            text.lower().startswith("exception")
            or text.startswith(",")
            or text.startswith("provided")
        ):
            return None
    text = trim_to_owned_span(text, str(section["number"]), section_numbers)
    if not text or text.lower() in {"exception:", "[bs]", "[bf]"}:
        return None
    text = re.sub(r"^\[(?:BS|BF)\]\s*", "", text).strip()
    return text, block.get("id") or ""


def section_page(section: dict[str, Any]) -> int | None:
    page = page_from(section.get("provenance"))
    if page is not None:
        return page
    for block in section.get("blocks", []):
        page = page_from(block.get("provenance"))
        if page is not None:
            return page
    return None


def clean_section_title(value: Any) -> str:
    # A few repaired section titles inherit the renderer's trailing INSIGHTS
    # marker.  It is navigation/commentary, not part of the regulatory title.
    return re.sub(r"\s*INSIGHTS\s*\(\d+\)\s*$", "", str(value or ""), flags=re.I).strip()


def own_section_records(
    section: dict[str, Any],
    section_numbers: set[str],
    exception_fallbacks: dict[str, str],
) -> list[dict[str, Any]]:
    number = str(section["number"])
    records: list[dict[str, Any]] = []
    title = clean_section_title(section.get("title"))
    heading = f"{number} {title}".strip()
    if heading:
        records.append({
            "section_number": number,
            "section_title": title,
            "page": section_page(section),
            "source_text": heading,
            "source_key": f"heading:{section.get('id')}",
            "kind": "section_heading",
        })
    fallback = exception_fallbacks.get(number)
    for block in section.get("blocks", []):
        result = allowed_block_text(section, block, section_numbers, fallback)
        if not result:
            continue
        text, source_key = result
        records.append({
            "section_number": number,
            "section_title": title,
            "page": page_from(block.get("provenance")) or section_page(section),
            "source_text": text,
            "source_key": source_key,
            "kind": "section_block",
        })
    return records


def ancestor_numbers(section: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    result: list[str] = []
    parent_id = section.get("parent_section_id")
    seen: set[str] = set()
    while parent_id and parent_id in by_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id[parent_id]
        if parent.get("number") != "unassigned":
            result.append(str(parent["number"]))
        parent_id = parent.get("parent_section_id")
    result.reverse()
    return result


def referenced_table_numbers(section: dict[str, Any], records: list[dict[str, Any]], table_by_id: dict[str, dict[str, Any]]) -> list[str]:
    numbers: list[str] = []
    for table_id in section.get("table_ids") or []:
        table = table_by_id.get(str(table_id))
        if table:
            numbers.append(str(table.get("table_number")))
    # Do not infer a table dependency merely because prose mentions a table
    # number.  The repaired structural corpus's explicit table_ids are the
    # deterministic dependency signal; this avoids injecting large unrelated
    # tables for otherwise bounded distractor passages.
    return list(dict.fromkeys(numbers))


def table_records(table: dict[str, Any], section_number: str, section_title: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    table_number = str(table.get("table_number") or "")
    title = norm_text(table.get("title")) or f"TABLE {table_number}"
    records.append({
        "section_number": section_number,
        "section_title": section_title,
        "page": page_from(table.get("provenance")),
        "source_text": title,
        "source_key": f"table:{table.get('id')}:title",
        "kind": "table_title",
    })
    for row in table.get("rows") or []:
        cells = [norm_text(cell) for cell in row.get("cells") or []]
        if not any(cells):
            continue
        pages = [page_from(p) for p in row.get("cell_provenance") or []]
        pages = [p for p in pages if p is not None]
        records.append({
            "section_number": section_number,
            "section_title": section_title,
            "page": min(pages) if pages else page_from(table.get("provenance")),
            "source_text": " | ".join(cells),
            "source_key": str(row.get("id") or f"table:{table_number}:row:{row.get('row_index')}"),
            "kind": "table_row",
        })
    for footnote in table.get("footnotes") or []:
        text = norm_text(footnote.get("text"))
        if not text:
            continue
        records.append({
            "section_number": section_number,
            "section_title": section_title,
            "page": page_from(footnote.get("provenance")) or page_from(table.get("provenance")),
            "source_text": f"Footnote: {text}",
            "source_key": str(footnote.get("id") or ""),
            "kind": "table_footnote",
        })
    return records
