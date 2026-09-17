"""Durable JSON and Parquet outputs for the structural v2 representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import Document, Equation, Reference, Section, Table, TableRow


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _provenance(value: Any) -> str:
    return _json(value.model_dump(mode="json") if hasattr(value, "model_dump") else value)


def write_outputs(document: Document, output_dir: str | Path) -> dict[str, Path]:
    """Write all requested artifacts and return their paths.

    Complex nested fields are serialized as JSON strings in Parquet so that
    Arrow schemas remain stable while every source/provenance field survives.
    """

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "document": output / "document.json",
        "sections": output / "sections.parquet",
        "tables": output / "tables.parquet",
        "table_rows": output / "table_rows.parquet",
        "references": output / "references.parquet",
        "equations": output / "equations.parquet",
    }
    paths["document"].write_text(document.model_dump_json(indent=2), encoding="utf-8")

    section_rows = []
    for section in document.sections:
        section_rows.append({
            "section_id": section.id, "number": section.number, "title": section.title,
            "level": section.level, "chapter_id": section.chapter_id,
            "parent_section_id": section.parent_section_id, "order": section.order,
            "blocks_json": _json([block.model_dump(mode="json") for block in section.blocks]),
            "table_ids_json": _json(section.table_ids), "equation_ids_json": _json(section.equation_ids),
            "reference_ids_json": _json(section.reference_ids), "provenance_json": _provenance(section.provenance),
        })
    pd.DataFrame(section_rows, columns=[
        "section_id", "number", "title", "level", "chapter_id", "parent_section_id", "order",
        "blocks_json", "table_ids_json", "equation_ids_json", "reference_ids_json", "provenance_json",
    ]).to_parquet(paths["sections"], index=False)

    table_rows = []
    for table in document.tables:
        table_rows.append({
            "table_id": table.id, "table_number": table.table_number, "title": table.title, "section_id": table.section_id,
            "num_rows": table.num_rows, "num_cols": table.num_cols,
            "row_ids_json": _json([row.id for row in table.rows]),
            "footnotes_json": _json([footnote.model_dump(mode="json") for footnote in table.footnotes]),
            "fragments_json": _json([fragment.model_dump(mode="json") for fragment in table.fragments]),
            "docling_markdown": table.docling_markdown, "provenance_json": _provenance(table.provenance),
        })
    pd.DataFrame(table_rows, columns=[
        "table_id", "table_number", "title", "section_id", "num_rows", "num_cols", "row_ids_json",
        "footnotes_json", "fragments_json", "docling_markdown", "provenance_json",
    ]).to_parquet(paths["tables"], index=False)

    row_rows = []
    for table in document.tables:
        for row in table.rows:
            row_rows.append({
                "row_id": row.id, "table_id": row.table_id, "row_index": row.row_index,
                "cells_json": _json(row.cells), "cell_provenance_json": _json([p.model_dump(mode="json") for p in row.cell_provenance]),
                "is_header": row.is_header, "provenance_json": _provenance(row.provenance),
            })
    pd.DataFrame(row_rows, columns=[
        "row_id", "table_id", "row_index", "cells_json", "cell_provenance_json", "is_header", "provenance_json",
    ]).to_parquet(paths["table_rows"], index=False)

    reference_rows = []
    for reference in document.references:
        reference_rows.append({
            "reference_id": reference.id, "text": reference.text, "target": reference.target,
            "reference_type": reference.reference_type,
            "section_id": reference.section_id, "order": reference.order,
            "provenance_json": _provenance(reference.provenance),
        })
    pd.DataFrame(reference_rows, columns=[
        "reference_id", "text", "target", "reference_type", "section_id", "order", "provenance_json",
    ]).to_parquet(paths["references"], index=False)

    equation_rows = []
    for equation in document.equations:
        equation_rows.append({
            "equation_id": equation.id, "text": equation.text, "latex": equation.latex,
            "equation_number": equation.equation_number, "variables_json": _json(equation.variables),
            "section_id": equation.section_id, "order": equation.order,
            "provenance_json": _provenance(equation.provenance),
        })
    pd.DataFrame(equation_rows, columns=[
        "equation_id", "equation_number", "text", "latex", "variables_json", "section_id", "order", "provenance_json",
    ]).to_parquet(paths["equations"], index=False)
    return paths
