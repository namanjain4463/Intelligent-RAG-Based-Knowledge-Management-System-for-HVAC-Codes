"""Version 2 structural ingestion for the HVAC code PDF.

This package deliberately stops at document structure and source provenance.
It does not create semantic entities, requirements, or graph records.
"""

from .models import (
    Chapter,
    ContentBlock,
    Document,
    Equation,
    ExceptionBlock,
    ListBlock,
    Reference,
    Section,
    Table,
    TableFragment,
    TableFootnote,
    TableRow,
    UnparsedBlock,
)

__all__ = [
    "Chapter",
    "ContentBlock",
    "Document",
    "Equation",
    "ExceptionBlock",
    "ListBlock",
    "Reference",
    "Section",
    "Table",
    "TableFragment",
    "TableFootnote",
    "TableRow",
    "UnparsedBlock",
]
