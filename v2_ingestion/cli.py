"""Command-line entry point for the v2 structural ingestion pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .parser import DoclingStructuralParser
from .writer import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docling structural ingestion for HVAC-Codes.pdf")
    parser.add_argument("source", type=Path, help="Path to the source PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and Parquet artifacts")
    parser.add_argument("--page-start", type=int, default=None)
    parser.add_argument("--page-end", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    page_range = None
    if (args.page_start is None) != (args.page_end is None):
        raise SystemExit("--page-start and --page-end must be supplied together")
    if args.page_start is not None:
        page_range = (args.page_start, args.page_end)
    document = DoclingStructuralParser(args.source).parse(page_range=page_range)
    paths = write_outputs(document, args.output_dir)
    failure_path = args.output_dir / "failures.json"
    failure_path.write_text(
        json.dumps([failure.model_dump(mode="json") for failure in document.failures], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "counts": {
            "chapters": len(document.chapters), "sections": len(document.sections),
            "tables": len(document.tables), "table_rows": sum(len(table.rows) for table in document.tables),
            "references": len(document.references), "equations": len(document.equations),
            "failures": len(document.failures),
        },
        "failure_report": str(failure_path.resolve()),
    }, indent=2))
    return 2 if any(failure.severity == "fatal" for failure in document.failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
