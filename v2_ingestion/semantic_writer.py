"""Parquet/JSON output for the Phase 2A semantic layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .semantic_models import Requirement, SemanticError, SemanticInputSpan


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[columns]
    frame.to_parquet(path, index=False)


def write_semantic_outputs(
    output_dir: Path,
    spans: Iterable[SemanticInputSpan],
    requirements: Iterable[Requirement],
    errors: Iterable[SemanticError],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spans_list = list(spans); reqs = list(requirements); errs = list(errors)
    entities: dict[str, dict[str, Any]] = {}; conditions: dict[str, dict[str, Any]] = {}; exceptions: dict[str, dict[str, Any]] = {}; evidence: dict[str, dict[str, Any]] = {}
    req_rows: list[dict[str, Any]] = []
    for req in reqs:
        row = req.model_dump(mode="json")
        for key in ("subject", "object", "conditions", "exceptions", "references", "evidence"):
            row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
        req_rows.append(row)
        if req.subject: entities[req.subject.entity_id] = {**req.subject.model_dump(mode="json"), "requirement_id": req.requirement_id}
        if req.object: entities[req.object.entity_id] = {**req.object.model_dump(mode="json"), "requirement_id": req.requirement_id}
        for item in req.conditions: conditions[item.condition_id] = {**item.model_dump(mode="json"), "requirement_id": req.requirement_id}
        for item in req.exceptions: exceptions[item.exception_id] = {**item.model_dump(mode="json"), "requirement_id": req.requirement_id}
        evidence[req.evidence.evidence_id] = {**req.evidence.model_dump(mode="json"), "requirement_id": req.requirement_id}
    paths = {
        "semantic_input": output_dir / "semantic_input.parquet",
        "requirements": output_dir / "requirements.parquet",
        "entities": output_dir / "entities.parquet",
        "conditions": output_dir / "conditions.parquet",
        "exceptions": output_dir / "exceptions.parquet",
        "evidence": output_dir / "evidence.parquet",
        "semantic_errors": output_dir / "semantic_errors.parquet",
        "summary": output_dir / "semantic_summary.json",
    }
    _write_rows(paths["semantic_input"], [s.model_dump(mode="json") for s in spans_list])
    _write_rows(paths["requirements"], req_rows)
    _write_rows(paths["entities"], list(entities.values()))
    _write_rows(paths["conditions"], list(conditions.values()))
    _write_rows(paths["exceptions"], list(exceptions.values()))
    _write_rows(paths["evidence"], list(evidence.values()))
    _write_rows(paths["semantic_errors"], [e.model_dump(mode="json") for e in errs])
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
