"""Minimal, idempotent Phase 3 loader for the validated v2 corpus.

This module intentionally contains no extraction logic.  It reads the existing
document and semantic Parquet outputs, creates only the requested graph shape,
and uses the official Neo4j Python driver.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - existing repository already provides it
    load_dotenv = None


CONSTRAINTS = {
    "document_id_unique": "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "chapter_id_unique": "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
    "section_id_unique": "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
    "requirement_id_unique": "CREATE CONSTRAINT requirement_id_unique IF NOT EXISTS FOR (n:Requirement) REQUIRE n.id IS UNIQUE",
}


def _json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None


def _mention(value: Any) -> str | None:
    parsed = _json(value)
    if not parsed:
        return None
    return parsed.get("mention") if isinstance(parsed, dict) else None


def _texts(value: Any) -> list[str]:
    parsed = _json(value)
    if not isinstance(parsed, list):
        return []
    return [str(item.get("text")) for item in parsed if isinstance(item, dict) and item.get("text")]


def _run_write(tx: Any, query: str, **params: Any) -> None:
    tx.run(query, **params).consume()


class Phase3Loader:
    def __init__(self, document_path: Path, semantic_dir: Path, report_path: Path) -> None:
        self.document_path = document_path
        self.semantic_dir = semantic_dir
        self.report_path = report_path
        if load_dotenv is not None:
            load_dotenv()
        required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing required Neo4j environment variables: {', '.join(missing)}")
        self.uri = os.environ["NEO4J_URI"]
        self.username = os.environ["NEO4J_USERNAME"]
        self.password = os.environ["NEO4J_PASSWORD"]
        self.database = os.environ["NEO4J_DATABASE"]

    def _read_inputs(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        document = json.loads(self.document_path.read_text(encoding="utf-8"))
        semantic_input = pd.read_parquet(self.semantic_dir / "semantic_input.parquet")
        requirements = pd.read_parquet(self.semantic_dir / "requirements.parquet")
        review_count = int(requirements["needs_review"].astype(bool).sum())
        eligible = requirements[~requirements["needs_review"].astype(bool)].copy()
        if len(eligible) != 2573 or review_count != 254:
            raise RuntimeError(f"Validated corpus invariant failed: eligible={len(eligible)}, review={review_count}")
        source_type_by_span = dict(zip(semantic_input["input_span_id"], semantic_input["source_type"]))
        table_derived = []
        for _, row in eligible.iterrows():
            evidence = _json(row["evidence"]) or {}
            source_type = source_type_by_span.get(evidence.get("input_span_id"))
            if source_type in {"table_cell", "table_row", "table_footnote"}:
                table_derived.append(row["requirement_id"])
        if table_derived:
            raise RuntimeError(f"Table-derived requirements reached the production input: {len(table_derived)}")

        chapters: list[dict[str, Any]] = []
        for chapter in document.get("chapters") or []:
            provenance = chapter.get("provenance") or {}
            if not chapter.get("id") or not chapter.get("title") or not provenance.get("page_no"):
                continue
            chapters.append({
                "id": chapter["id"], "number": chapter.get("number"), "title": chapter["title"],
                "order": chapter.get("order"), "page": provenance.get("page_no"),
            })
        sections: list[dict[str, Any]] = []
        section_ids = set()
        for section in document.get("sections") or []:
            provenance = section.get("provenance") or {}
            if not section.get("id") or not section.get("title") or not provenance.get("page_no"):
                continue
            section_ids.add(section["id"])
            sections.append({
                "id": section["id"], "number": section.get("number"), "title": section["title"],
                "chapter_id": section.get("chapter_id"), "parent_section_id": section.get("parent_section_id"),
                "order": section.get("order"), "page": provenance.get("page_no"),
            })
        chapter_ids = {row["id"] for row in chapters}
        if any(row.get("chapter_id") not in chapter_ids for row in sections):
            raise RuntimeError("A structurally valid section has no valid chapter")
        if any(row.get("parent_section_id") and row["parent_section_id"] not in section_ids for row in sections):
            raise RuntimeError("A section has an unresolved parent section")

        requirement_rows: list[dict[str, Any]] = []
        for _, row in eligible.iterrows():
            evidence = _json(row["evidence"]) or {}
            requirement_rows.append({
                "id": row["requirement_id"],
                "predicate": row["predicate"],
                "subject": _mention(row["subject"]),
                "object": _mention(row["object"]),
                "value": row["value_text"] if pd.notna(row["value_text"]) else None,
                "unit": row["unit"] if pd.notna(row["unit"]) else None,
                "condition": _texts(row["conditions"]),
                "exception": _texts(row["exceptions"]),
                "source_text": evidence.get("source_text"),
                "page": evidence.get("page_no"),
                "source_block_id": evidence.get("source_block_id"),
                "char_start": evidence.get("char_start"),
                "char_end": evidence.get("char_end"),
                "confidence": float(row["confidence"]),
                "section_id": row["section_id"],
            })
        if any(row["section_id"] not in section_ids for row in requirement_rows):
            raise RuntimeError("An eligible requirement has no structurally valid section")
        counts = {
            "chapters_source": len(chapters), "sections_source": len(sections),
            "eligible_requirements_source": len(requirement_rows), "review_requirements_excluded": review_count,
            "quarantined_missing_provenance_excluded": 28, "table_derived_excluded": 0,
        }
        return document, chapters, sections, requirement_rows, counts

    @staticmethod
    def _snapshot(session: Any) -> dict[str, Any]:
        labels = session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label").data()
        relationships = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type").data()
        total_nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        total_relationships = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        return {
            "total_nodes": total_nodes, "total_relationships": total_relationships,
            "nodes_by_label": {row["label"]: row["count"] for row in labels},
            "relationships_by_type": {row["type"]: row["count"] for row in relationships},
        }

    @staticmethod
    def _constraint_snapshot(session: Any) -> list[dict[str, Any]]:
        rows = session.run("SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties ORDER BY name").data()
        return rows

    def _create_constraints(self, session: Any) -> None:
        for query in CONSTRAINTS.values():
            session.execute_write(_run_write, query)

    def _load_nodes(self, session: Any, document: dict[str, Any], chapters: list[dict[str, Any]], sections: list[dict[str, Any]], requirements: list[dict[str, Any]]) -> None:
        session.execute_write(_run_write, """
            MERGE (n:Document {id: $row.id})
            SET n += $row
        """, row={
            "id": document["id"], "title": document.get("title"), "source_file": document.get("source_file"),
            "source_sha256": document.get("source_sha256"), "parser_name": document.get("parser_name"),
            "parser_version": document.get("parser_version"),
        })
        for rows, query in (
            (chapters, """
                UNWIND $rows AS row
                MERGE (n:Chapter {id: row.id})
                SET n += row
            """),
            (sections, """
                UNWIND $rows AS row
                MERGE (n:Section {id: row.id})
                SET n += row
            """),
            (requirements, """
                UNWIND $rows AS row
                MERGE (n:Requirement {id: row.id})
                SET n += row
            """),
        ):
            for start in range(0, len(rows), 500):
                session.execute_write(_run_write, query, rows=rows[start:start + 500])

    def _load_relationships(self, session: Any, document_id: str, chapters: list[dict[str, Any]], sections: list[dict[str, Any]], requirements: list[dict[str, Any]]) -> None:
        session.execute_write(_run_write, """
            MATCH (d:Document {id: $document_id})
            UNWIND $rows AS row
            MATCH (c:Chapter {id: row.id})
            MERGE (d)-[:CONTAINS]->(c)
        """, document_id=document_id, rows=chapters)
        session.execute_write(_run_write, """
            UNWIND $rows AS row
            MATCH (c:Chapter {id: row.chapter_id})
            MATCH (s:Section {id: row.id})
            MERGE (c)-[:CONTAINS]->(s)
        """, rows=sections)
        subsection_rows = [row for row in sections if row.get("parent_section_id")]
        if subsection_rows:
            session.execute_write(_run_write, """
                UNWIND $rows AS row
                MATCH (parent:Section {id: row.parent_section_id})
                MATCH (child:Section {id: row.id})
                MERGE (parent)-[:CONTAINS]->(child)
            """, rows=subsection_rows)
        for start in range(0, len(requirements), 500):
            session.execute_write(_run_write, """
                UNWIND $rows AS row
                MATCH (s:Section {id: row.section_id})
                MATCH (r:Requirement {id: row.id})
                MERGE (s)-[:STATES]->(r)
            """, rows=requirements[start:start + 500])

    def run(self) -> dict[str, Any]:
        document, chapters, sections, requirements, source_counts = self._read_inputs()
        report: dict[str, Any] = {"source": source_counts, "errors": [], "neo4j_write_mechanism": "official neo4j Python driver"}
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            driver.verify_connectivity()
            with driver.session(database=self.database) as session:
                before = self._snapshot(session)
                if before["total_nodes"] != 0 or before["total_relationships"] != 0:
                    raise RuntimeError(f"Aura was not empty before load: {before}")
                report["before_load"] = before
                existing_constraints = self._constraint_snapshot(session)
                report["constraints_before"] = existing_constraints
                unexpected = [row for row in existing_constraints if row.get("name") not in CONSTRAINTS]
                if unexpected:
                    raise RuntimeError(f"Unexpected pre-existing constraints: {unexpected}")
                self._create_constraints(session)
                report["constraints_after_create"] = self._constraint_snapshot(session)
                self._load_nodes(session, document, chapters, sections, requirements)
                self._load_relationships(session, document["id"], chapters, sections, requirements)
                report["first_load"] = self._snapshot(session)
                # The second invocation must be a true idempotency check.
                self._load_nodes(session, document, chapters, sections, requirements)
                self._load_relationships(session, document["id"], chapters, sections, requirements)
                report["second_load"] = self._snapshot(session)
                report["idempotent"] = report["first_load"] == report["second_load"]
                report["constraints_final"] = self._constraint_snapshot(session)
                if not report["idempotent"]:
                    raise RuntimeError("Counts changed on the second idempotent load")
        except Exception as exc:
            report["errors"].append(str(exc))
            raise
        finally:
            driver.close()
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load validated Phase 3 HVAC graph into Neo4j Aura")
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = Phase3Loader(args.document, args.semantic_dir, args.report).run()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
