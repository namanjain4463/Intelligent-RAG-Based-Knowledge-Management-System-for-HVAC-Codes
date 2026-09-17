"""Phase 4A retrieval baseline: exact sections and Neo4j full-text only.

No OpenAI, embeddings, vector indexes, LLMs, query rewriting, synonyms, or
reranking are used here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


FULLTEXT_INDEX_NAME = "requirement_text_fulltext"
SECTION_RE = re.compile(r"(?<!\d)(\d{3,4}(?:\.\d+)+)(?!\d)")


# Each expected section was selected manually from the existing Section nodes
# and source-backed Phase 3 graph before retrieval was evaluated.
EVALUATION_QUESTIONS: list[dict[str, Any]] = [
    {"id": "section-01", "category": "explicit_section", "question": "What rooms or spaces are prohibited for fuel-fired appliances under Section 303.3?", "expected_sections": ["303.3"]},
    {"id": "section-02", "category": "explicit_section", "question": "What outdoor airflow rate does Section 403.3.1.1 require?", "expected_sections": ["403.3.1.1"]},
    {"id": "section-03", "category": "explicit_section", "question": "What are the machinery room requirements in Section 1104.2?", "expected_sections": ["1104.2"]},
    {"id": "section-04", "category": "explicit_section", "question": "What ventilation requirements apply under Section 1105.6?", "expected_sections": ["1105.6"]},
    {"id": "numeric-01", "category": "numeric_value", "question": "How far above the pit floor must appliances be installed?", "expected_sections": ["303.7"]},
    {"id": "numeric-02", "category": "numeric_value", "question": "What minimum distance is required between bored holes and the top or bottom of a joist?", "expected_sections": ["302.2"]},
    {"id": "numeric-03", "category": "numeric_value", "question": "What maximum refrigerant quantity threshold triggers machinery room requirements?", "expected_sections": ["1104.2"]},
    {"id": "numeric-04", "category": "numeric_value", "question": "How far from a building opening must an outdoor refrigerating system be before the ventilation exception applies?", "expected_sections": ["1105.6"]},
    {"id": "prohibition-01", "category": "prohibition_permission", "question": "Where may fuel-fired appliances not be located?", "expected_sections": ["303.3"]},
    {"id": "prohibition-02", "category": "prohibition_permission", "question": "In which locations is refrigerant piping prohibited?", "expected_sections": ["1109.2.3"]},
    {"id": "prohibition-03", "category": "prohibition_permission", "question": "What equipment or appliances are prohibited in Group H occupancies?", "expected_sections": ["304.4"]},
    {"id": "prohibition-04", "category": "prohibition_permission", "question": "Are multiple fans permitted to produce the emergency ventilation rate?", "expected_sections": ["1105.6"]},
    {"id": "condition-01", "category": "condition_exception", "question": "What exception applies to refrigerant circuit access ports in controlled areas?", "expected_sections": ["1101.9"]},
    {"id": "condition-02", "category": "condition_exception", "question": "Under what conditions is natural or mechanical ventilation required for an outdoor refrigerating system?", "expected_sections": ["1105.6"]},
    {"id": "condition-03", "category": "condition_exception", "question": "What exception applies to the prohibited locations in Section 303.3?", "expected_sections": ["303.3"]},
    {"id": "condition-04", "category": "condition_exception", "question": "When do components containing refrigerant need to be located in a machinery room?", "expected_sections": ["1104.2"]},
    {"id": "free-text-01", "category": "ordinary_regulatory", "question": "How must joints between different piping materials be made?", "expected_sections": ["1108.1.1"]},
    {"id": "free-text-02", "category": "ordinary_regulatory", "question": "What does the code require for refrigerant pipe penetrations?", "expected_sections": ["1109.5"]},
    {"id": "free-text-03", "category": "ordinary_regulatory", "question": "How must boilers and other equipment be mounted?", "expected_sections": ["1004.4"]},
    {"id": "free-text-04", "category": "ordinary_regulatory", "question": "What testing or certification is required for piping, tubing and fittings?", "expected_sections": ["301.5"]},
]


def _required_environment() -> tuple[str, str, str, str]:
    if load_dotenv is not None:
        load_dotenv()
    names = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required Neo4j environment variables: {', '.join(missing)}")
    return tuple(os.environ[name] for name in names)  # type: ignore[return-value]


def _requirement_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"), "source_text": row.get("source_text"), "page": row.get("page"),
        "predicate": row.get("predicate"), "value": row.get("value"), "unit": row.get("unit"),
        "condition": row.get("condition") or [], "exception": row.get("exception") or [],
        "subject": row.get("subject"), "object": row.get("object"),
        "source_block_id": row.get("source_block_id"), "char_start": row.get("char_start"), "char_end": row.get("char_end"),
        "confidence": row.get("confidence"),
    }


class Phase4ARetriever:
    def __init__(self) -> None:
        uri, username, password, database = _required_environment()
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def ensure_fulltext_index(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE FULLTEXT INDEX requirement_text_fulltext IF NOT EXISTS "
                "FOR (n:Requirement) ON EACH [n.source_text, n.subject, n.object]"
            ).consume()
            deadline = time.monotonic() + 30.0
            latest: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                latest = session.run(
                    "SHOW FULLTEXT INDEXES YIELD name, state, entityType, labelsOrTypes, properties "
                    "WHERE name = $name RETURN name, state, entityType, labelsOrTypes, properties",
                    name=FULLTEXT_INDEX_NAME,
                ).data()
                if latest and latest[0].get("state") == "ONLINE":
                    return latest[0]
                time.sleep(0.25)
            raise RuntimeError(f"Full-text index did not become ONLINE: {latest}")

    def exact_section_retrieval(self, query: str) -> dict[str, Any]:
        identifiers = SECTION_RE.findall(query)
        if not identifiers:
            return {"query": query, "section_identifiers": [], "sections": []}
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                UNWIND $numbers AS number
                MATCH (s:Section {number: number})
                OPTIONAL MATCH (s)-[:STATES]->(r:Requirement)
                RETURN s.number AS section_number, s.title AS section_title,
                       collect(CASE WHEN r IS NULL THEN NULL ELSE {
                         id:r.id, source_text:r.source_text, page:r.page, predicate:r.predicate,
                         value:r.value, unit:r.unit, condition:r.condition, exception:r.exception,
                         source_block_id:r.source_block_id, char_start:r.char_start,
                         char_end:r.char_end, confidence:r.confidence
                       } END) AS requirements
                ORDER BY section_number
                """,
                numbers=identifiers,
            ).data()
        sections = []
        for row in rows:
            sections.append({
                "section_number": row["section_number"], "section_title": row["section_title"],
                "requirements": [_requirement_record(item) for item in row["requirements"] if item is not None],
            })
        return {"query": query, "section_identifiers": identifiers, "sections": sections}

    def fulltext_requirement_retrieval(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                CALL db.index.fulltext.queryNodes($index_name, $search_query) YIELD node, score
                MATCH (s:Section)-[:STATES]->(node)
                RETURN node.id AS id, score, s.number AS section_number, s.title AS section_title,
                       node.source_text AS source_text, node.page AS page, node.predicate AS predicate,
                       node.value AS value, node.unit AS unit, node.condition AS condition,
                       node.exception AS exception, node.source_block_id AS source_block_id,
                       node.char_start AS char_start, node.char_end AS char_end,
                       node.confidence AS confidence
                ORDER BY score DESC, id
                LIMIT $limit
                """,
                index_name=FULLTEXT_INDEX_NAME, search_query=query, limit=limit,
            ).data()
        return [_requirement_record(row | {"score": row.get("score"), "section_number": row.get("section_number"), "section_title": row.get("section_title")}) | {"score": row.get("score"), "section_number": row.get("section_number"), "section_title": row.get("section_title")} for row in rows]

    def evaluate(self, output_path: Path) -> dict[str, Any]:
        index = self.ensure_fulltext_index()
        exact_questions = [item for item in EVALUATION_QUESTIONS if item["category"] == "explicit_section"]
        exact_rows: list[dict[str, Any]] = []
        for item in exact_questions:
            result = self.exact_section_retrieval(item["question"])
            returned = {section["section_number"] for section in result["sections"]}
            expected = set(item["expected_sections"])
            exact_rows.append({"id": item["id"], "question": item["question"], "expected_sections": sorted(expected), "returned_sections": sorted(returned), "success": expected.issubset(returned), "result": result})

        fulltext_rows: list[dict[str, Any]] = []
        for item in EVALUATION_QUESTIONS:
            results = self.fulltext_requirement_retrieval(item["question"], limit=10)
            section_numbers = [row.get("section_number") for row in results]
            expected = set(item["expected_sections"])
            fulltext_rows.append({
                "id": item["id"], "category": item["category"], "question": item["question"],
                "expected_sections": sorted(expected), "top10_sections": section_numbers,
                "top1_hit": bool(section_numbers and section_numbers[0] in expected),
                "top5_hit": bool(set(section_numbers[:5]) & expected), "results": results,
            })
        exact_success = sum(row["success"] for row in exact_rows)
        top1_success = sum(row["top1_hit"] for row in fulltext_rows)
        top5_success = sum(row["top5_hit"] for row in fulltext_rows)
        report = {
            "phase": "4A",
            "openai_called": False, "embeddings_created": False, "vector_index_created": False, "llm_used": False,
            "database": self.database, "fulltext_index": index,
            "evaluation_question_count": len(EVALUATION_QUESTIONS),
            "exact_section_retrieval": {"success_count": exact_success, "question_count": len(exact_rows), "success_rate": exact_success / len(exact_rows), "results": exact_rows},
            "fulltext_top1_section_hit_rate": {"hit_count": top1_success, "question_count": len(fulltext_rows), "rate": top1_success / len(fulltext_rows)},
            "fulltext_top5_section_hit_rate": {"hit_count": top5_success, "question_count": len(fulltext_rows), "rate": top5_success / len(fulltext_rows)},
            "failure_examples": [row for row in fulltext_rows if not row["top5_hit"]][:5],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4A exact-section and full-text retrieval baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    retriever = Phase4ARetriever()
    try:
        report = retriever.evaluate(args.output)
    finally:
        retriever.close()
    print(json.dumps({key: report[key] for key in ("database", "fulltext_index", "evaluation_question_count", "exact_section_retrieval", "fulltext_top1_section_hit_rate", "fulltext_top5_section_hit_rate", "failure_examples")}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
