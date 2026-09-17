"""Phase 6E read-only retrieval benchmark with one corrected gold label."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from .phase6d_refresh import (
    OUTPUT,
    RETRIEVAL,
    REPORT_PATH,
    atomic_json,
    finite_vector,
    load_env,
    neo4j_driver,
    section_parent_map,
)


ARTIFACT_PATH = RETRIEVAL / "phase6e_corrected_benchmark.json"


def load_corrected_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = json.loads((OUTPUT / "generation" / "phase5b2_end_to_end.json").read_text(encoding="utf-8"))
    cases = copy.deepcopy(source["cases"])
    corrections = []
    for case in cases:
        if case["question_id"] == "numeric-02":
            old = list(case["expected_sections"])
            if old != ["302.2"]:
                raise RuntimeError(f"numeric-02 no longer has the expected stale label: {old}")
            case["expected_sections"] = ["302.3.1"]
            corrections.append({
                "question_id": "numeric-02",
                "old_expected_section": "302.2",
                "new_expected_section": "302.3.1",
                "reason": "Phase 6B independently verified from the canonical PDF structure that the bored-hole requirement belongs to Section 302.3.1.",
                "provenance_basis": "Canonical PDF Section 302.3.1 heading and source block block:texts_59.",
            })
    if len(corrections) != 1:
        raise RuntimeError(f"Expected exactly one gold-label correction, found {len(corrections)}")
    if len(cases) != 20:
        raise RuntimeError(f"Expected 20 frozen cases, found {len(cases)}")
    return cases, {"source_artifact": "v2_output/generation/phase5b2_end_to_end.json", "corrections": corrections}


def evaluate(driver: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    query = __import__("pandas").read_parquet(RETRIEVAL / "phase4b3_query_embeddings.parquet")
    if len(query) != 20:
        raise RuntimeError("The cached query embedding set is not the frozen 20-question set")
    query_vectors = {str(row.question_id): finite_vector(row.embedding) for row in query.itertuples(index=False)}
    parent, number = section_parent_map()

    def ancestors(section_id: str) -> set[str]:
        result: set[str] = set()
        current = section_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            section_number = number.get(current)
            if section_number not in (None, "None"):
                result.add(section_number)
            next_parent = parent.get(current)
            if not next_parent or next_parent == "section:unassigned":
                break
            current = str(next_parent)
        return result

    def vector_hits(vector: list[float]) -> list[dict[str, Any]]:
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            return session.run("""
                CALL db.index.vector.queryNodes('hvac_passage_embeddings', 10, $embedding)
                YIELD node, score
                OPTIONAL MATCH (s:Section)-[:STATES]->(node)
                RETURN node.id AS requirement_id, node.retrieval_hash AS retrieval_hash,
                       score, s.id AS section_id, s.number AS section_number, s.title AS section_title
                ORDER BY score DESC
            """, embedding=vector).data()

    rows = []
    for case in cases:
        qid = str(case["question_id"])
        expected = {str(value) for value in case["expected_sections"]}
        explicit = re.search(r"\bSection\s+(\d+(?:\.\d+)*)", str(case["question"]), re.IGNORECASE)
        hits = vector_hits(query_vectors[qid])
        raw_sections = list(dict.fromkeys(str(hit["section_number"]) for hit in hits if hit.get("section_number") is not None))
        expanded_by_rank = [ancestors(str(hit["section_id"])) for hit in hits if hit.get("section_id")]
        expanded_top1 = expanded_by_rank[0] if expanded_by_rank else set()
        expanded_top5 = set().union(*expanded_by_rank[:5]) if expanded_by_rank else set()
        if explicit:
            router_top1_sections = {explicit.group(1)}
            router_top5_sections = {explicit.group(1)}
            route = "exact-section"
        else:
            router_top1_sections = expanded_top1
            router_top5_sections = expanded_top5
            route = "vector-plus-ancestors"
        rows.append({
            "question_id": qid,
            "question": case["question"],
            "expected_sections": sorted(expected),
            "route": route,
            "raw_vector_hits": [
                {"rank": index + 1, "section_number": hit.get("section_number"), "score": hit.get("score"), "requirement_id": hit.get("requirement_id")}
                for index, hit in enumerate(hits)
            ],
            "raw_vector_top1_hit": bool(expected & set(raw_sections[:1])),
            "raw_vector_top5_hit": bool(expected & set(raw_sections[:5])),
            "ancestor_expanded_top1_sections": sorted(expanded_top1),
            "ancestor_expanded_top5_sections": sorted(expanded_top5),
            "ancestor_expanded_top1_hit": bool(expected & expanded_top1),
            "ancestor_expanded_top5_hit": bool(expected & expanded_top5),
            "deterministic_router_top1_hit": bool(expected & router_top1_sections),
            "deterministic_router_top5_hit": bool(expected & router_top5_sections),
            "expected_section_evidence_present": bool(expected & router_top5_sections),
        })

    def metric(key: str) -> dict[str, Any]:
        hits = sum(bool(row[key]) for row in rows)
        return {"hits": hits, "questions": len(rows), "rate": hits / len(rows)}

    selected = {row["question_id"]: row for row in rows}
    return {
        "question_count": len(rows),
        "raw_vector_top1": metric("raw_vector_top1_hit"),
        "raw_vector_top5": metric("raw_vector_top5_hit"),
        "ancestor_expanded_vector_top1": metric("ancestor_expanded_top1_hit"),
        "ancestor_expanded_vector_top5": metric("ancestor_expanded_top5_hit"),
        "deterministic_router_top1": metric("deterministic_router_top1_hit"),
        "deterministic_router_top5": metric("deterministic_router_top5_hit"),
        "expected_section_evidence_coverage": metric("expected_section_evidence_present"),
        "numeric-02": selected["numeric-02"],
        "prohibition-02": selected["prohibition-02"],
        "prohibition-04": selected["prohibition-04"],
        "cases": rows,
    }


def main() -> None:
    load_env()
    cases, correction = load_corrected_cases()
    driver = neo4j_driver()
    try:
        driver.verify_connectivity()
        result = evaluate(driver, cases)
    finally:
        driver.close()
    artifact = {
        "phase": "6E",
        "status": "completed_read_only",
        "openai_api_calls": 0,
        "aura_writes": 0,
        "questions_source": correction["source_artifact"],
        "ground_truth_corrections": correction["corrections"],
        "evaluation": result,
    }
    atomic_json(artifact, ARTIFACT_PATH)
    print(json.dumps({
        "artifact": str(ARTIFACT_PATH),
        "correction": correction["corrections"],
        "metrics": {key: result[key] for key in (
            "raw_vector_top1", "raw_vector_top5", "ancestor_expanded_vector_top1",
            "ancestor_expanded_vector_top5", "deterministic_router_top1",
            "deterministic_router_top5", "expected_section_evidence_coverage",
        )},
        "numeric-02": {
            "expected": result["numeric-02"]["expected_sections"],
            "raw_hits": result["numeric-02"]["raw_vector_hits"][:10],
            "ancestor_top1": result["numeric-02"]["ancestor_expanded_top1_sections"],
            "ancestor_top5": result["numeric-02"]["ancestor_expanded_top5_sections"],
        },
        "prohibition-02": {"expected": result["prohibition-02"]["expected_sections"], "raw_hits": result["prohibition-02"]["raw_vector_hits"][:5], "evidence_present": result["prohibition-02"]["expected_section_evidence_present"]},
        "prohibition-04": {"expected": result["prohibition-04"]["expected_sections"], "raw_hits": result["prohibition-04"]["raw_vector_hits"][:5], "ancestor_top5": result["prohibition-04"]["ancestor_expanded_top5_sections"], "evidence_present": result["prohibition-04"]["expected_section_evidence_present"]},
    }, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
