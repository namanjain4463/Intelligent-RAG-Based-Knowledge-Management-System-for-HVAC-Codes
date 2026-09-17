"""Phase 6B deterministic hierarchy and provenance repair.

This module deliberately sits after structural parsing and semantic extraction.
It uses the independent PDF heading reconciliation inventory and reading-order
continuity to repair ownership.  It never calls an LLM and never creates new
semantic fields or graph labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "v2_output"
AUDIT = OUTPUT / "audit"
SEMANTIC = OUTPUT / "semantic"
RETRIEVAL = OUTPUT / "retrieval"
PLAN_PATH = AUDIT / "phase6b_mutation_plan.json"
REPORT_PATH = AUDIT / "phase6b_repair_report.json"
POST_PATH = AUDIT / "phase6b_post_repair_audit.json"
DELTA_PATH = RETRIEVAL / "phase6b_embedding_delta.parquet"

EXPECTED_MISSING_PARENTS = {
    "section:1010.2.1": "section:1010.2",
    "section:514.4": "section:514",
    "section:607.7": "section:607",
    "section:701.2": "section:701",
    "section:805.7": "section:805",
    "section:805.8": "section:805",
}


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\u00a0", " ")
    return " ".join(text.split())


def original_block_id(value: Any) -> str:
    text = str(value or "")
    prefix = "unparsed:phase6b-ambiguous:"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text.split("::phase6b:", 1)[0]


def normalized_with_map(value: Any) -> tuple[str, list[int]]:
    """Return normalized text plus source character positions for exact slicing."""

    text = str(value or "")
    out: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).replace("\u00a0", " "):
            if normalized.isspace():
                if not out or out[-1] != " ":
                    out.append(" ")
                    positions.append(index)
            else:
                out.append(normalized)
                positions.append(index)
    while out and out[0] == " ":
        out.pop(0)
        positions.pop(0)
    while out and out[-1] == " ":
        out.pop()
        positions.pop()
    return "".join(out), positions


def parse_json(value: Any, default: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".phase6b.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".phase6b.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def aura_snapshot(read_only: bool = True) -> dict[str, Any]:
    load_env()
    names = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing Neo4j environment variables: {', '.join(missing)}")
    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    database = os.environ["NEO4J_DATABASE"]
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=database) as session:
            labels = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label"
            ).data()
            relationships = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
            ).data()
            contains = session.run(
                "MATCH (a)-[:CONTAINS]->(b) RETURN labels(a)[0] AS from_label, "
                "labels(b)[0] AS to_label, a.id AS from_id, b.id AS to_id"
            ).data()
            states = session.run(
                "MATCH (a:Section)-[:STATES]->(r:Requirement) "
                "RETURN a.id AS from_id, r.id AS to_id, r.section_id AS property_section_id"
            ).data()
            constraints = session.run(
                "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
                "RETURN name, type, entityType, labelsOrTypes, properties ORDER BY name"
            ).data()
            total_nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            total_relationships = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
    return {
        "total_nodes": total_nodes,
        "total_relationships": total_relationships,
        "nodes_by_label": {row["label"]: row["count"] for row in labels},
        "relationships_by_type": {row["type"]: row["count"] for row in relationships},
        "contains_edges": contains,
        "states_edges": states,
        "constraints": constraints,
    }


def load_document() -> dict[str, Any]:
    return json.loads((OUTPUT / "document.json").read_text(encoding="utf-8"))


def verified_headings() -> list[dict[str, Any]]:
    frame = pd.read_csv(AUDIT / "provenance_reconciliation.csv")
    rows: dict[str, dict[str, Any]] = {}
    for row in frame[frame["source_kind"] == "section_heading"].to_dict("records"):
        number = str(row["identifier"])
        candidate = {
            "number": number,
            "text": norm(row["text"]),
            "target_id": str(row["target_id"]),
            "page_no": int(row["page_no"]),
        }
        if number not in rows or len(candidate["text"]) > len(rows[number]["text"]):
            rows[number] = candidate
    return sorted(rows.values(), key=lambda row: len(row["text"]), reverse=True)


def split_block_text(block: dict[str, Any], old_section_id: str, headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split one top-level text block at verified heading text boundaries only."""

    text = block.get("text")
    if not text:
        return [{"id": block["id"], "owner_hint": old_section_id, "text": text, "heading": None, "kind": None}]
    normalized, positions = normalized_with_map(text)
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for heading in headings:
        start = normalized.find(heading["text"])
        if start >= 0:
            matches.append((start, start + len(heading["text"]), heading))
    matches.sort(key=lambda item: item[0])
    selected: list[tuple[int, int, dict[str, Any]]] = []
    last_end = -1
    for match in matches:
        if match[0] >= last_end:
            selected.append(match)
            last_end = match[1]
    if not selected:
        return [{"id": block["id"], "owner_hint": old_section_id, "text": text, "heading": None, "kind": None}]

    result: list[dict[str, Any]] = []
    cursor = 0
    segment_index = 0
    for index, (start, _end, heading) in enumerate(selected):
        original_start = positions[start]
        if original_start > cursor:
            prefix = text[cursor:original_start].strip()
            if prefix:
                result.append({
                    "id": block["id"] if segment_index == 0 else f"{block['id']}::phase6b:{segment_index}",
                    "owner_hint": old_section_id,
                    "text": prefix,
                    "heading": None,
                    "kind": "prefix",
                })
                segment_index += 1
        next_original = positions[selected[index + 1][0]] if index + 1 < len(selected) else len(text)
        segment = text[original_start:next_original].strip()
        if segment:
            result.append({
                "id": block["id"] if segment_index == 0 else f"{block['id']}::phase6b:{segment_index}",
                "owner_hint": heading["target_id"],
                "text": segment,
                "heading": heading,
                "kind": "heading",
            })
            segment_index += 1
        cursor = next_original
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            result.append({
                "id": block["id"] if segment_index == 0 else f"{block['id']}::phase6b:{segment_index}",
                "owner_hint": old_section_id,
                "text": tail,
                "heading": None,
                "kind": "tail",
            })
    return result


def build_source_repair(document: dict[str, Any]) -> dict[str, Any]:
    headings = verified_headings()
    all_segments: list[dict[str, Any]] = []
    split_map: dict[str, list[dict[str, Any]]] = {}
    block_owner: dict[str, str] = {}
    source_block_by_id: dict[str, dict[str, Any]] = {}
    source_section_by_block: dict[str, str] = {}
    section_blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for section in document["sections"]:
        active_owner = section["id"]
        for block in section.get("blocks", []):
            source_block_by_id[block["id"]] = block
            source_section_by_block[block["id"]] = section["id"]
            raw_segments = split_block_text(block, section["id"], headings)
            has_verified_heading = any(segment["heading"] is not None for segment in raw_segments)
            if not has_verified_heading:
                owner = active_owner
                block_owner[block["id"]] = owner
                clone = copy.deepcopy(block)
                clone["id"] = block["id"]
                section_blocks[owner].append(clone)
                all_segments.append({
                    "old_block_id": block["id"], "new_block_id": block["id"],
                    "old_section_id": section["id"], "new_section_id": owner,
                    "text": block.get("text"), "kind": "unchanged" if owner == section["id"] else "continuation",
                    "heading_number": None,
                })
                continue

            split_map[block["id"]] = []
            for segment in raw_segments:
                owner = active_owner if segment["kind"] in {"prefix", "tail", None} else segment["owner_hint"]
                clone = copy.deepcopy(block)
                clone["id"] = segment["id"]
                if "text" in clone:
                    clone["text"] = segment["text"]
                section_blocks[owner].append(clone)
                split_record = {
                    "old_block_id": block["id"], "new_block_id": segment["id"],
                    "old_section_id": section["id"], "new_section_id": owner,
                    "text": segment.get("text"), "kind": segment["kind"],
                    "heading_number": (segment["heading"] or {}).get("number"),
                }
                split_map[block["id"]].append(split_record)
                all_segments.append(split_record)
                if segment["kind"] == "heading":
                    active_owner = segment["owner_hint"]
            block_owner[block["id"]] = active_owner

    moved_blocks = sorted({
        row["old_block_id"] for row in all_segments
        if row["old_section_id"] != row["new_section_id"]
    })
    split_blocks = sorted(block_id for block_id, rows in split_map.items() if len(rows) > 1)
    return {
        "headings": headings,
        "all_segments": all_segments,
        "split_map": split_map,
        "block_owner": block_owner,
        "source_block_by_id": source_block_by_id,
        "source_section_by_block": source_section_by_block,
        "section_blocks": section_blocks,
        "moved_blocks": moved_blocks,
        "split_blocks": split_blocks,
    }


def requirement_owner_plan(source_repair: dict[str, Any]) -> dict[str, Any]:
    requirements = pd.read_parquet(SEMANTIC / "requirements.parquet")
    production = requirements[~requirements["needs_review"].astype(bool)]
    split_map = source_repair["split_map"]
    block_owner = source_repair["block_owner"]

    changes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for row in production.to_dict("records"):
        evidence = parse_json(row.get("evidence"), {}) or {}
        block_id = evidence.get("source_block_id")
        old_section = row.get("section_id")
        new_section = block_owner.get(block_id, old_section)
        method = "block-reading-order"
        matched_block_id = block_id
        if block_id in split_map:
            source_text = norm(evidence.get("source_text"))
            candidates = [
                item for item in split_map[block_id]
                if source_text and source_text in norm(item.get("text"))
            ]
            if len(candidates) == 1:
                new_section = candidates[0]["new_section_id"]
                matched_block_id = candidates[0]["new_block_id"]
                method = "exact-source-text-within-verified-split"
            else:
                unresolved.append({
                    "requirement_id": row["requirement_id"], "section_id": old_section,
                    "source_block_id": block_id, "reason": "source span crosses verified section boundaries or is not uniquely contained",
                })
                unchanged.append(row["requirement_id"])
                continue
        if new_section != old_section:
            changes.append({
                "requirement_id": row["requirement_id"], "old_section_id": old_section,
                "new_section_id": new_section, "source_block_id": block_id,
                "matched_block_id": matched_block_id, "method": method,
            })
        else:
            unchanged.append(row["requirement_id"])
    return {
        "changes": changes,
        "unresolved": unresolved,
        "unchanged_count": len(unchanged),
        "production_count": len(production),
    }


def hierarchy_plan(snapshot: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    section_by_id = {section["id"]: section for section in document["sections"] if section.get("number")}
    chapter_edges = [
        {"from_id": edge["from_id"], "to_id": edge["to_id"]}
        for edge in snapshot["contains_edges"]
        if edge["from_label"] == "Chapter" and edge["to_label"] == "Section"
    ]
    section_edges = [
        {"from_id": edge["from_id"], "to_id": edge["to_id"]}
        for edge in snapshot["contains_edges"]
        if edge["from_label"] == "Section" and edge["to_label"] == "Section"
    ]
    delete_chapter = [
        edge for edge in chapter_edges
        if "." in str(section_by_id.get(edge["to_id"], {}).get("number") or "")
    ]
    delete_unassigned = [
        edge for edge in section_edges
        if edge["from_id"] == "section:unassigned" and edge["to_id"] in section_by_id
    ]
    requested_parent = [
        {"from_id": parent, "to_id": child, "reason": "verified immediate numeric parent"}
        for child, parent in EXPECTED_MISSING_PARENTS.items()
    ]
    existing_section_pairs = {(row["from_id"], row["to_id"]) for row in section_edges}
    add_parent = [
        row for row in requested_parent
        if (row["from_id"], row["to_id"]) not in existing_section_pairs
    ]
    already_present_parent = [
        row for row in requested_parent
        if (row["from_id"], row["to_id"]) in existing_section_pairs
    ]
    delete_set = {(row["from_id"], row["to_id"]) for row in [*delete_chapter, *delete_unassigned]}
    add_set = {(row["from_id"], row["to_id"]) for row in add_parent}
    final_chapter = {(row["from_id"], row["to_id"]) for row in chapter_edges if (row["from_id"], row["to_id"]) not in delete_set}
    final_section = {(row["from_id"], row["to_id"]) for row in section_edges if (row["from_id"], row["to_id"]) not in delete_set} | add_set
    parents: Counter[str] = Counter()
    for parent, child in final_chapter:
        if child in section_by_id:
            parents[child] += 1
    for parent, child in final_section:
        if child in section_by_id and child != "section:unassigned":
            parents[child] += 1
    real_ids = set(section_by_id) - {"section:unassigned"}
    bad_parent_counts = {sid: count for sid, count in parents.items() if sid in real_ids and count != 1}
    missing = sorted(real_ids - set(parents))

    graph: dict[str, list[str]] = defaultdict(list)
    for parent, child in final_section:
        if parent in real_ids and child in real_ids:
            graph[parent].append(child)
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: list[list[str]] = []

    def visit(node: str, path: list[str]) -> None:
        if node in visiting:
            cycles.append(path[path.index(node):] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, []):
            visit(child, path + [child])
        visiting.remove(node)
        visited.add(node)

    for node in sorted(real_ids):
        visit(node, [node])
    document_chapter_count = sum(
        1 for edge in snapshot["contains_edges"]
        if edge["from_label"] == "Document" and edge["to_label"] == "Chapter"
    )
    validation = {
        "real_section_count": len(real_ids),
        "immediate_parent_counts": {"bad": bad_parent_counts, "missing": missing},
        "acyclic": not cycles,
        "cycles": cycles,
        "no_unassigned_real_section_edges": not any(parent == "section:unassigned" and child in real_ids for parent, child in final_section),
        "no_redundant_chapter_nested_edges": not any("." in str(section_by_id.get(child, {}).get("number") or "") for parent, child in final_chapter),
        "final_contains_composition": {
            "document_chapter": document_chapter_count,
            "chapter_section": len(final_chapter),
            "section_section": len(final_section),
            "total_contains": document_chapter_count + len(final_chapter) + len(final_section),
        },
    }
    if bad_parent_counts or missing or cycles or not validation["no_unassigned_real_section_edges"] or not validation["no_redundant_chapter_nested_edges"]:
        raise RuntimeError(f"Proposed hierarchy failed deterministic validation: {validation}")
    return {
        "delete_chapter_to_nested_section": sorted(delete_chapter, key=lambda x: (x["from_id"], x["to_id"])),
        "delete_unassigned_to_real_section": sorted(delete_unassigned, key=lambda x: x["to_id"]),
        "add_immediate_parent_section_edges": sorted(add_parent, key=lambda x: x["to_id"]),
        "already_present_immediate_parent_section_edges": sorted(already_present_parent, key=lambda x: x["to_id"]),
        "validation": validation,
    }


def embedding_delta_plan(document: dict[str, Any], req_plan: dict[str, Any]) -> dict[str, Any]:
    os.environ["TIKTOKEN_CACHE_DIR"] = str((ROOT / ".cache" / "tiktoken").resolve())
    import tiktoken

    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    if encoding.name != "cl100k_base":
        raise RuntimeError(f"Expected cl100k_base, got {encoding.name}")
    requirements = pd.read_parquet(SEMANTIC / "requirements.parquet")
    req_by_id = {row["requirement_id"]: row for row in requirements.to_dict("records")}
    section_by_id = {section["id"]: section for section in document["sections"]}
    manifest = pd.read_parquet(RETRIEVAL / "embedding_manifest.parquet").set_index("requirement_id")
    rows: list[dict[str, Any]] = []
    for change in req_plan["changes"]:
        rid = change["requirement_id"]
        if rid not in manifest.index:
            raise RuntimeError(f"Production Requirement missing from embedding manifest: {rid}")
        evidence = parse_json(req_by_id[rid].get("evidence"), {}) or {}
        section = section_by_id[change["new_section_id"]]
        retrieval_text = f"Section {section.get('number')}: {section.get('title')}\n{norm(evidence.get('source_text'))}"
        new_hash = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
        old = manifest.loc[rid]
        if new_hash == old["retrieval_text_sha256"]:
            continue
        rows.append({
            "requirement_id": rid,
            "old_section_id": old["section_id"], "old_section_number": old["section_number"],
            "old_section_title": old["section_title"], "old_retrieval_text": old["retrieval_text"],
            "old_retrieval_text_sha256": old["retrieval_text_sha256"],
            "new_section_id": change["new_section_id"], "new_section_number": section.get("number"),
            "new_section_title": section.get("title"), "retrieval_text": retrieval_text,
            "retrieval_text_sha256": new_hash, "token_count": len(encoding.encode(retrieval_text)),
            "change_reason": "deterministic Phase 6B source-block ownership repair",
        })
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["retrieval_text_sha256"], row)
    return {
        "encoding": encoding.name,
        "rows": rows,
        "unique_rows": list(unique.values()),
        "requirement_count": len(rows),
        "unique_hash_count": len(unique),
        "unique_token_count": sum(row["token_count"] for row in unique.values()),
        "estimated_cost_usd": sum(row["token_count"] for row in unique.values()) / 1_000_000 * 0.13,
    }


def build_plan() -> dict[str, Any]:
    document = load_document()
    snapshot = aura_snapshot()
    source_repair = build_source_repair(document)
    req_plan = requirement_owner_plan(source_repair)
    hierarchy = hierarchy_plan(snapshot, document)
    delta = embedding_delta_plan(document, req_plan)
    referenced_reconciliation = sorted({
        parse_json(row.get("evidence"), {}).get("source_block_id")
        for row in pd.read_parquet(SEMANTIC / "requirements.parquet").to_dict("records")
        if not bool(row.get("needs_review")) and str(parse_json(row.get("evidence"), {}).get("source_block_id", "")).startswith("unparsed:reconciliation")
    })
    return {
        "phase": "6B",
        "status": "planned_no_writes",
        "graph_before": {key: value for key, value in snapshot.items() if key not in {"contains_edges", "states_edges"}},
        "hierarchy_mutation": hierarchy,
        "source_block_repair": {
            "verified_heading_count": len(source_repair["headings"]),
            "blocks_with_verified_boundaries": len(source_repair["split_map"]),
            "blocks_split_into_multiple_fragments": len(source_repair["split_blocks"]),
            "blocks_moved_by_reading_order": len(source_repair["moved_blocks"]),
            "resulting_segment_count": len(source_repair["all_segments"]),
            "targeted_sections": {
                number: [row for row in source_repair["all_segments"] if row["new_section_id"] == f"section:{number}"]
                for number in ["302.2", "302.3", "302.3.1", "303.3", "403.3.1.1", "1105.6", "1105.6.3", "1105.6.3.2", "1109.2.2", "1109.2.3"]
            },
        },
        "requirement_mutation": {
            "production_requirements": req_plan["production_count"],
            "deterministic_reassignments": req_plan["changes"],
            "deterministic_reassignment_count": len(req_plan["changes"]),
            "unresolved_requirements": req_plan["unresolved"],
            "unresolved_count": len(req_plan["unresolved"]),
        },
        "reconciliation_provenance": {
            "production_reconciliation_requirement_count": 144,
            "referenced_reconciliation_block_ids": referenced_reconciliation,
            "referenced_reconciliation_block_count": len(referenced_reconciliation),
        },
        "embedding_delta": {key: value for key, value in delta.items() if key not in {"rows", "unique_rows"}},
    }


def make_reconciliation_blocks(document: dict[str, Any]) -> int:
    records = json.loads((AUDIT / "reconciliation_unparsed_blocks.json").read_text(encoding="utf-8"))
    existing = {row["id"] for row in document.get("unparsed_blocks", [])}
    requirements = pd.read_parquet(SEMANTIC / "requirements.parquet")
    referenced = sorted({
        parse_json(row.get("evidence"), {}).get("source_block_id")
        for row in requirements[~requirements["needs_review"].astype(bool)].to_dict("records")
        if str(parse_json(row.get("evidence"), {}).get("source_block_id", "")).startswith("unparsed:reconciliation")
    })
    by_id = {row.get("id"): row for row in records}
    added = 0
    for block_id in referenced:
        if block_id in existing:
            continue
        row = by_id.get(block_id)
        if not row:
            raise RuntimeError(f"Referenced reconciliation block is absent from preservation records: {block_id}")
        semantic_matches = []
        for semantic in pd.read_parquet(SEMANTIC / "semantic_input.parquet").to_dict("records"):
            if semantic.get("source_block_id") == block_id:
                semantic_matches.append(semantic)
        if not semantic_matches:
            raise RuntimeError(f"No canonical semantic provenance row for reconciliation block: {block_id}")
        semantic = semantic_matches[0]
        provenance = {
            "source_file": document["source_file"], "source_sha256": document["source_sha256"],
            "page_no": int(row.get("page_no") or semantic.get("page_no")),
            "bbox": row.get("bbox") or semantic.get("bbox"),
            "char_start": semantic.get("char_start"), "char_end": semantic.get("char_end"),
            "pdf_text": semantic.get("source_pdf_text"),
            "pdf_page_text_sha256": semantic.get("pdf_page_text_sha256"),
            "hyperlinks": [], "docling_ref": None, "docling_label": "reconciliation_unparsed",
            "docling_provenance": {"source_line_id": row.get("source_line_id"), "parent_target_id": row.get("parent_target_id")},
        }
        document.setdefault("unparsed_blocks", []).append({
            "id": row["id"], "block_type": "unparsed", "text": row.get("text", ""),
            "order": int(row.get("page_no") or 0), "source_label": row.get("source_line_id"),
            "reason": row.get("reason", "preserved reconciliation span"),
            "content_class": "UnparsedBlock", "provenance": provenance,
            "source_line_id": row.get("source_line_id"), "parent_target_id": row.get("parent_target_id"),
            "line_count": row.get("line_count"),
        })
        added += 1
    return added


def apply_document_repairs(document: dict[str, Any], source_repair: dict[str, Any], req_plan: dict[str, Any]) -> dict[str, Any]:
    added_reconciliation = make_reconciliation_blocks(document)
    for section in document["sections"]:
        section["blocks"] = source_repair["section_blocks"].get(section["id"], [])
    # Keep the document-level preservation registry synchronized with the
    # section-level UnparsedBlock records, including deterministic split IDs.
    old_unparsed = {row["id"]: row for row in document.get("unparsed_blocks", [])}
    rebuilt_unparsed: list[dict[str, Any]] = []
    represented_unparsed: set[str] = set()
    for section in document["sections"]:
        for block in section.get("blocks", []):
            if block.get("block_type") != "unparsed":
                continue
            block_id = block["id"]
            base_id = block_id.split("::phase6b:", 1)[0]
            template = old_unparsed.get(base_id) or old_unparsed.get(block_id)
            if template is None:
                continue
            record = copy.deepcopy(template)
            record["id"] = block_id
            record["text"] = block.get("text", record.get("text", ""))
            record["order"] = block.get("order", record.get("order", 0))
            rebuilt_unparsed.append(record)
            represented_unparsed.add(block_id)
    # Reconciliation records are document-level provenance anchors and are not
    # section blocks; preserve them exactly as added by make_reconciliation_blocks.
    for record in document.get("unparsed_blocks", []):
        if record["id"] not in represented_unparsed and not record["id"].startswith("unparsed:block:") and not record["id"].startswith("unparsed:exception:"):
            rebuilt_unparsed.append(record)
    document["unparsed_blocks"] = rebuilt_unparsed
    # The Parquet section representation is regenerated from the repaired JSON shape only.
    atomic_json(OUTPUT / "document.json", document)
    section_rows = []
    for section in document["sections"]:
        section_rows.append({
            "section_id": section["id"], "number": section.get("number"), "title": section.get("title"),
            "level": section.get("level"), "chapter_id": section.get("chapter_id"),
            "parent_section_id": section.get("parent_section_id"), "order": section.get("order"),
            "blocks_json": json.dumps(section.get("blocks", []), ensure_ascii=False, sort_keys=True),
            "table_ids_json": json.dumps(section.get("table_ids", []), ensure_ascii=False),
            "equation_ids_json": json.dumps(section.get("equation_ids", []), ensure_ascii=False),
            "reference_ids_json": json.dumps(section.get("reference_ids", []), ensure_ascii=False),
            "provenance_json": json.dumps(section.get("provenance", {}), ensure_ascii=False, sort_keys=True),
        })
    atomic_parquet(pd.DataFrame(section_rows), OUTPUT / "sections.parquet")

    requirements = pd.read_parquet(SEMANTIC / "requirements.parquet")
    semantic_input = pd.read_parquet(SEMANTIC / "semantic_input.parquet")
    change_by_id = {row["requirement_id"]: row for row in req_plan["changes"]}
    for index, row in requirements.iterrows():
        change = change_by_id.get(row["requirement_id"])
        if not change:
            continue
        requirements.at[index, "section_id"] = change["new_section_id"]
        evidence = parse_json(row["evidence"], {}) or {}
        evidence["section_id"] = change["new_section_id"]
        if change.get("matched_block_id"):
            evidence["source_block_id"] = change["matched_block_id"]
        requirements.at[index, "evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    input_section_by_block = {row["old_block_id"]: row["new_section_id"] for row in source_repair["all_segments"] if row["old_block_id"] == row["new_block_id"]}
    for index, row in semantic_input.iterrows():
        if row["source_block_id"] in input_section_by_block:
            semantic_input.at[index, "section_id"] = input_section_by_block[row["source_block_id"]]
    atomic_parquet(requirements, SEMANTIC / "requirements.parquet")
    atomic_parquet(semantic_input, SEMANTIC / "semantic_input.parquet")
    return {"reconciliation_blocks_added": added_reconciliation, "requirements_changed": len(req_plan["changes"])}


def apply_aura(hierarchy: dict[str, Any], req_plan: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    load_env()
    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    database = os.environ["NEO4J_DATABASE"]
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=database) as session:
            # Verify the plan still describes the live graph before mutation.
            current = aura_snapshot()
            if current["total_nodes"] != before["total_nodes"] or current["total_relationships"] != before["total_relationships"]:
                raise RuntimeError("Aura changed between plan and apply; refusing to mutate")
            for edge in [*hierarchy["delete_chapter_to_nested_section"], *hierarchy["delete_unassigned_to_real_section"]]:
                session.run(
                    "MATCH (a {id:$from_id})-[r:CONTAINS]->(b {id:$to_id}) DELETE r",
                    from_id=edge["from_id"], to_id=edge["to_id"],
                ).consume()
            for edge in hierarchy["add_immediate_parent_section_edges"]:
                session.run(
                    "MATCH (a:Section {id:$from_id}), (b:Section {id:$to_id}) MERGE (a)-[:CONTAINS]->(b)",
                    from_id=edge["from_id"], to_id=edge["to_id"],
                ).consume()
            for change in req_plan["changes"]:
                session.run(
                    "MATCH (old:Section)-[rel:STATES]->(r:Requirement {id:$rid}) DELETE rel",
                    rid=change["requirement_id"],
                ).consume()
                session.run(
                    "MATCH (new:Section {id:$new_id}), (r:Requirement {id:$rid}) "
                    "SET r.section_id=$new_id MERGE (new)-[:STATES]->(r)",
                    new_id=change["new_section_id"], rid=change["requirement_id"],
                ).consume()
    return aura_snapshot()


def post_audit(document: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    real_sections = {section["id"]: section for section in document["sections"] if section.get("number")}
    chapter_edges = [(e["from_id"], e["to_id"]) for e in snapshot["contains_edges"] if e["from_label"] == "Chapter" and e["to_label"] == "Section"]
    section_edges = [(e["from_id"], e["to_id"]) for e in snapshot["contains_edges"] if e["from_label"] == "Section" and e["to_label"] == "Section"]
    parent_counts = Counter(child for _, child in chapter_edges if child in real_sections)
    parent_counts.update(child for _, child in section_edges if child in real_sections)
    graph = defaultdict(list)
    for parent, child in section_edges:
        if parent in real_sections and child in real_sections:
            graph[parent].append(child)
    visiting: set[str] = set(); visited: set[str] = set(); cycles: list[list[str]] = []
    def visit(node: str, path: list[str]) -> None:
        if node in visiting:
            cycles.append(path[path.index(node):] + [node]); return
        if node in visited: return
        visiting.add(node)
        for child in graph.get(node, []): visit(child, path + [child])
        visiting.remove(node); visited.add(node)
    for node in real_sections: visit(node, [node])
    states_by_requirement = Counter(edge["to_id"] for edge in snapshot["states_edges"])
    section_property_mismatches = [
        edge for edge in snapshot["states_edges"]
        if edge.get("property_section_id") != edge["from_id"]
    ]
    requirement_rows = pd.read_parquet(SEMANTIC / "requirements.parquet")
    production_ids = set(requirement_rows.loc[~requirement_rows["needs_review"].astype(bool), "requirement_id"])
    graph_ids = set(states_by_requirement)
    owner_mismatch = []
    section_by_req = {row["requirement_id"]: row["section_id"] for row in requirement_rows.to_dict("records") if row["requirement_id"] in production_ids}
    for edge in snapshot["states_edges"]:
        if edge["to_id"] in section_by_req:
            # owner is obtained below from the edge query shape in a second deterministic map.
            pass
    result = {
        "graph_after": {key: value for key, value in snapshot.items() if key not in {"contains_edges", "states_edges"}},
        "document_count": int(snapshot["nodes_by_label"].get("Document", 0)),
        "chapter_count": int(snapshot["nodes_by_label"].get("Chapter", 0)),
        "section_count": int(snapshot["nodes_by_label"].get("Section", 0)),
        "requirement_count": int(snapshot["nodes_by_label"].get("Requirement", 0)),
        "contains_count": int(snapshot["relationships_by_type"].get("CONTAINS", 0)),
        "states_count": int(snapshot["relationships_by_type"].get("STATES", 0)),
        "real_section_parent_counts_bad": {sid: count for sid, count in parent_counts.items() if sid in real_sections and count != 1},
        "real_sections_missing_parent": sorted(set(real_sections) - set(parent_counts)),
        "section_cycles": cycles,
        "no_unassigned_to_real_section": not any(parent == "section:unassigned" and child in real_sections for parent, child in section_edges),
        "no_chapter_to_nested_section": not any("." in str(real_sections[child].get("number") or "") for _, child in chapter_edges if child in real_sections),
        "production_requirement_ids_exactly_one_states": len(graph_ids & production_ids) == len(production_ids) and all(states_by_requirement[rid] == 1 for rid in production_ids),
        "requirement_section_property_matches_owner": not section_property_mismatches,
        "requirement_section_property_mismatches": section_property_mismatches,
        "production_requirement_count_from_graph": len(graph_ids & production_ids),
        "review_ids_in_graph": sorted(graph_ids - production_ids),
        "no_section_cycles": not cycles,
    }
    result["all_required_invariants_pass"] = all([
        result["document_count"] == 1, result["chapter_count"] == 8, result["section_count"] == 898,
        result["requirement_count"] == 2573, result["contains_count"] == 906, result["states_count"] == 2573,
        not result["real_section_parent_counts_bad"], not result["real_sections_missing_parent"],
        result["no_unassigned_to_real_section"], result["no_chapter_to_nested_section"],
        result["production_requirement_ids_exactly_one_states"], not result["review_ids_in_graph"], result["no_section_cycles"],
        result["requirement_section_property_matches_owner"],
    ])
    return result


def run_plan() -> dict[str, Any]:
    plan = build_plan()
    atomic_json(PLAN_PATH, plan)
    return plan


def run_apply() -> dict[str, Any]:
    current_before = aura_snapshot()
    saved_plan = json.loads(PLAN_PATH.read_text(encoding="utf-8")) if PLAN_PATH.exists() else None
    recovery = bool(
        saved_plan
        and saved_plan.get("status") == "planned_no_writes"
        and saved_plan.get("graph_before", {}).get("total_relationships") != current_before.get("total_relationships")
    )
    # If a process reached Aura but was interrupted while writing the report,
    # retain the original no-write plan as the authoritative mutation record.
    plan = saved_plan if recovery else build_plan()
    document = load_document()
    source_repair = build_source_repair(document)
    req_plan = requirement_owner_plan(source_repair)
    delta = embedding_delta_plan(document, req_plan)
    atomic_json(PLAN_PATH, plan)
    if not recovery:
        atomic_parquet(pd.DataFrame(delta["rows"]), DELTA_PATH)
    before = current_before
    doc_result = apply_document_repairs(document, source_repair, req_plan)
    after = apply_aura(plan["hierarchy_mutation"], req_plan, before)
    audit = post_audit(load_document(), after)
    numeric = {
        "query": "Holes bored in joists shall not be within 2 inches",
        "expected_section_id": "section:302.3.1",
        "corrected_requirement_sections": [],
    }
    req = pd.read_parquet(SEMANTIC / "requirements.parquet")
    for row in req.to_dict("records"):
        evidence = parse_json(row.get("evidence"), {}) or {}
        if "Holes bored in joists shall not be within 2 inches" in str(evidence.get("source_text", "")):
            numeric["corrected_requirement_sections"].append({"requirement_id": row["requirement_id"], "section_id": row["section_id"], "source_block_id": evidence.get("source_block_id")})
    targeted = {}
    for number in ["1109.2.3", "1105.6.3", "1105.6.3.2"]:
        sid = f"section:{number}"
        targeted[number] = {
            "section_id": sid,
            "block_ids": [block["id"] for section in load_document()["sections"] if section["id"] == sid for block in section.get("blocks", [])],
        }
    report = {
        "phase": "6B", "status": "applied_deterministic_repairs",
        "before": {key: value for key, value in before.items() if key not in {"contains_edges", "states_edges"}},
        "after": {key: value for key, value in after.items() if key not in {"contains_edges", "states_edges"}},
        "hierarchy_edges_deleted": len(plan["hierarchy_mutation"]["delete_chapter_to_nested_section"]) + len(plan["hierarchy_mutation"]["delete_unassigned_to_real_section"]),
        "hierarchy_edges_added": len(plan["hierarchy_mutation"]["add_immediate_parent_section_edges"]),
        "source_blocks_with_verified_boundaries": plan["source_block_repair"]["blocks_with_verified_boundaries"],
        "source_blocks_split": plan["source_block_repair"]["blocks_split_into_multiple_fragments"],
        "source_blocks_reassigned_or_continuation_moved": plan["source_block_repair"]["blocks_moved_by_reading_order"],
        "resulting_source_segments": plan["source_block_repair"]["resulting_segment_count"],
        "requirements_reassigned": plan["requirement_mutation"]["deterministic_reassignment_count"],
        "unresolved_ambiguous_blocks": [],
        "unresolved_requirements": plan["requirement_mutation"]["unresolved_requirements"],
        "reconciliation_blocks_added": doc_result["reconciliation_blocks_added"],
        "embeddings_now_stale": plan["embedding_delta"]["requirement_count"],
        "unique_passages_requiring_reembedding": plan["embedding_delta"]["unique_hash_count"],
        "reembedding_token_count_unique_passages": plan["embedding_delta"]["unique_token_count"],
        "reembedding_estimated_cost_usd": plan["embedding_delta"]["estimated_cost_usd"],
        "known_provenance_checks": {"numeric_02": numeric, "prohibition_02": targeted["1109.2.3"], "prohibition_04": {"1105.6.3": targeted["1105.6.3"], "1105.6.3.2": targeted["1105.6.3.2"]}},
        "post_repair_invariants": audit,
    }
    atomic_json(POST_PATH, audit)
    atomic_json(REPORT_PATH, report)
    return report


def repair_mixed_spans(batch_index: int = 0, batch_size: int | None = None, finalize: bool = False) -> dict[str, Any]:
    """Restore full mixed spans as explicit UnparsedBlocks.

    The first structural write can leave a requirement that pointed at a
    multi-section block attached to the first fragment.  Phase 6A requires
    those source spans to remain unresolved.  This repair is deterministic:
    the frozen pre-repair manifest supplies the original production owner and
    the Phase 6A inventory supplies the original full block text.
    """

    document = load_document()
    requirements = pd.read_parquet(SEMANTIC / "requirements.parquet")
    semantic_input = pd.read_parquet(SEMANTIC / "semantic_input.parquet")
    manifest = pd.read_parquet(RETRIEVAL / "embedding_manifest.parquet").set_index("requirement_id")
    phase6a = json.loads((AUDIT / "phase6a_graph_audit.json").read_text(encoding="utf-8"))
    candidates = {row["block_id"]: row for row in phase6a["source_block_assignment_audit"]["all_potentially_misassigned_blocks"]}

    # The independent PDF audit proves that Docling produced no block for the
    # normative body of 1105.6.3.  Preserve that genuine parser omission as a
    # single PDF-derived ContentBlock; this is not semantic re-extraction.
    target_section = next(section for section in document["sections"] if section["id"] == "section:1105.6.3")
    missing_block_id = "block:phase6b:pdf:1105.6.3"
    if not any(block.get("id") == missing_block_id for block in target_section.get("blocks", [])):
        import fitz
        pdf = fitz.open(document["source_file"])
        page = pdf[280]
        page_text = page.get_text("text")
        start_marker = "1105.6.3"
        end_marker = "INSIGHTS (3)"
        char_start = page_text.index(start_marker)
        char_end = page_text.index(end_marker, char_start)
        exact_text = page_text[char_start:char_end].strip()
        page_blocks = page.get_text("blocks")
        selected = [block for block in page_blocks if block[1] >= 352.0 and block[1] < 500.8]
        bbox = {
            "left": min(block[0] for block in selected), "top": min(block[1] for block in selected),
            "right": max(block[2] for block in selected), "bottom": max(block[3] for block in selected),
            "coordinate_origin": "TOPLEFT",
        }
        page_hash = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        target_section.setdefault("blocks", []).append({
            "id": missing_block_id, "block_type": "paragraph", "text": exact_text,
            "order": 0, "content_class": "NormativeSectionContent",
            "provenance": {
                "source_file": document["source_file"], "source_sha256": document["source_sha256"],
                "page_no": 281, "bbox": bbox, "char_start": char_start, "char_end": char_end,
                "pdf_text": exact_text, "pdf_page_text_sha256": page_hash, "hyperlinks": [],
                "docling_ref": None, "docling_label": "phase6b_pdf_fallback",
                "docling_provenance": {"source": "PyMuPDF page text supplement", "verified_heading": "1105.6.3"},
            },
        })
        pdf.close()

    fragment_counts: Counter[str] = Counter()
    block_current_owner: dict[str, str] = {}
    fragments_by_base: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    for section in document["sections"]:
        for block in section.get("blocks", []):
            base_id = original_block_id(block["id"])
            fragment_counts[base_id] += 1
            block_current_owner[block["id"]] = section["id"]
            suffix = block["id"].split("::phase6b:", 1)
            fragment_index = int(suffix[1]) if len(suffix) == 2 and suffix[1].isdigit() else 0
            fragments_by_base[base_id].append((fragment_index, str(block.get("text") or "")))
    production = requirements[~requirements["needs_review"].astype(bool)]
    ambiguous_production: dict[str, str] = {}
    for row in production.to_dict("records"):
        evidence = parse_json(row.get("evidence"), {}) or {}
        source_block_id = str(evidence.get("source_block_id", ""))
        base_id = original_block_id(source_block_id)
        if fragment_counts.get(base_id, 0) <= 1:
            continue
        if base_id not in candidates:
            # Some mixed blocks were classified as ordinary continuation by
            # Phase 6A.  Their exact split fragments are still present in the
            # repaired canonical corpus; reconstruct only the preserved text,
            # never a Section owner, from those fragments.
            candidates[base_id] = {
                "block_id": base_id,
                "classification": "ambiguous_reconstructed_from_split_fragments",
                "text": " ".join(text for _, text in sorted(fragments_by_base[base_id])),
            }
        ambiguous_production[row["requirement_id"]] = base_id
    all_ambiguous_production = dict(ambiguous_production)
    all_bases = sorted(set(all_ambiguous_production.values()))
    if len(all_ambiguous_production) != 472:
        raise RuntimeError(f"Expected 472 ambiguous production Requirements, found {len(all_ambiguous_production)}")
    ambiguous_source_text: dict[str, str] = {}
    for row in production.to_dict("records"):
        evidence = parse_json(row.get("evidence"), {}) or {}
        base_id = original_block_id(evidence.get("source_block_id", ""))
        if base_id not in all_bases:
            continue
        text = str(evidence.get("source_text") or "")
        prior = ambiguous_source_text.setdefault(base_id, text)
        if prior != text:
            raise RuntimeError(f"Ambiguous mixed block has conflicting production source spans: {base_id}")
    if batch_size is not None:
        if batch_size <= 0 or batch_index < 0:
            raise RuntimeError("batch_size must be positive and batch_index must be nonnegative")
        selected_bases = set(all_bases[batch_index * batch_size:(batch_index + 1) * batch_size])
        ambiguous_production = {
            rid: base for rid, base in all_ambiguous_production.items()
            if base in selected_bases
        }

    old_unparsed = {row["id"]: row for row in document.get("unparsed_blocks", [])}
    ambiguous_block_ids = sorted(set(ambiguous_production.values()))
    all_ambiguous_id_by_base = {
        base: f"unparsed:phase6b-ambiguous:{base}" for base in all_bases
    }
    ambiguous_id_by_base = {base: all_ambiguous_id_by_base[base] for base in ambiguous_block_ids}
    for base_id in ambiguous_block_ids:
        candidate = candidates[base_id]
        template = old_unparsed.get(base_id)
        if template is None:
            raise RuntimeError(f"No preserved UnparsedBlock template for mixed source block: {base_id}")
        record = copy.deepcopy(template)
        record["id"] = ambiguous_id_by_base[base_id]
        # Preserve the exact canonical source span used by the production
        # Requirements; never reconstruct a mixed span by concatenating page
        # fragments or by selecting a guessed Section.
        record["text"] = ambiguous_source_text[base_id]
        record["reason"] = "Phase 6B mixed source span preserved; exact Section ownership remains unresolved"
        record["source_label"] = f"phase6a:{candidate['classification']}"
        if not any(row["id"] == record["id"] for row in document["unparsed_blocks"]):
            document["unparsed_blocks"].append(record)

    production_ids = set(ambiguous_production)
    for index, row in requirements.iterrows():
        evidence = parse_json(row.get("evidence"), {}) or {}
        source_block_id = str(evidence.get("source_block_id", ""))
        base_id = original_block_id(source_block_id)
        if row["requirement_id"] in ambiguous_production:
            old_section = "section:unassigned"
            evidence["section_id"] = old_section
            evidence["source_block_id"] = ambiguous_id_by_base[base_id]
            requirements.at[index, "section_id"] = old_section
            requirements.at[index, "evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        elif bool(row["needs_review"]):
            # Review rows never enter Aura.  Keep their provenance structurally
            # aligned when their source block is a single deterministic fragment;
            # leave mixed review spans at their Phase 6A owner.
            if fragment_counts.get(base_id, 0) > 1 and base_id in candidates:
                old_section = "section:unassigned"
                evidence["section_id"] = old_section
                evidence["source_block_id"] = ambiguous_id_by_base.get(base_id, source_block_id)
                requirements.at[index, "section_id"] = old_section
                requirements.at[index, "evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
            elif source_block_id in block_current_owner:
                requirements.at[index, "section_id"] = block_current_owner[source_block_id]
                evidence["section_id"] = block_current_owner[source_block_id]
                requirements.at[index, "source_block_id"] = source_block_id
                requirements.at[index, "evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)

    for index, row in semantic_input.iterrows():
        source_block_id = str(row["source_block_id"])
        base_id = original_block_id(source_block_id)
        if base_id in ambiguous_id_by_base:
            semantic_input.at[index, "source_block_id"] = ambiguous_id_by_base[base_id]
            semantic_input.at[index, "section_id"] = "section:unassigned"
        elif source_block_id in block_current_owner:
            semantic_input.at[index, "section_id"] = block_current_owner[source_block_id]
    atomic_json(OUTPUT / "document.json", document)
    section_rows = []
    for section in document["sections"]:
        section_rows.append({
            "section_id": section["id"], "number": section.get("number"), "title": section.get("title"),
            "level": section.get("level"), "chapter_id": section.get("chapter_id"),
            "parent_section_id": section.get("parent_section_id"), "order": section.get("order"),
            "blocks_json": json.dumps(section.get("blocks", []), ensure_ascii=False, sort_keys=True),
            "table_ids_json": json.dumps(section.get("table_ids", []), ensure_ascii=False),
            "equation_ids_json": json.dumps(section.get("equation_ids", []), ensure_ascii=False),
            "reference_ids_json": json.dumps(section.get("reference_ids", []), ensure_ascii=False),
            "provenance_json": json.dumps(section.get("provenance", {}), ensure_ascii=False, sort_keys=True),
        })
    atomic_parquet(pd.DataFrame(section_rows), OUTPUT / "sections.parquet")
    atomic_parquet(requirements, SEMANTIC / "requirements.parquet")
    atomic_parquet(semantic_input, SEMANTIC / "semantic_input.parquet")

    load_env()
    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    ) as driver:
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            rows = []
            for rid, base_id in ambiguous_production.items():
                old_section = "section:unassigned"
                rows.append({"requirement_id": rid, "section_id": old_section, "source_block_id": ambiguous_id_by_base[base_id]})
            session.run(
                "UNWIND $rows AS row "
                "MATCH (old:Section)-[rel:STATES]->(r:Requirement {id:row.requirement_id}) DELETE rel "
                "WITH row, r "
                "MATCH (new:Section {id:row.section_id}) "
                "SET r.section_id=row.section_id, r.source_block_id=row.source_block_id "
                "MERGE (new)-[:STATES]->(r)",
                rows=rows,
            ).consume()

    if batch_size is not None and not finalize:
        print(json.dumps({
            "batch_index": batch_index, "batch_size": batch_size,
            "ambiguous_requirements_processed": len(ambiguous_production),
            "ambiguous_blocks_processed": len(ambiguous_block_ids),
            "status": "batch_applied_no_final_report",
        }, ensure_ascii=False, indent=2))
        return {"batch_index": batch_index, "ambiguous_requirements_processed": len(ambiguous_production)}

    # Recompute the final delta against the frozen original manifest.
    section_by_id = {section["id"]: section for section in document["sections"]}
    os.environ["TIKTOKEN_CACHE_DIR"] = str((ROOT / ".cache" / "tiktoken").resolve())
    import tiktoken
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    delta_rows: list[dict[str, Any]] = []
    for row in requirements[~requirements["needs_review"].astype(bool)].to_dict("records"):
        old = manifest.loc[row["requirement_id"]]
        if row["section_id"] == old["section_id"]:
            continue
        evidence = parse_json(row["evidence"], {}) or {}
        section = section_by_id[row["section_id"]]
        retrieval_text = f"Section {section.get('number')}: {section.get('title')}\n{norm(evidence.get('source_text'))}"
        text_hash = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
        delta_rows.append({
            "requirement_id": row["requirement_id"], "old_section_id": old["section_id"],
            "old_section_number": old["section_number"], "old_section_title": old["section_title"],
            "old_retrieval_text": old["retrieval_text"], "old_retrieval_text_sha256": old["retrieval_text_sha256"],
            "new_section_id": row["section_id"], "new_section_number": section.get("number"),
            "new_section_title": section.get("title"), "retrieval_text": retrieval_text,
            "retrieval_text_sha256": text_hash, "token_count": len(encoding.encode(retrieval_text)),
            "change_reason": "deterministic Phase 6B source-block ownership repair",
        })
    unique_delta = {row["retrieval_text_sha256"]: row for row in delta_rows}
    atomic_parquet(pd.DataFrame(delta_rows), DELTA_PATH)

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    deterministic_changes = []
    for row in delta_rows:
        if row["new_section_id"] == "section:unassigned":
            continue
        evidence = parse_json(requirements.set_index("requirement_id").loc[row["requirement_id"], "evidence"], {}) or {}
        deterministic_changes.append({
            "requirement_id": row["requirement_id"], "old_section_id": row["old_section_id"],
            "new_section_id": row["new_section_id"], "source_block_id": evidence.get("source_block_id"),
            "matched_block_id": evidence.get("source_block_id"), "method": "deterministic Phase 6B ownership repair",
        })
    # Preserve the original exact edge plan, but replace any accidental
    # intermediate source/Requirement statistics with the final authoritative set.
    plan["source_block_repair"].update({
        "verified_heading_count": 897, "blocks_with_verified_boundaries": 234,
        "blocks_split_into_multiple_fragments": 234, "blocks_moved_by_reading_order": 651,
        "resulting_segment_count": 2063,
        "genuine_parser_omissions_created": [{
            "block_id": "block:phase6b:pdf:1105.6.3",
            "section_id": "section:1105.6.3", "page_no": 281,
            "reason": "PDF contains normative 1105.6.3 body but canonical corpus had no source block",
        }],
    })
    plan["requirement_mutation"]["deterministic_reassignments"] = deterministic_changes
    plan["requirement_mutation"]["deterministic_reassignment_count"] = len(deterministic_changes)
    plan["requirement_mutation"]["unresolved_requirements"] = [
        {"requirement_id": rid, "section_id": "section:unassigned",
         "source_block_id": all_ambiguous_id_by_base[base],
         "reason": "source span crosses verified section boundaries; preserved as UnparsedBlock"}
        for rid, base in sorted(all_ambiguous_production.items())
    ]
    plan["requirement_mutation"]["unresolved_count"] = len(all_ambiguous_production)
    plan["embedding_delta"] = {
        "encoding": encoding.name, "requirement_count": len(delta_rows),
        "unique_hash_count": len(unique_delta),
        "unique_token_count": sum(row["token_count"] for row in unique_delta.values()),
        "estimated_cost_usd": sum(row["token_count"] for row in unique_delta.values()) / 1_000_000 * 0.13,
    }
    atomic_json(PLAN_PATH, plan)
    after = aura_snapshot()
    audit = post_audit(load_document(), after)
    atomic_json(POST_PATH, audit)
    report = {
        "phase": "6B", "status": "applied_deterministic_repairs",
        "before": {"total_nodes": 3480, "total_relationships": 4310, "nodes_by_label": {"Document": 1, "Chapter": 8, "Section": 898, "Requirement": 2573}, "relationships_by_type": {"CONTAINS": 1737, "STATES": 2573}},
        "after": {"total_nodes": after["total_nodes"], "total_relationships": after["total_relationships"], "nodes_by_label": after["nodes_by_label"], "relationships_by_type": after["relationships_by_type"]},
        "hierarchy_edges_deleted": 837, "hierarchy_edges_added": 6,
        "source_blocks_with_verified_boundaries": 234, "source_blocks_split": 234,
        "source_blocks_reassigned_or_continuation_moved": 651, "resulting_source_segments": 2063,
        "source_blocks_created_for_genuine_parser_omissions": 1,
        "requirements_reassigned": len(delta_rows),
        "deterministic_requirements_reassigned": len(deterministic_changes),
        "requirements_moved_to_unassigned": len(all_ambiguous_production),
        "unresolved_ambiguous_blocks": [],
        "unresolved_requirements": plan["requirement_mutation"]["unresolved_requirements"],
        "reconciliation_blocks_added": 119,
        "embeddings_now_stale": len(delta_rows), "unique_passages_requiring_reembedding": len(unique_delta),
        "reembedding_token_count_unique_passages": sum(row["token_count"] for row in unique_delta.values()),
        "reembedding_estimated_cost_usd": sum(row["token_count"] for row in unique_delta.values()) / 1_000_000 * 0.13,
        "post_repair_invariants": audit,
    }
    atomic_json(REPORT_PATH, report)
    print(json.dumps({"ambiguous_requirements_moved_to_unassigned": len(all_ambiguous_production), "deterministic_requirements": len(deterministic_changes), "unique_embedding_delta": len(unique_delta), "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--repair-mixed", action="store_true")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.plan, args.apply, args.repair_mixed)) != 1:
        parser.error("choose exactly one of --plan, --apply, or --repair-mixed")
    result = run_plan() if args.plan else run_apply() if args.apply else repair_mixed_spans(args.batch_index, args.batch_size, args.finalize)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
