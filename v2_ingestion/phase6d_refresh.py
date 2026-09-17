"""Phase 6D production embedding/cache and Aura retrieval refresh.

The script intentionally has no GPT/Responses/Chat/LLM code.  It uses the
embeddings endpoint only for hashes missing from the repaired production
manifest, preserves the historical cache, and performs idempotent Aura
metadata/vector-property updates with the official Neo4j driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "v2_output"
RETRIEVAL = OUTPUT / "retrieval"
AUDIT = OUTPUT / "audit"
MANIFEST_PATH = RETRIEVAL / "phase6c_production_embedding_manifest.parquet"
UNASSIGNED_MANIFEST_PATH = RETRIEVAL / "phase6c_unassigned_embedding_manifest.parquet"
HISTORICAL_CACHE_PATH = RETRIEVAL / "embedding_cache.parquet"
CURRENT_CACHE_PATH = RETRIEVAL / "phase6d_embedding_cache_current.parquet"
USAGE_PATH = RETRIEVAL / "phase6d_embedding_usage.json"
REPORT_PATH = AUDIT / "phase6d_refresh_report.json"
MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
EMBEDDING_RATE = 0.13
MAX_BATCH = 128
MAX_NEW_TEXTS = 259
MAX_TOKENS = 24_000


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def parse_json(value: Any, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def finite_vector(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != DIMENSIONS:
        raise RuntimeError(f"Embedding dimension is not {DIMENSIONS}")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise RuntimeError("Embedding contains NaN or infinity")
    return vector


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def check_embedding_frame(frame: pd.DataFrame, expected_hashes: set[str] | None = None) -> None:
    required = {"retrieval_text_sha256", "requirement_id", "embedding_model", "embedding_dimension", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Embedding cache missing columns: {sorted(missing)}")
    if frame["retrieval_text_sha256"].astype(str).duplicated().any():
        raise RuntimeError("Embedding cache contains duplicate hashes")
    for row in frame.itertuples(index=False):
        if row.embedding_model != MODEL or int(row.embedding_dimension) != DIMENSIONS:
            raise RuntimeError(f"Invalid embedding metadata for {row.retrieval_text_sha256}")
        finite_vector(row.embedding)
    if expected_hashes is not None:
        actual = set(frame["retrieval_text_sha256"].astype(str))
        extra = actual - expected_hashes
        if extra:
            raise RuntimeError(f"Current cache contains non-production hashes: {len(extra)}")


def load_manifest() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], dict[str, str]]:
    production = pd.read_parquet(MANIFEST_PATH)
    unassigned = pd.read_parquet(UNASSIGNED_MANIFEST_PATH)
    if len(production) != 2100:
        raise RuntimeError(f"Expected 2100 deterministic production Requirements, found {len(production)}")
    if len(unassigned) != 473:
        raise RuntimeError(f"Expected 473 separate unassigned rows, found {len(unassigned)}")
    if production["section_id"].eq("section:unassigned").any():
        raise RuntimeError("Production manifest contains section:unassigned")
    if production["retrieval_text"].eq("").any():
        raise RuntimeError("Production manifest contains empty retrieval text")
    if production["retrieval_text_sha256"].eq("").any():
        raise RuntimeError("Production manifest contains an empty hash")
    by_hash: dict[str, dict[str, Any]] = {}
    for row in production.to_dict("records"):
        h = str(row["retrieval_text_sha256"])
        prior = by_hash.get(h)
        if prior is not None and prior["retrieval_text"] != row["retrieval_text"]:
            raise RuntimeError(f"Hash maps to conflicting retrieval text: {h}")
        if prior is None or str(row["requirement_id"]) < str(prior["requirement_id"]):
            by_hash[h] = row
    representative_by_id = {str(row["requirement_id"]): h for h, row in by_hash.items()}
    return production, unassigned, by_hash, representative_by_id


def load_tokenizer() -> Any:
    cache_dir = ROOT / ".cache" / "tiktoken"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    import tiktoken
    encoding = tiktoken.encoding_for_model(MODEL)
    if encoding.name != "cl100k_base":
        raise RuntimeError(f"Unexpected tokenizer: {encoding.name}")
    return encoding


def preflight_local() -> dict[str, Any]:
    production, unassigned, by_hash, _ = load_manifest()
    encoding = load_tokenizer()
    historical = pd.read_parquet(HISTORICAL_CACHE_PATH)
    check_embedding_frame(historical)
    historical_hashes = set(historical["retrieval_text_sha256"].astype(str))
    production_hashes = set(by_hash)
    reusable = production_hashes & historical_hashes
    missing = production_hashes - historical_hashes
    missing_tokens = sum(len(encoding.encode(str(by_hash[h]["retrieval_text"]))) for h in missing)
    if len(production) != 2100 or len(by_hash) != 1107 or len(reusable) != 848 or len(missing) != 259 or missing_tokens != 22014 or len(unassigned) != 473:
        raise RuntimeError({
            "production": len(production), "unique": len(by_hash), "reusable": len(reusable),
            "missing": len(missing), "tokens": missing_tokens, "unassigned": len(unassigned),
        })
    if len(missing) > MAX_NEW_TEXTS or missing_tokens > MAX_TOKENS:
        raise RuntimeError(f"Phase 6D API safety gate failed: texts={len(missing)} tokens={missing_tokens}")
    query = pd.read_parquet(RETRIEVAL / "phase4b3_query_embeddings.parquet")
    if len(query) != 20:
        raise RuntimeError(f"Expected 20 cached evaluation query embeddings, found {len(query)}")
    if set(query["embedding_model"].astype(str)) != {MODEL} or set(query["embedding_dimension"].astype(int)) != {DIMENSIONS}:
        raise RuntimeError("Cached query embedding metadata is invalid")
    for value in query["embedding"]:
        finite_vector(value)
    return {
        "production_requirement_count": len(production),
        "unique_production_retrieval_texts": len(by_hash),
        "reusable_cached_hashes": len(reusable),
        "missing_new_hashes": len(missing),
        "tokens_requiring_api_embedding": missing_tokens,
        "unassigned_requirements_excluded": len(unassigned),
        "cached_query_embeddings": len(query),
        "query_embedding_api_calls": 0,
        "tokenizer": "cl100k_base",
        "historical_cache_rows": len(historical),
    }


def neo4j_driver() -> Any:
    from neo4j import GraphDatabase
    return GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))


def aura_read_snapshot(driver: Any) -> dict[str, Any]:
    with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
        counts = session.run("""
            CALL { MATCH (n:Document) RETURN 'Document' AS label, count(n) AS value
            UNION ALL MATCH (n:Chapter) RETURN 'Chapter' AS label, count(n) AS value
            UNION ALL MATCH (n:Section) RETURN 'Section' AS label, count(n) AS value
            UNION ALL MATCH (n:Requirement) RETURN 'Requirement' AS label, count(n) AS value }
            RETURN label, value ORDER BY label
        """).data()
        rels = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS value ORDER BY type").data()
        index_rows = session.run("SHOW VECTOR INDEXES YIELD name, state, type, entityType, labelsOrTypes, properties, options RETURN name, state, type, entityType, labelsOrTypes, properties, options").data()
        unassigned = session.run("MATCH (r:Requirement) WHERE r.section_id='section:unassigned' RETURN count(r) AS value").single()["value"]
        embedded = session.run("MATCH (r:Requirement) WHERE r.retrieval_embedding IS NOT NULL RETURN count(r) AS value").single()["value"]
        embedded_unassigned = session.run("MATCH (r:Requirement) WHERE r.section_id='section:unassigned' AND r.retrieval_embedding IS NOT NULL RETURN count(r) AS value").single()["value"]
    index = next((row for row in index_rows if row.get("name") == "hvac_passage_embeddings"), None)
    return {
        "nodes_by_label": {row["label"]: row["value"] for row in counts},
        "relationships_by_type": {row["type"]: row["value"] for row in rels},
        "vector_index": index,
        "unassigned_requirements": unassigned,
        "embedded_requirements": embedded,
        "embedded_unassigned_requirements": embedded_unassigned,
    }


def validate_index(index: dict[str, Any] | None) -> None:
    if not index:
        raise RuntimeError("hvac_passage_embeddings does not exist")
    config = (index.get("options") or {}).get("indexConfig") or {}
    if index.get("state") != "ONLINE":
        raise RuntimeError(f"Vector index is not ONLINE before refresh: {index.get('state')}")
    if index.get("entityType") != "NODE" or index.get("labelsOrTypes") != ["Requirement"] or index.get("properties") != ["retrieval_embedding"]:
        raise RuntimeError("Vector index target changed")
    if int(config.get("vector.dimensions", -1)) != DIMENSIONS or str(config.get("vector.similarity_function", "")).upper() != "COSINE":
        raise RuntimeError(f"Vector index configuration changed: {config}")


def load_checkpoint(expected_missing: set[str], expected_production: set[str]) -> dict[str, dict[str, Any]]:
    if not CURRENT_CACHE_PATH.exists():
        return {}
    checkpoint = pd.read_parquet(CURRENT_CACHE_PATH)
    check_embedding_frame(checkpoint, expected_production)
    rows = {str(row.retrieval_text_sha256): {
        "retrieval_text_sha256": str(row.retrieval_text_sha256),
        "requirement_id": str(row.requirement_id), "embedding_model": str(row.embedding_model),
        "embedding_dimension": int(row.embedding_dimension), "embedding": finite_vector(row.embedding),
    } for row in checkpoint.itertuples(index=False)}
    # The checkpoint is the authoritative current cache and therefore also
    # contains the 848 reused historical rows.  Only its rows in the old
    # missing set count as completed Phase 6D API work.
    unexpected = set(rows) - expected_production
    if unexpected:
        raise RuntimeError(f"Phase 6D checkpoint contains hashes outside the missing set: {len(unexpected)}")
    return {h: row for h, row in rows.items() if h in expected_missing}


def write_current_cache(rows: dict[str, dict[str, Any]]) -> None:
    ordered = [rows[h] for h in sorted(rows)]
    atomic_parquet(pd.DataFrame(ordered, columns=["retrieval_text_sha256", "requirement_id", "embedding_model", "embedding_dimension", "embedding"]), CURRENT_CACHE_PATH)


def embed_missing(by_hash: dict[str, dict[str, Any]], historical: pd.DataFrame, preflight: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    production_hashes = set(by_hash)
    historical_rows = {str(row.retrieval_text_sha256): {
        "retrieval_text_sha256": str(row.retrieval_text_sha256),
        "requirement_id": str(by_hash[str(row.retrieval_text_sha256)]["requirement_id"]),
        "embedding_model": MODEL, "embedding_dimension": DIMENSIONS,
        "embedding": finite_vector(row.embedding),
    } for row in historical.itertuples(index=False) if str(row.retrieval_text_sha256) in production_hashes}
    missing_hashes = production_hashes - set(historical_rows)
    checkpoint = load_checkpoint(missing_hashes, production_hashes)
    rows = dict(historical_rows)
    rows.update(checkpoint)
    write_current_cache(rows)
    usage = {"api_requests": 0, "newly_embedded_texts": 0, "actual_prompt_tokens": 0, "actual_total_tokens": 0, "failed_embeddings": [], "model": MODEL}
    if USAGE_PATH.exists():
        prior = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        if prior.get("model") == MODEL:
            usage.update({key: prior.get(key, value) for key, value in usage.items()})
    already = set(rows) & missing_hashes
    remaining = sorted(missing_hashes - already)
    estimated_submitted_tokens = sum(int(by_hash[h]["token_count"]) for h in already)
    remaining_tokens = sum(int(by_hash[h]["token_count"]) for h in remaining)
    if len(already) + len(remaining) > MAX_NEW_TEXTS or estimated_submitted_tokens + remaining_tokens > MAX_TOKENS:
        raise RuntimeError(f"API safety gate failed: texts={len(already) + len(remaining)} tokens={estimated_submitted_tokens + remaining_tokens}")
    if not remaining:
        cache = pd.DataFrame([rows[h] for h in sorted(rows)])
        check_embedding_frame(cache, production_hashes)
        if len(cache) != len(production_hashes):
            raise RuntimeError("Current cache is incomplete")
        return cache, usage

    # Import and instantiate the OpenAI client only after all local/Aura gates.
    from openai import OpenAI
    client = OpenAI()
    for start in range(0, len(remaining), MAX_BATCH):
        batch_hashes = remaining[start:start + MAX_BATCH]
        batch_tokens = sum(int(by_hash[h]["token_count"]) for h in batch_hashes)
        if estimated_submitted_tokens + batch_tokens > MAX_TOKENS:
            raise RuntimeError("API token safety limit would be exceeded before request")
        texts = [str(by_hash[h]["retrieval_text"]) for h in batch_hashes]
        try:
            response = client.embeddings.create(model=MODEL, input=texts)
            data = sorted(response.data, key=lambda item: int(item.index))
            if len(data) != len(batch_hashes):
                raise RuntimeError(f"Embedding response count {len(data)} != batch size {len(batch_hashes)}")
            for h, item in zip(batch_hashes, data):
                rows[h] = {
                    "retrieval_text_sha256": h,
                    "requirement_id": str(by_hash[h]["requirement_id"]),
                    "embedding_model": MODEL,
                    "embedding_dimension": DIMENSIONS,
                    "embedding": finite_vector(item.embedding),
                }
            usage_obj = response.usage
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens) or prompt_tokens)
            usage["api_requests"] += 1
            usage["newly_embedded_texts"] += len(batch_hashes)
            usage["actual_prompt_tokens"] += prompt_tokens
            usage["actual_total_tokens"] += total_tokens
            estimated_submitted_tokens += batch_tokens
            write_current_cache(rows)
            atomic_json(usage, USAGE_PATH)
        except Exception as exc:
            usage["failed_embeddings"].append({"hashes": batch_hashes, "error": str(exc)})
            atomic_json(usage, USAGE_PATH)
            raise
    cache = pd.DataFrame([rows[h] for h in sorted(rows)])
    check_embedding_frame(cache, production_hashes)
    if len(cache) != len(production_hashes):
        raise RuntimeError(f"Current cache rows {len(cache)} != {len(production_hashes)}")
    return cache, usage


def update_aura(driver: Any, production: pd.DataFrame, by_hash: dict[str, dict[str, Any]], cache: pd.DataFrame) -> dict[str, Any]:
    representatives = {h: by_hash[h] for h in by_hash}
    representative_ids = sorted(str(row["requirement_id"]) for row in representatives.values())
    hash_rows = [{"id": str(row["requirement_id"]), "hash": str(row["retrieval_text_sha256"])} for row in production.to_dict("records")]
    vector_rows = []
    cache_by_hash = {str(row.retrieval_text_sha256): row for row in cache.itertuples(index=False)}
    for h in sorted(representatives):
        row = representatives[h]
        vector_rows.append({
            "id": str(row["requirement_id"]), "hash": h,
            "embedding": finite_vector(cache_by_hash[h].embedding),
        })
    with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
        session.run("""
            MATCH (r:Requirement)
            WHERE NOT r.id IN $representative_ids
            REMOVE r.retrieval_embedding, r.embedding_model, r.is_embedding_representative
        """, representative_ids=representative_ids).consume()
        for start in range(0, len(hash_rows), 500):
            session.run("""
                UNWIND $rows AS row
                MATCH (r:Requirement {id: row.id})
                SET r.retrieval_hash = row.hash
            """, rows=hash_rows[start:start + 500]).consume()
        for start in range(0, len(vector_rows), 25):
            session.run("""
                UNWIND $rows AS row
                MATCH (r:Requirement {id: row.id})
                SET r.retrieval_embedding = row.embedding,
                    r.embedding_model = $model,
                    r.is_embedding_representative = true
            """, rows=vector_rows[start:start + 25], model=MODEL).consume()
    return {"representative_ids": representative_ids, "production_hash_rows": len(hash_rows)}


def wait_for_index(driver: Any, timeout_seconds: int = 120) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    latest = None
    while time.time() < deadline:
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            latest = session.run("SHOW VECTOR INDEXES YIELD name, state, options RETURN name, state, options").data()
        row = next((item for item in latest if item.get("name") == "hvac_passage_embeddings"), None)
        if row and row.get("state") == "ONLINE":
            return row
        time.sleep(1)
    raise RuntimeError(f"Vector index did not become ONLINE: {latest}")


def graph_validate(driver: Any, representative_ids: list[str]) -> dict[str, Any]:
    rep_set = set(representative_ids)
    with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
        labels = session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS value ORDER BY label").data()
        rels = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS value ORDER BY type").data()
        requirement_ids = session.run("MATCH (r:Requirement) RETURN collect(r.id) AS ids").single()["ids"]
        embedded_ids = session.run("MATCH (r:Requirement) WHERE r.retrieval_embedding IS NOT NULL RETURN collect(r.id) AS ids").single()["ids"]
        unassigned_embedded = session.run("MATCH (r:Requirement) WHERE r.section_id='section:unassigned' AND r.retrieval_embedding IS NOT NULL RETURN count(r) AS value").single()["value"]
        production_hash_count = session.run("MATCH (r:Requirement) WHERE r.section_id <> 'section:unassigned' AND r.retrieval_hash IS NOT NULL RETURN count(r) AS value").single()["value"]
        index = session.run("SHOW VECTOR INDEXES YIELD name, state, options RETURN name, state, options").data()
    actual_ids = set(embedded_ids)
    return {
        "nodes_by_label": {row["label"]: row["value"] for row in labels},
        "relationships_by_type": {row["type"]: row["value"] for row in rels},
        "requirement_nodes": len(requirement_ids),
        "production_requirements_with_retrieval_hash": production_hash_count,
        "embedded_requirement_nodes": len(actual_ids),
        "embedded_representatives_expected": len(rep_set),
        "embedded_ids_unexpected": sorted(actual_ids - rep_set),
        "embedded_ids_missing": sorted(rep_set - actual_ids),
        "unassigned_embeddings": unassigned_embedded,
        "vector_index": next((row for row in index if row.get("name") == "hvac_passage_embeddings"), None),
    }


def section_parent_map() -> tuple[dict[str, str | None], dict[str, str]]:
    document = json.loads((OUTPUT / "document.json").read_text(encoding="utf-8"))
    parent = {str(row["id"]): row.get("parent_section_id") for row in document["sections"]}
    number = {str(row["id"]): str(row.get("number")) for row in document["sections"]}
    return parent, number


def evaluation(driver: Any) -> dict[str, Any]:
    query = pd.read_parquet(RETRIEVAL / "phase4b3_query_embeddings.parquet")
    cases = json.loads((OUTPUT / "generation" / "phase5b2_end_to_end.json").read_text(encoding="utf-8"))["cases"]
    parent, number = section_parent_map()
    query_vectors = {str(row.question_id): finite_vector(row.embedding) for row in query.itertuples(index=False)}

    def ancestors(section_id: str) -> set[str]:
        out: set[str] = set()
        current = section_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            section_number = number.get(current)
            if section_number not in (None, "None"):
                out.add(section_number)
            next_parent = parent.get(current)
            if not next_parent or next_parent == "section:unassigned":
                break
            current = str(next_parent)
        return out

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
        expected = {str(value) for value in case.get("expected_sections", [])}
        explicit_match = re.search(r"\bSection\s+(\d+(?:\.\d+)*)", str(case["question"]), re.IGNORECASE)
        hits = vector_hits(query_vectors[qid])
        raw_sections = [str(hit["section_number"]) for hit in hits if hit.get("section_number") is not None]
        raw_sections = list(dict.fromkeys(raw_sections))
        ancestor_sections_by_rank = [ancestors(str(hit["section_id"])) for hit in hits if hit.get("section_id")]
        expanded_top1 = sorted(ancestor_sections_by_rank[0]) if ancestor_sections_by_rank else []
        expanded_top5 = sorted(set().union(*ancestor_sections_by_rank[:5])) if ancestor_sections_by_rank else []
        if explicit_match:
            exact_number = explicit_match.group(1)
            router_sections = [exact_number]
            router_route = "exact-section"
        else:
            router_sections = expanded_top5
            router_route = "vector-plus-ancestors"
        rows.append({
            "question_id": qid,
            "expected_sections": sorted(expected),
            "explicit_section": explicit_match.group(1) if explicit_match else None,
            "raw_vector_sections": raw_sections,
            "raw_vector_top1_hit": bool(expected & set(raw_sections[:1])),
            "raw_vector_top5_hit": bool(expected & set(raw_sections[:5])),
            "ancestor_expanded_top1_sections": expanded_top1,
            "ancestor_expanded_top5_sections": expanded_top5,
            "ancestor_expanded_top1_hit": bool(expected & set(expanded_top1)),
            "ancestor_expanded_top5_hit": bool(expected & set(expanded_top5)),
            "router_route": router_route,
            "router_sections": router_sections,
            "router_top1_hit": bool(expected & set(router_sections[:1])) if router_route == "exact-section" else bool(expected & set(router_sections)),
            "router_top5_hit": bool(expected & set(router_sections[:5])) if router_route == "exact-section" else bool(expected & set(router_sections)),
            "expected_section_evidence_present": bool(expected & set(router_sections)),
        })
    previous = {str(case["question_id"]): set(str(v) for v in case.get("expected_sections", [])) & set(str(v) for v in case.get("retrieved_sections", [])) for case in cases}
    regressions = [row["question_id"] for row in rows if previous.get(row["question_id"]) and not row["expected_section_evidence_present"]]
    def rate(key: str) -> dict[str, Any]:
        count = sum(bool(row[key]) for row in rows)
        return {"hits": count, "questions": len(rows), "rate": count / len(rows)}
    return {
        "question_count": len(rows),
        "raw_vector_top1": rate("raw_vector_top1_hit"),
        "raw_vector_top5": rate("raw_vector_top5_hit"),
        "ancestor_expanded_vector_top1": rate("ancestor_expanded_top1_hit"),
        "ancestor_expanded_vector_top5": rate("ancestor_expanded_top5_hit"),
        "deterministic_router_top1": rate("router_top1_hit"),
        "deterministic_router_top5": rate("router_top5_hit"),
        "expected_section_evidence_coverage": rate("expected_section_evidence_present"),
        "regressions_in_expected_evidence_coverage": regressions,
        "cases": rows,
        "comparison_note": "Regression check compares expected-section coverage with the frozen Phase 5B2 retrieved-section evidence; no stale pre-refresh vector score artifact was persisted locally.",
    }


def run(preflight_only: bool = False) -> dict[str, Any]:
    load_env()
    local = preflight_local()
    driver = neo4j_driver()
    try:
        driver.verify_connectivity()
        before = aura_read_snapshot(driver)
        validate_index(before["vector_index"])
        report: dict[str, Any] = {"phase": "6D", "preflight": local, "aura_before": before, "openai_calls": 0, "aura_writes": 0}
        if preflight_only:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return report
        production, _, by_hash, _ = load_manifest()
        historical = pd.read_parquet(HISTORICAL_CACHE_PATH)
        cache, usage = embed_missing(by_hash, historical, local)
        report["embedding"] = usage
        report["embedding"]["estimated_cost_usd"] = usage["actual_total_tokens"] / 1_000_000 * EMBEDDING_RATE
        report["embedding"]["cache_rows"] = len(cache)
        report["embedding"]["embedding_dimension"] = DIMENSIONS
        update = update_aura(driver, production, by_hash, cache)
        report["aura_writes"] = 1
        report["aura_update"] = {"production_requirements_updated": update["production_hash_rows"], "representatives_updated": len(update["representative_ids"]), "stale_vector_properties_removed_from_nonrepresentatives": True}
        report["vector_index_after_refresh"] = wait_for_index(driver)
        report["aura_after"] = graph_validate(driver, update["representative_ids"])
        report["evaluation"] = evaluation(driver)
        report["unassigned_embeddings_remaining"] = report["aura_after"]["unassigned_embeddings"]
        report["production_representative_count"] = report["aura_after"]["embedded_representatives_expected"]
        atomic_json(report, REPORT_PATH)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return report
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
