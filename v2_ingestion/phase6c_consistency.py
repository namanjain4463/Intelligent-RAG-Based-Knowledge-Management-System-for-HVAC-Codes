"""Phase 6C read-only retrieval consistency gate.

This utility rebuilds local manifests from the repaired canonical corpus.  It
does not import OpenAI, write to Aura, change ownership, or alter retrieval
artifacts other than the two explicitly named Phase 6C manifests and report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "v2_output"
AUDIT = OUTPUT / "audit"
RETRIEVAL = OUTPUT / "retrieval"
TIKTOKEN_DIR = ROOT / ".cache" / "tiktoken"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_DIR))
import tiktoken  # noqa: E402


def parse_json(value: Any, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> list[str]:
    return re.findall(r"[\w]+", unicodedata.normalize("NFKC", str(value or "")).lower())


def source_text(row: pd.Series) -> str:
    evidence = parse_json(row.get("evidence"), {}) or {}
    return norm(evidence.get("source_text"))


def source_block_id(row: pd.Series) -> str:
    evidence = parse_json(row.get("evidence"), {}) or {}
    return str(evidence.get("source_block_id") or row.get("source_block_id") or "")


def base_block_id(value: str) -> str:
    prefix = "unparsed:phase6b-ambiguous:"
    return value[len(prefix):] if value.startswith(prefix) else value


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


def text_match(needle: str, haystack: list[str]) -> bool:
    """Match a supplied source passage without relying on raw line equality."""
    n = norm(needle)
    if not n:
        return False
    nt = tokens(n)
    for item in haystack:
        h = norm(item)
        if n in h:
            return True
        ht = tokens(h)
        if len(nt) >= 12 and len(ht) >= len(nt):
            # Evidence may contain page labels between otherwise identical
            # source text.  Require near-complete token inclusion for the
            # individual source string, not the concatenated evaluation file.
            from collections import Counter
            need = Counter(nt)
            have = Counter(ht)
            covered = sum(min(count, have[token]) for token, count in need.items())
            if covered / len(nt) >= 0.995:
                return True
    return False


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, dict[str, dict[str, Any]], set[str]]:
    document = json.loads((OUTPUT / "document.json").read_text(encoding="utf-8"))
    requirements = pd.read_parquet(OUTPUT / "semantic" / "requirements.parquet")
    sections = {str(row["id"]): row for row in document.get("sections", [])}
    report = json.loads((AUDIT / "phase6b_repair_report.json").read_text(encoding="utf-8"))
    ambiguous_ids = {str(row["requirement_id"]) for row in report["unresolved_requirements"]}
    if len(ambiguous_ids) != 472:
        raise RuntimeError(f"Expected 472 Phase 6B ambiguous Requirement IDs, found {len(ambiguous_ids)}")
    production = requirements[~requirements["needs_review"].astype(bool)].copy()
    if len(production) != 2573:
        raise RuntimeError(f"Expected 2573 production Requirements, found {len(production)}")
    return document, production, sections, ambiguous_ids


def build_manifest_rows(
    production: pd.DataFrame,
    sections: dict[str, dict[str, Any]],
    encoding: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    for _, row in production.iterrows():
        rid = str(row["requirement_id"])
        sid = str(row["section_id"])
        section = sections.get(sid)
        if section is None:
            raise RuntimeError(f"Requirement {rid} references missing Section {sid}")
        section_number = section.get("number")
        section_title = section.get("title")
        retrieval = f"Section {section_number}: {section_title}\n{source_text(row)}"
        record = {
            "requirement_id": rid,
            "section_id": sid,
            "section_number": section_number,
            "section_title": section_title,
            "retrieval_text": retrieval,
            "retrieval_text_sha256": hashlib.sha256(retrieval.encode("utf-8")).hexdigest(),
            "token_count": len(encoding.encode(retrieval)),
            "source_block_id": source_block_id(row),
            "source_text": source_text(row),
        }
        (unassigned if sid == "section:unassigned" else real).append(record)
    return real, unassigned


def unique_by_hash(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        previous = out.get(row["retrieval_text_sha256"])
        if previous and previous["retrieval_text"] != row["retrieval_text"]:
            raise RuntimeError("One retrieval hash maps to conflicting retrieval text")
        if previous is None or row["requirement_id"] < previous["requirement_id"]:
            out[row["retrieval_text_sha256"]] = row
    return out


def eval_dependency(
    ambiguous_rows: list[dict[str, Any]],
    sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    phase4 = json.loads((RETRIEVAL / "phase4a_evaluation.json").read_text(encoding="utf-8"))
    phase5 = json.loads((OUTPUT / "generation" / "phase5b2_end_to_end.json").read_text(encoding="utf-8"))
    phase4_strings = list(walk_strings(phase4))
    phase5_strings = list(walk_strings(phase5))
    phase5_evidence = [str(case.get("evidence_supplied") or "") for case in phase5["cases"]]

    appeared_ids: set[str] = set()
    luna_ids: set[str] = set()
    appeared_passages: set[str] = set()
    luna_passages: set[str] = set()
    for row in ambiguous_rows:
        rid = row["requirement_id"]
        base_id = base_block_id(row["source_block_id"])
        if any(rid in item or (base_id and base_id in item) for item in phase4_strings + phase5_strings):
            appeared_ids.add(rid)
            appeared_passages.add(row["retrieval_text_sha256"])
        if text_match(row["source_text"], phase5_evidence):
            luna_ids.add(rid)
            luna_passages.add(row["retrieval_text_sha256"])

    case_dependencies: list[dict[str, Any]] = []
    for case in phase5["cases"]:
        evidence = str(case.get("evidence_supplied") or "")
        expected = [str(value) for value in case.get("expected_sections", [])]
        potential_unassigned = [
            row for row in ambiguous_rows if text_match(row["source_text"], [evidence])
        ]
        expected_body_present = []
        for number in expected:
            section = next((s for s in sections.values() if str(s.get("number")) == number), None)
            body_present = False
            if section:
                for block in section.get("blocks", []):
                    block_text = norm(block.get("text"))
                    if block_text and text_match(block_text, [evidence]):
                        body_present = True
                        break
            expected_body_present.append({"section": number, "canonical_body_present": body_present})
        exclusive = bool(potential_unassigned) and not all(item["canonical_body_present"] for item in expected_body_present)
        case_dependencies.append({
            "question_id": case.get("question_id"),
            "expected_sections": expected,
            "unassigned_requirements_in_supplied_evidence": len({r["requirement_id"] for r in potential_unassigned}),
            "expected_canonical_body_present": expected_body_present,
            "depends_on_unassigned": exclusive,
        })
    return {
        "phase4_5_appeared_requirement_count": len(appeared_ids),
        "phase4_5_appeared_unique_passage_count": len(appeared_passages),
        "luna_supplied_requirement_count": len(luna_ids),
        "luna_supplied_unique_passage_count": len(luna_passages),
        "expected_answer_dependency": case_dependencies,
        "expected_answers_depending_on_unassigned": sum(1 for item in case_dependencies if item["depends_on_unassigned"]),
    }


def local_valid_retrieval_dependency(
    real_rows: list[dict[str, Any]],
    sections: dict[str, dict[str, Any]],
    phase5: dict[str, Any],
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Inspect the frozen 20 mappings using only current real-section hashes
    that still have a cached embedding.  This is a dependency check, not an
    accuracy rerun.
    """
    unique = unique_by_hash(real_rows)
    cache_by_hash = {
        str(row.retrieval_text_sha256): np.asarray(row.embedding, dtype=np.float32)
        for row in cache.itertuples(index=False)
    }
    valid = [row for h, row in unique.items() if h in cache_by_hash]
    if valid:
        matrix = np.vstack([cache_by_hash[row["retrieval_text_sha256"]] for row in valid])
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1
        matrix = matrix / norms[:, None]
    else:
        matrix = np.empty((0, 3072), dtype=np.float32)
    query = pd.read_parquet(RETRIEVAL / "phase4b3_query_embeddings.parquet")
    query_by_id = {str(row.question_id): np.asarray(row.embedding, dtype=np.float32) for row in query.itertuples(index=False)}

    parent = {str(s["id"]): s.get("parent_section_id") for s in sections.values()}
    number_by_id = {str(s["id"]): str(s.get("number")) for s in sections.values()}
    results: list[dict[str, Any]] = []
    for case in phase5["cases"]:
        qid = str(case["question_id"])
        expected = {str(value) for value in case.get("expected_sections", [])}
        explicit = bool(re.search(r"\bSection\s+\d+(?:\.\d+)*", str(case.get("question") or ""), flags=re.IGNORECASE))
        direct = [str(value) for value in case.get("expected_sections", [])[:1]] if explicit else []
        if direct:
            evidence_sections = set(direct)
            route = "exact-section"
        elif qid in query_by_id and len(valid):
            vector = query_by_id[qid]
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            scores = matrix @ vector
            order = np.argsort(-scores)[:10]
            hit_ids = [valid[int(index)]["section_id"] for index in order]
            evidence_sections = {number_by_id.get(sid, sid) for sid in hit_ids}
            # Ancestor expansion is identity-only; it does not affect ranking.
            for sid in hit_ids:
                current = sid
                while parent.get(current):
                    current = str(parent[current])
                    evidence_sections.add(number_by_id.get(current, current))
            route = "vector-valid-cache-only"
        else:
            evidence_sections = set()
            route = "vector-no-valid-cache"
        results.append({
            "question_id": qid,
            "route": route,
            "expected_sections": sorted(expected),
            "expected_section_present_in_current_real_mapping": bool(expected & evidence_sections),
            "current_evidence_sections": sorted(evidence_sections),
        })
    return {
        "questions": results,
        "questions_with_expected_mapping": sum(bool(row["expected_section_present_in_current_real_mapping"]) for row in results),
        "note": "Dependency inspection only; stale cached hashes were excluded and no retrieval accuracy is reported.",
    }


def main() -> None:
    document, production, sections, ambiguous_ids = load_inputs()
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    if encoding.name != "cl100k_base":
        raise RuntimeError(f"Unexpected tokenizer: {encoding.name}")
    real_rows, unassigned_rows = build_manifest_rows(production, sections, encoding)
    if {row["requirement_id"] for row in unassigned_rows} != ambiguous_ids | ({row["requirement_id"] for row in unassigned_rows} - ambiguous_ids):
        raise RuntimeError("Unexpected unassigned production rows")
    # Keep the exact 472 Phase 6B ambiguous rows distinct from the one older
    # unassigned production row for the requested audit.
    ambiguous_rows = [row for row in unassigned_rows if row["requirement_id"] in ambiguous_ids]
    if len(ambiguous_rows) != 472:
        raise RuntimeError(f"Expected 472 ambiguous rows, found {len(ambiguous_rows)}")

    manifest_columns = [
        "requirement_id", "section_id", "section_number", "section_title",
        "retrieval_text", "retrieval_text_sha256", "token_count",
    ]
    pd.DataFrame(real_rows)[manifest_columns].to_parquet(RETRIEVAL / "phase6c_production_embedding_manifest.parquet", index=False)
    pd.DataFrame(unassigned_rows)[manifest_columns].to_parquet(RETRIEVAL / "phase6c_unassigned_embedding_manifest.parquet", index=False)

    cache = pd.read_parquet(RETRIEVAL / "embedding_cache.parquet")
    cache_hashes = set(cache["retrieval_text_sha256"].astype(str))
    real_unique = unique_by_hash(real_rows)
    real_hashes = set(real_unique)
    reusable = real_hashes & cache_hashes
    missing = real_hashes - cache_hashes
    orphaned = cache_hashes - real_hashes
    missing_tokens = sum(int(real_unique[h]["token_count"]) for h in missing)

    phase5 = json.loads((OUTPUT / "generation" / "phase5b2_end_to_end.json").read_text(encoding="utf-8"))
    dependency = eval_dependency(ambiguous_rows, sections)
    retrieval_dependency = local_valid_retrieval_dependency(real_rows, sections, phase5, cache)
    delta = pd.read_parquet(RETRIEVAL / "phase6b_embedding_delta.parquet")
    delta_real = delta[delta["new_section_id"].astype(str) != "section:unassigned"]
    delta_real_unique = delta_real.drop_duplicates("retrieval_text_sha256")

    report = {
        "phase": "6C",
        "openai_calls": 0,
        "aura_writes": 0,
        "unassigned_audit": {
            "ambiguous_requirement_count": len(ambiguous_rows),
            "unique_source_passages": len({row["source_text"] for row in ambiguous_rows}),
            "unique_retrieval_hashes": len({row["retrieval_text_sha256"] for row in ambiguous_rows}),
            "embedding_counts_from_aura": "reported separately from read-only Aura inspection",
            **dependency,
        },
        "repaired_manifest": {
            "production_real_section_requirement_count": len(real_rows),
            "section_unassigned_requirement_count": len(unassigned_rows),
            "unique_production_retrieval_texts": len(real_hashes),
            "embeddings_reusable_unchanged_unique_hashes": len(reusable),
            "embeddings_missing_because_hashes_changed_unique_hashes": len(missing),
            "cached_embeddings_orphaned_or_stale_unique_hashes": len(orphaned),
            "new_unique_token_count_requiring_embedding": missing_tokens,
            "phase6b_all_estimate": {"unique_passages": 315, "tokens": 36851},
            "phase6b_delta_excluding_unassigned": {"unique_passages": len(delta_real_unique), "tokens": int(delta_real_unique["token_count"].sum())},
            "recomputed_manifest_missing_hashes": {"unique_passages": len(missing), "tokens": missing_tokens},
        },
        "retrieval_dependency": retrieval_dependency,
        "manifest_paths": {
            "production": str(RETRIEVAL / "phase6c_production_embedding_manifest.parquet"),
            "unassigned": str(RETRIEVAL / "phase6c_unassigned_embedding_manifest.parquet"),
        },
    }
    (AUDIT / "phase6c_consistency_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
