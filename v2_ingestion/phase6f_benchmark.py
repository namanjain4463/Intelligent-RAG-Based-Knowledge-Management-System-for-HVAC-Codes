"""Phase 6F frozen grounded generation benchmark.

This utility reads the frozen Phase 6E retrieval artifact and the repaired
canonical structural corpus.  It deliberately has no Neo4j or embedding
write path.  The preflight mode is entirely local; the run mode instantiates
the OpenAI client only after all local evidence and token gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from .evidence import (allowed_block_text, ancestor_numbers, build_exception_fallbacks, clean_section_title, json_value, known_heading_matches, load_json, norm_text, own_section_records, page_from, referenced_table_numbers, section_number_from_id, section_page, table_records, trim_to_owned_span)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "v2_output"
RETRIEVAL = OUTPUT / "retrieval"
GENERATION = OUTPUT / "generation"
DOC_PATH = OUTPUT / "document.json"
BENCHMARK_PATH = RETRIEVAL / "phase6e_corrected_benchmark.json"
PRODUCTION_MANIFEST_PATH = RETRIEVAL / "phase6c_production_embedding_manifest.parquet"
ARTIFACT_PATH = GENERATION / "phase6f_final_benchmark.json"

MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
MAX_CALLS = 20
MAX_ESTIMATED_INPUT_TOKENS = 25_000
INPUT_RATE = 0.20
OUTPUT_RATE = 1.20

SECTION_REF_RE = re.compile(r"\bSection\s+(\d{3,4}(?:\.\d+)+)", re.I)
ANY_SECTION_HEADING_RE = re.compile(r"(?<![A-Za-z0-9])(?P<number>\d{3,4}(?:\.\d+)+)(?=\s+[A-Z\[])" )
EVIDENCE_ID_RE = re.compile(r"\[(E\d+)\]")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")

GENERATION_INSTRUCTIONS = """You answer HVAC code questions using only the supplied canonical code evidence.
Answer the user's question directly and concisely, normally in 200 words or fewer.
Never invent a requirement or rely on knowledge outside the supplied evidence.
Preserve all material qualifications, conditions, exceptions, and permissions.
If the supplied evidence genuinely cannot answer the question, say so explicitly.
Cite every substantive regulatory statement using one or more supplied evidence identifiers
in the exact form [E1], [E2]. Use only identifiers that appear in the supplied evidence.
Never generate section-number/page-number citations yourself and never use [Section ...] citations.
Do not claim that the supplied context is the complete code unless the evidence supports that claim."""

# Frozen substring anchors expose clause-level coverage separately from the
# benchmark's section-ID intersection metric.
FROZEN_CLAUSE_ANCHORS: dict[str, tuple[str, ...]] = {
    "numeric-01": ("not less than 3 inches (76 mm) above the pit floor",),
    "numeric-02": (
        "Holes bored in joists shall not be within 2 inches (51 mm) of the top or bottom of the joist",
    ),
    "prohibition-02": (
        "Exposed within a fire-resistance-rated exit access corridor.",
        "Within an interior exit stairway.",
        "Within an interior exit ramp.",
        "Within an exit passageway.",
        "Within an elevator, dumbwaiter or other shaft containing a moving object.",
    ),
    "prohibition-04": (
        "Multiple fans or multispeed fans shall be allowed to produce the emergency ventilation rate",
    ),
}


def clause_level_evidence_present(question_id: str, evidence: Any) -> bool | None:
    """Check frozen source anchors without changing relevance labels."""

    anchors = FROZEN_CLAUSE_ANCHORS.get(str(question_id))
    if not anchors:
        return None
    if isinstance(evidence, str):
        text = evidence
    else:
        text = "\n".join(norm_text(item.get("source_text")) for item in evidence or [])
    lowered = text.casefold()
    return all(anchor.casefold() in lowered for anchor in anchors)












def load_tokenizer():
    # The cache is repository-local and was populated in the prior Phase 4B1
    # run.  Set it before importing tiktoken so this path cannot contact the
    # network while calculating the preflight estimate.
    cache_dir = ROOT / ".cache" / "tiktoken"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    import tiktoken

    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    if encoding.name != "cl100k_base":
        raise RuntimeError(f"Expected cl100k_base tokenizer, got {encoding.name}")
    return encoding


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    document = load_json(DOC_PATH)
    benchmark = load_json(BENCHMARK_PATH)
    manifest = pd.read_parquet(PRODUCTION_MANIFEST_PATH)
    requirement_to_hash = {
        str(row.requirement_id): str(row.retrieval_text_sha256)
        for row in manifest.itertuples(index=False)
    }
    return document, benchmark, requirement_to_hash






















def build_case_evidence(
    case: dict[str, Any],
    sections_by_number: dict[str, dict[str, Any]],
    sections_by_id: dict[str, dict[str, Any]],
    table_by_id: dict[str, dict[str, Any]],
    section_numbers: set[str],
    exception_fallbacks: dict[str, str],
    id_counter: list[int],
    requirement_to_hash: dict[str, str],
) -> dict[str, Any]:
    if case.get("route") == "exact-section":
        explicit = SECTION_REF_RE.findall(str(case.get("question") or ""))
        direct_numbers = [explicit[0] if explicit else str(case["expected_sections"][0])]
        if direct_numbers[0] not in sections_by_number:
            raise RuntimeError(f"Frozen exact-section target missing from corpus: {direct_numbers[0]}")
    else:
        seen_hashes: set[str] = set()
        direct_numbers = []
        for hit in case.get("raw_vector_hits") or []:
            rid = str(hit.get("requirement_id") or "")
            h = requirement_to_hash.get(rid) or f"requirement:{rid}"
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            number = str(hit.get("section_number") or "")
            if number in sections_by_number and number != "unassigned":
                direct_numbers.append(number)
            if len(direct_numbers) == 3:
                break
    if not direct_numbers:
        raise RuntimeError(f"No direct evidence sections for {case['question_id']}")

    selected_numbers: list[str] = []
    for number in direct_numbers:
        if number not in selected_numbers:
            selected_numbers.append(number)
        section = sections_by_number[number]
        for ancestor in ancestor_numbers(section, sections_by_id):
            if ancestor not in selected_numbers:
                selected_numbers.append(ancestor)

    records: list[dict[str, Any]] = []
    direct_set = set(direct_numbers)
    for number in selected_numbers:
        section = sections_by_number[number]
        own = own_section_records(section, section_numbers, exception_fallbacks)
        records.extend(own)
        # Tables are included only for directly retrieved sections whose
        # canonical section structure explicitly references them.
        if number in direct_set:
            for table_number in referenced_table_numbers(section, own, table_by_id):
                table = next((t for t in table_by_id.values() if str(t.get("table_number")) == table_number), None)
                if table:
                    records.extend(table_records(table, number, clean_section_title(section.get("title"))))

    deduped: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for record in records:
        text = norm_text(record.get("source_text"))
        if not text:
            continue
        key = text.casefold()
        if key in seen_text:
            continue
        seen_text.add(key)
        record = dict(record)
        evidence_id = f"E{id_counter[0]}"
        id_counter[0] += 1
        record["evidence_id"] = evidence_id
        record["source_text"] = text
        deduped.append(record)

    lines = [
        "Supplied canonical HVAC evidence. Cite only the bracketed evidence identifiers.",
    ]
    for record in deduped:
        page = f", p. {record['page']}" if record.get("page") is not None else ""
        lines.append(f"[{record['evidence_id']}] Section {record['section_number']}{page}: {record['source_text']}")
    lines.append("")
    lines.append(f"Question: {case['question']}")
    return {
        "question_id": case["question_id"],
        "question": case["question"],
        "expected_sections": [str(x) for x in case.get("expected_sections") or []],
        "route": case.get("route"),
        "direct_retrieved_sections": direct_numbers,
        "retrieved_sections": selected_numbers,
        "evidence_blocks": deduped,
        "evidence_supplied": "\n".join(lines),
        "evidence_id_map": {
            r["evidence_id"]: {
                "section_number": r["section_number"],
                "page": r.get("page"),
                "source_text": r["source_text"],
            }
            for r in deduped
        },
    }


def assemble_preflight() -> dict[str, Any]:
    document, benchmark, requirement_to_hash = load_frozen_inputs()
    sections = [
        s for s in document.get("sections", [])
        if re.fullmatch(r"\d{3,4}(?:\.\d+)*", str(s.get("number") or ""))
    ]
    sections_by_number = {str(s["number"]): s for s in sections}
    sections_by_id = {str(s["id"]): s for s in sections}
    section_numbers = set(sections_by_number)
    table_by_id = {str(t["id"]): t for t in document.get("tables", [])}
    exception_fallbacks = build_exception_fallbacks()
    tokenizer = load_tokenizer()

    id_counter = [1]
    cases: list[dict[str, Any]] = []
    for source_case in benchmark["evaluation"]["cases"]:
        case = build_case_evidence(
            source_case,
            sections_by_number,
            sections_by_id,
            table_by_id,
            section_numbers,
            exception_fallbacks,
            id_counter,
            requirement_to_hash,
        )
        prompt_text = GENERATION_INSTRUCTIONS + "\n\n" + case["evidence_supplied"]
        case["estimated_input_tokens"] = len(tokenizer.encode(prompt_text))
        case["expected_section_present"] = all(
            expected in set(case["retrieved_sections"]) for expected in case["expected_sections"]
        )
        case["clause_level_evidence_present"] = clause_level_evidence_present(
            case["question_id"], case["evidence_blocks"]
        )
        case["evidence_block_count"] = len(case["evidence_blocks"])
        cases.append(case)

    total_estimated = sum(int(c["estimated_input_tokens"]) for c in cases)
    numeric = next(c for c in cases if c["question_id"] == "numeric-02")
    p02 = next(c for c in cases if c["question_id"] == "prohibition-02")
    p04 = next(c for c in cases if c["question_id"] == "prohibition-04")
    all_text = "\n".join(r["source_text"] for c in cases for r in c["evidence_blocks"])
    five_locations = [
        "Exposed within a fire-resistance-rated exit access corridor.",
        "Within an interior exit stairway.",
        "Within an interior exit ramp.",
        "Within an exit passageway.",
        "Within an elevator, dumbwaiter or other shaft containing a moving object.",
    ]
    p02_text = "\n".join(r["source_text"] for r in p02["evidence_blocks"])
    p04_text = "\n".join(r["source_text"] for r in p04["evidence_blocks"])
    preflight = {
        "tokenizer": "cl100k_base",
        "question_count": len(cases),
        "total_estimated_input_tokens": total_estimated,
        "max_estimated_input_tokens": max(int(c["estimated_input_tokens"]) for c in cases),
        "mean_estimated_input_tokens": total_estimated / len(cases),
        "expected_section_coverage": sum(bool(c["expected_section_present"]) for c in cases),
        "clause_level_evidence_coverage": sum(
            bool(c["clause_level_evidence_present"])
            for c in cases
            if c["clause_level_evidence_present"] is not None
        ),
        "all_questions_have_evidence": all(bool(c["evidence_blocks"]) for c in cases),
        "numeric-02_section_302.3.1_present": "302.3.1" in numeric["retrieved_sections"] and "Holes bored in joists" in all_text,
        "prohibition-02_five_locations_present": all(location in p02_text for location in five_locations),
        "prohibition-04_multiple_fan_clause_present": "Multiple fans or multispeed fans shall be allowed to produce the emergency ventilation rate" in p04_text,
        "gate_passed": (
            len(cases) == 20
            and all(bool(c["evidence_blocks"]) for c in cases)
            and all(bool(c["expected_section_present"]) for c in cases)
            and "302.3.1" in numeric["retrieved_sections"]
            and "Holes bored in joists" in all_text
            and all(location in p02_text for location in five_locations)
            and "Multiple fans or multispeed fans shall be allowed to produce the emergency ventilation rate" in p04_text
            and total_estimated <= MAX_ESTIMATED_INPUT_TOKENS
        ),
    }
    return {
        "phase": "6F",
        "status": "preflight",
        "frozen_source": str(BENCHMARK_PATH.relative_to(ROOT)),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "api_calls": 0,
        "preflight": preflight,
        "cases": cases,
        "cost_rates": {"input_usd_per_million": INPUT_RATE, "output_usd_per_million": OUTPUT_RATE},
    }


def save_artifact(artifact: dict[str, Any]) -> None:
    GENERATION.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    def value(name: str):
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)
    return value("input_tokens"), value("output_tokens")


def output_text(response: Any) -> str:
    value = getattr(response, "output_text", None)
    if value:
        return str(value)
    output = getattr(response, "output", None) or []
    pieces: list[str] = []
    for item in output:
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        for part in content or []:
            text = getattr(part, "text", None) if not isinstance(part, dict) else part.get("text")
            if text:
                pieces.append(str(text))
    return "\n".join(pieces)


def validate_case(case: dict[str, Any], raw_answer: str) -> dict[str, Any]:
    evidence_map = case.get("evidence_id_map") or {}
    bracket_values = BRACKET_RE.findall(raw_answer)
    cited_ids = EVIDENCE_ID_RE.findall(raw_answer)
    unsupported: list[str] = []
    malformed: list[str] = []
    for value in bracket_values:
        if value.startswith("E") and not re.fullmatch(r"E\d+", value):
            malformed.append(value)
        elif value.startswith("E") and value not in evidence_map:
            unsupported.append(value)
    for evidence_id in cited_ids:
        if evidence_id not in evidence_map and evidence_id not in unsupported:
            unsupported.append(evidence_id)
    rendered = raw_answer
    for evidence_id in cited_ids:
        if evidence_id in evidence_map:
            item = evidence_map[evidence_id]
            page = f", p. {item['page']}" if item.get("page") is not None else ""
            rendered = rendered.replace(f"[{evidence_id}]", f"[Section {item['section_number']}{page}]")
    has_section_page_citation = bool(re.search(r"\[Section\s+\d", raw_answer, re.I))
    rendered_sections = re.findall(r"\[Section\s+(\d{3,4}(?:\.\d+)+)", rendered, re.I)
    expected = set(case.get("expected_sections") or [])
    insufficient = bool(re.search(r"\b(insufficient evidence|not enough information|cannot determine|unable to answer)\b", raw_answer, re.I))
    return {
        "cited_evidence_ids": cited_ids,
        "unsupported_evidence_ids": sorted(set(unsupported)),
        "malformed_or_forbidden_citations": malformed,
        "citation_format_violation": has_section_page_citation,
        "rendered_answer": rendered,
        "citation_valid": bool(cited_ids) and not unsupported and not malformed and not has_section_page_citation,
        "cites_expected_section": bool(expected.intersection(rendered_sections)),
        "inappropriate_insufficient_evidence": insufficient and bool(case.get("expected_section_present")),
    }


def load_or_create_artifact() -> dict[str, Any]:
    if ARTIFACT_PATH.exists():
        artifact = load_json(ARTIFACT_PATH)
        if artifact.get("phase") == "6F" and artifact.get("preflight"):
            return artifact
    artifact = assemble_preflight()
    save_artifact(artifact)
    return artifact


def run_generation() -> int:
    artifact = load_or_create_artifact()
    preflight = artifact.get("preflight") or {}
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if not preflight.get("gate_passed"):
        print("STOP: Phase 6F preflight gate failed; zero API calls made.")
        return 2
    cases = artifact.get("cases") or []
    if len(cases) != 20:
        raise RuntimeError(f"Expected 20 cases, found {len(cases)}")

    completed = sum(bool(c.get("completed")) for c in cases)
    if completed >= MAX_CALLS:
        return 0

    # Load credentials only after the zero-cost gate.  This does not contact
    # the API; the client is also instantiated only after this point.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    from openai import OpenAI
    client = OpenAI()

    calls = int(artifact.get("api_calls") or 0)
    for case in cases:
        if case.get("completed"):
            continue
        if calls >= MAX_CALLS:
            break
        try:
            response = client.responses.create(
                model=MODEL,
                reasoning={"effort": REASONING_EFFORT},
                instructions=GENERATION_INSTRUCTIONS,
                input=case["evidence_supplied"],
                max_output_tokens=450,
                store=False,
            )
            raw = output_text(response)
            input_tokens, output_tokens = extract_usage(response)
            validation = validate_case(case, raw)
            case.update({
                "completed": True,
                "raw_answer": raw,
                "generated_answer": validation["rendered_answer"],
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
                "model": MODEL,
                "response_model": getattr(response, "model", MODEL),
                "citation_validation": validation,
                "api_error": None,
            })
            calls += 1
            artifact["api_calls"] = calls
            artifact["status"] = "in_progress"
            save_artifact(artifact)
            print(f"completed {case['question_id']} input={input_tokens} output={output_tokens}")
        except Exception as exc:  # checkpoint failures without retrying the case
            case.update({"completed": False, "api_error": str(exc), "model": MODEL})
            calls += 1
            artifact["api_calls"] = calls
            artifact["status"] = "in_progress"
            save_artifact(artifact)
            print(f"failed {case['question_id']}: {exc}")

    complete_cases = [c for c in cases if c.get("completed")]
    artifact["status"] = "completed" if len(complete_cases) == 20 else "partial"
    artifact["questions_answered"] = len(complete_cases)
    artifact["total_input_tokens"] = sum(int(c.get("actual_input_tokens") or 0) for c in complete_cases)
    artifact["total_output_tokens"] = sum(int(c.get("actual_output_tokens") or 0) for c in complete_cases)
    artifact["estimated_api_cost_usd"] = (
        artifact["total_input_tokens"] / 1_000_000 * INPUT_RATE
        + artifact["total_output_tokens"] / 1_000_000 * OUTPUT_RATE
    )
    artifact["citation_valid_answers"] = sum(bool(c.get("citation_validation", {}).get("citation_valid")) for c in complete_cases)
    artifact["answers_citing_expected_section"] = sum(bool(c.get("citation_validation", {}).get("cites_expected_section")) for c in complete_cases)
    artifact["inappropriate_insufficient_evidence_answers"] = sum(bool(c.get("citation_validation", {}).get("inappropriate_insufficient_evidence")) for c in complete_cases)
    artifact["unsupported_evidence_ids"] = {
        c["question_id"]: c.get("citation_validation", {}).get("unsupported_evidence_ids", [])
        for c in complete_cases
        if c.get("citation_validation", {}).get("unsupported_evidence_ids")
    }
    save_artifact(artifact)
    return 0 if len(complete_cases) == 20 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="assemble and save local preflight only")
    parser.add_argument("--run", action="store_true", help="run the gated Luna benchmark")
    args = parser.parse_args()
    if not args.preflight and not args.run:
        parser.error("choose --preflight or --run")
    if args.preflight:
        artifact = assemble_preflight()
        save_artifact(artifact)
        print(json.dumps(artifact["preflight"], indent=2, ensure_ascii=False))
        for case in artifact["cases"]:
            print(f"{case['question_id']}: {case['estimated_input_tokens']} tokens, sections={case['retrieved_sections']}, evidence_blocks={case['evidence_block_count']}")
        return 0 if artifact["preflight"]["gate_passed"] else 2
    return run_generation()


if __name__ == "__main__":
    raise SystemExit(main())
