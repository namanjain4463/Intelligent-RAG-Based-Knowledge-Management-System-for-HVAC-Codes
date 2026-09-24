"""Frozen-v2 ReAct GraphRAG runtime.

This module is intentionally separate from the structural and semantic
pipelines. It reads the repaired Aura graph, uses the existing production
vector index, and binds every model citation to canonical local evidence.
There is no write-capable Neo4j path in this runtime.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from neo4j import GraphDatabase

from .phase6f_benchmark import (
    OUTPUT,
    build_exception_fallbacks,
    clean_section_title,
    json_value,
    load_json,
    own_section_records,
    referenced_table_numbers,
    table_records,
)


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = OUTPUT / "document.json"
REQUIREMENTS_PATH = OUTPUT / "semantic" / "requirements.parquet"
MODEL = "gpt-5.6-luna"
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_INDEX = "hvac_passage_embeddings"
VECTOR_DIMENSIONS = 3072
MAX_TOOL_CALLS = 6
MAX_CYPHER_ROWS = 100
MAX_VECTOR_RESULTS = 10

SECTION_NUMBER_RE = re.compile(r"^\d{3,4}(?:\.\d+)*$")
EVIDENCE_ID_RE = re.compile(r"\[(E\d+)\]")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")

GRAPH_SCHEMA_DESCRIPTION = """Current Aura graph schema:
(:Document)-[:CONTAINS]->(:Chapter)
(:Chapter)-[:CONTAINS]->(:Section)
(:Section)-[:CONTAINS]->(:Section)
(:Section)-[:STATES]->(:Requirement)

Document properties: id, title, source_file, source_sha256, parser_name, parser_version
Chapter properties: id, number, title, order, page
Section properties: id, number, title, chapter_id, parent_section_id, order, page
Requirement properties: id, section_id, predicate, subject, object, value, unit,
condition, exception, source_text, page, source_block_id, char_start, char_end,
confidence, retrieval_hash, embedding_model, is_embedding_representative

The special section:unassigned ownership is not authoritative retrieval evidence.
The vector index is hvac_passage_embeddings on Requirement.retrieval_embedding,
with 3072 dimensions and cosine similarity. Vector properties exist only on
representative production Requirements."""

SYSTEM_INSTRUCTIONS = f"""You are the HVAC Codes ReAct GraphRAG assistant.

{GRAPH_SCHEMA_DESCRIPTION}

You have exactly three retrieval tools: CypherSearch, VectorSearch, and HybridSearch.
Choose the first tool yourself. You may call any tool first, call tools sequentially,
and revise your retrieval strategy after each observation. Do not follow a fixed
deterministic router.

Scope and response policy:
- The authoritative domain is the supplied HVAC code corpus and questions that
  naturally help a user navigate it.
- For regulatory/code questions, use retrieval and answer only from the supplied
  canonical evidence.
- For greetings, identity questions, capability questions, and ordinary follow-ups
  that do not require a code claim, respond naturally without retrieval. These
  responses do not need citations.
- For questions outside HVAC codes, politely explain that you are focused on the
  supplied HVAC code corpus and suggest an in-scope alternative. Do not invent
  outside facts and do not call retrieval tools just to manufacture an answer.
- If a request is ambiguous, ask a concise clarifying question rather than
  presenting an unsupported code interpretation.

CypherSearch accepts read-only Cypher supplied by you. Use only the schema above;
do not invent legacy labels, relationships, or properties. Include a bounded LIMIT
and return section identifiers or requirement section_id values so the application
can bind the results to canonical evidence.

VectorSearch and HybridSearch accept semantic search text, not Cypher.

Answer only from the supplied canonical evidence. Preserve exact terminology,
conditions, exceptions, permissions, and prohibitions. Never invent a requirement.
If the gathered evidence genuinely cannot answer the question, say so explicitly.
Tool observations are compact: `new_evidence` contains complete canonical source
blocks, while `existing_evidence` names source blocks already present in the
conversation-wide evidence ledger. Use only those evidence IDs for citations.
When retrieval was used, cite every factual or regulatory statement using only
supplied evidence identifiers such as [E1]. Put a relevant evidence identifier at
the end of each paragraph or list group. If no retrieval was needed, do not invent
a citation merely to satisfy a format rule.
Never generate Section/page citations yourself. Do not use citations such as
[Section 303.3, p. 10]. Keep the final answer concise and answer directly.
"""

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "CypherSearch",
        "description": "Execute bounded parameterized read-only Cypher against the current Aura graph and return canonical evidence.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "One read-only Cypher statement with LIMIT <= 100."},
                "parameters": {"type": "object", "additionalProperties": True},
            },
            "required": ["query", "parameters"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "VectorSearch",
        "description": "Embed a semantic HVAC query with text-embedding-3-large and retrieve up to 10 unique production passages from hvac_passage_embeddings.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "HybridSearch",
        "description": "Use semantic vector candidates followed by graph traversal to recover Sections, ancestors, linked production Requirements, and canonical evidence.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
]


def load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _validate_parameters(value: Any, path: str = "parameters") -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_parameters(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_parameters(item, f"{path}.{key}")
        return
    raise ValueError(f"Unsupported Cypher parameter type at {path}: {type(value).__name__}")


def validate_read_only_cypher(query: str, max_rows: int = MAX_CYPHER_ROWS) -> str:
    """Reject write/schema/admin Cypher before it reaches Aura."""

    text = str(query or "").strip()
    if not text:
        raise ValueError("Cypher query is empty")
    if ";" in text.rstrip(";"):
        raise ValueError("Multiple Cypher statements are not permitted")
    if "//" in text or "/*" in text or "*/" in text:
        raise ValueError("Cypher comments are not permitted")

    tokens = [token.upper() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)]
    denied = {
        "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "ALTER",
        "RENAME", "LOAD", "CSV", "START", "FOREACH", "CALL", "USE", "SHOW",
        "TERMINATE", "GRANT", "DENY", "REVOKE", "DATABASE", "TRANSACTION",
        "CONSTRAINT", "INDEX", "ADMIN", "RETRIEVAL_EMBEDDING",
    }
    found = sorted(set(tokens) & denied)
    if found:
        raise ValueError(f"Read-only Cypher rejected forbidden clause(s): {', '.join(found)}")
    if not re.match(r"^\s*(MATCH|OPTIONAL|UNWIND|WITH|RETURN)\b", text, re.I):
        raise ValueError("Cypher must begin with a read-only clause")
    limits = re.findall(r"\bLIMIT\s+(\d+)\b", text, re.I)
    if not limits:
        raise ValueError(f"Cypher must include LIMIT <= {max_rows}")
    if any(int(value) > max_rows for value in limits):
        raise ValueError(f"Cypher LIMIT cannot exceed {max_rows}")
    return text


@dataclass
class EvidenceBlock:
    evidence_id: str
    section_number: str
    page: int | None
    source_text: str
    section_title: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "page": self.page,
            "source_text": self.source_text,
        }


class EvidenceRegistry:
    """Per-question deterministic E# registry and citation renderer."""

    def __init__(self) -> None:
        self._blocks: list[EvidenceBlock] = []
        self._by_key: dict[tuple[str, int | None, str], EvidenceBlock] = {}

    @staticmethod
    def _key(section_number: str, page: int | None, source_text: str) -> tuple[str, int | None, str]:
        normalized = re.sub(r"\s+", " ", str(source_text or "")).strip()
        return str(section_number), page, normalized.casefold()

    def add(self, section_number: str, page: int | None, source_text: str, section_title: str = "") -> str:
        text = re.sub(r"\s+", " ", str(source_text or "")).strip()
        if not text:
            raise ValueError("Cannot register empty evidence")
        key = self._key(section_number, page, text)
        existing = self._by_key.get(key)
        if existing:
            return existing.evidence_id
        evidence_id = f"E{len(self._blocks) + 1}"
        block = EvidenceBlock(evidence_id, str(section_number), page, text, str(section_title or ""))
        self._blocks.append(block)
        self._by_key[key] = block
        return evidence_id

    def get(self, evidence_id: str) -> EvidenceBlock | None:
        return next((block for block in self._blocks if block.evidence_id == evidence_id), None)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [block.as_dict() for block in self._blocks]

    def render(self, raw_answer: str) -> dict[str, Any]:
        raw = str(raw_answer or "")
        cited_ids = EVIDENCE_ID_RE.findall(raw)
        unsupported: list[str] = []
        malformed: list[str] = []
        forbidden_section_citations: list[str] = []
        known_ids = {block.evidence_id for block in self._blocks}
        for value in BRACKET_RE.findall(raw):
            stripped = value.strip()
            if re.match(r"^E\d+", stripped):
                if not re.fullmatch(r"E\d+", stripped):
                    malformed.append(f"[{value}]")
                elif stripped not in known_ids:
                    unsupported.append(stripped)
            if re.match(r"^Section\s+", stripped, re.I):
                forbidden_section_citations.append(f"[{value}]")
        for evidence_id in cited_ids:
            if evidence_id not in known_ids:
                unsupported.append(evidence_id)
        if malformed or unsupported or forbidden_section_citations or not cited_ids:
            return {
                "valid": False,
                "rendered_answer": None,
                "cited_evidence_ids": cited_ids,
                "unsupported_evidence_ids": sorted(set(unsupported)),
                "malformed_citations": malformed,
                "forbidden_section_citations": forbidden_section_citations,
            }

        rendered = raw
        for evidence_id in cited_ids:
            block = self.get(evidence_id)
            if block is None:
                continue
            page = f", p. {block.page}" if block.page is not None else ""
            rendered = rendered.replace(f"[{evidence_id}]", f"[Section {block.section_number}{page}]")
        return {
            "valid": True,
            "rendered_answer": rendered,
            "cited_evidence_ids": cited_ids,
            "unsupported_evidence_ids": [],
            "malformed_citations": [],
            "forbidden_section_citations": [],
        }


_TOKEN_ENCODER: Any | None = None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def estimate_payload_tokens(value: Any) -> int:
    """Estimate payload tokens locally without contacting OpenAI."""

    global _TOKEN_ENCODER
    text = value if isinstance(value, str) else _compact_json(value)
    if not text:
        return 0
    if _TOKEN_ENCODER is None:
        try:
            import tiktoken

            _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKEN_ENCODER = False
    if _TOKEN_ENCODER is False:
        return max(1, (len(text) + 3) // 4)
    return len(_TOKEN_ENCODER.encode(text))


def _compact_evidence_block(block: dict[str, Any]) -> dict[str, Any] | None:
    section_number = str(block.get("section_number") or "")
    source_text = str(block.get("source_text") or "").strip()
    if section_number == "unassigned" or not source_text:
        return None
    return {
        "evidence_id": str(block.get("evidence_id") or ""),
        "section_number": section_number,
        "section_title": str(block.get("section_title") or ""),
        "page": block.get("page"),
        "source_text": source_text,
    }


def compact_tool_observation(
    observation: dict[str, Any],
    emitted_evidence_ids: set[str],
) -> dict[str, Any]:
    """Reduce a rich internal tool result to the model-visible observation.

    The executor may retain arbitrary graph/Requirement records internally. Only
    ranked section identities and complete, canonical evidence blocks cross the
    model boundary. ``emitted_evidence_ids`` is per-question state and is updated
    in-place so evidence IDs remain stable across ReAct iterations.
    """

    observation = observation if isinstance(observation, dict) else {}
    tool_name = str(observation.get("tool") or "UnknownTool")
    compact: dict[str, Any] = {"tool": tool_name, "candidates": []}

    canonical_blocks = observation.get("canonical_evidence") or []
    evidence_by_section: dict[str, dict[str, Any]] = {}
    for raw_block in canonical_blocks:
        if not isinstance(raw_block, dict):
            continue
        block = _compact_evidence_block(raw_block)
        if block is None:
            continue
        evidence_by_section.setdefault(block["section_number"], block)

    candidates: list[dict[str, Any]] = []
    passages = observation.get("passages") or []
    if isinstance(passages, list) and passages:
        for rank, passage in enumerate(passages, start=1):
            if not isinstance(passage, dict):
                continue
            section = passage.get("section") or {}
            if not isinstance(section, dict):
                continue
            number = str(section.get("number") or "")
            if number == "unassigned" or not SECTION_NUMBER_RE.fullmatch(number):
                continue
            evidence = evidence_by_section.get(number) or {}
            candidates.append({
                "rank": rank,
                "section_number": number,
                "section_title": str(section.get("title") or evidence.get("section_title") or ""),
                "page": section.get("page") if section.get("page") is not None else evidence.get("page"),
            })
    else:
        selected = observation.get("selected_sections") or []
        if isinstance(selected, list):
            for rank, value in enumerate(selected, start=1):
                number = str(value or "")
                block = evidence_by_section.get(number)
                if block is None or number == "unassigned":
                    continue
                candidates.append({
                    "rank": rank,
                    "section_number": number,
                    "section_title": block["section_title"],
                    "page": block["page"],
                })

    seen_candidates: set[tuple[str, Any]] = set()
    for candidate in candidates:
        key = (candidate["section_number"], candidate.get("page"))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        compact["candidates"].append(candidate)

    if isinstance(observation.get("passages"), list):
        compact["result_count"] = len(observation["passages"])
    elif isinstance(observation.get("rows"), list):
        compact["result_count"] = len(observation["rows"])

    existing: list[str] = []
    new: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for raw_block in canonical_blocks:
        if not isinstance(raw_block, dict):
            continue
        block = _compact_evidence_block(raw_block)
        if block is None:
            continue
        evidence_id = block["evidence_id"]
        if not evidence_id or evidence_id in seen_evidence:
            continue
        seen_evidence.add(evidence_id)
        if evidence_id in emitted_evidence_ids:
            existing.append(evidence_id)
        else:
            new.append(block)
            emitted_evidence_ids.add(evidence_id)
    if existing:
        compact["existing_evidence"] = existing
    if new:
        compact["new_evidence"] = new
    warnings = observation.get("warnings") or []
    if warnings:
        compact["warnings"] = [str(value) for value in warnings]
    if observation.get("error"):
        compact["error"] = str(observation["error"])
    return compact


def build_payload_accounting(
    question: str,
    input_items: list[Any],
    registry: EvidenceRegistry,
) -> dict[str, Any]:
    """Return non-sensitive token accounting for one Responses API turn."""

    history = list(input_items[1:]) if len(input_items) > 1 else []
    latest_observation = None
    prior_history = history
    if history and isinstance(history[-1], dict) and history[-1].get("type") == "function_call_output":
        latest_observation = history[-1].get("output")
        prior_history = history[:-1]
    system_tokens = estimate_payload_tokens(SYSTEM_INSTRUCTIONS)
    tool_tokens = estimate_payload_tokens(TOOL_SPECS)
    question_tokens = estimate_payload_tokens(question)
    prior_tokens = estimate_payload_tokens(prior_history)
    new_observation_tokens = estimate_payload_tokens(latest_observation or "")
    evidence_tokens = estimate_payload_tokens(registry.as_dicts())
    return {
        "system_instruction_tokens": system_tokens,
        "tool_schema_tokens": tool_tokens,
        "question_tokens": question_tokens,
        "prior_compact_history_tokens": prior_tokens,
        "new_tool_observation_tokens": new_observation_tokens,
        "cumulative_unique_evidence_tokens": evidence_tokens,
        "approximate_total_input_tokens": system_tokens + tool_tokens + question_tokens + prior_tokens + new_observation_tokens,
    }


class CanonicalEvidenceAssembler:
    """Adapter over the validated v2 structural/evidence records."""

    def __init__(self, document_path: Path = DOCUMENT_PATH, requirements_path: Path = REQUIREMENTS_PATH) -> None:
        self.document = load_json(document_path)
        sections = [
            section for section in self.document.get("sections", [])
            if SECTION_NUMBER_RE.fullmatch(str(section.get("number") or ""))
        ]
        self.sections_by_number = {str(section["number"]): section for section in sections}
        self.sections_by_id = {str(section["id"]): section for section in sections}
        self.section_numbers = set(self.sections_by_number)
        self.table_by_id = {str(table["id"]): table for table in self.document.get("tables", [])}
        self.exception_fallbacks = build_exception_fallbacks()
        # Semantic review flags do not invalidate the underlying canonical
        # structural source.  Evidence assembly must expose every normative
        # block owned by a selected section; review/quarantine filtering is a
        # Requirement-corpus concern, not a source-block omission rule.
        self.review_block_ids = self._load_review_block_ids(requirements_path)

    @staticmethod
    def _load_review_block_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        frame = pd.read_parquet(path)
        result: set[str] = set()
        for row in frame[frame["needs_review"].astype(bool)].itertuples(index=False):
            evidence = json_value(getattr(row, "evidence", None), {}) or {}
            block_id = evidence.get("source_block_id") or getattr(row, "source_block_id", None)
            if block_id:
                result.add(str(block_id))
        return result

    def _ancestors(self, section: dict[str, Any]) -> list[str]:
        result: list[str] = []
        current = section
        seen: set[str] = set()
        while current and current.get("parent_section_id") and current["parent_section_id"] not in seen:
            parent_id = str(current["parent_section_id"])
            seen.add(parent_id)
            parent = self.sections_by_id.get(parent_id)
            if not parent:
                break
            number = str(parent.get("number") or "")
            if number in self.sections_by_number:
                result.append(number)
            current = parent
        result.reverse()
        return result

    def assemble(self, direct_section_numbers: list[str], registry: EvidenceRegistry) -> dict[str, Any]:
        direct: list[str] = []
        for number in direct_section_numbers:
            number = str(number)
            if number == "unassigned" or number not in self.sections_by_number:
                continue
            if number not in direct:
                direct.append(number)

        selected: list[str] = []
        for number in direct:
            selected.append(number)
            for ancestor in self._ancestors(self.sections_by_number[number]):
                if ancestor not in selected:
                    selected.append(ancestor)

        evidence_ids: list[str] = []
        for number in selected:
            section = self.sections_by_number[number]
            title = clean_section_title(section.get("title"))
            records = own_section_records(section, self.section_numbers, self.exception_fallbacks)
            for record in records:
                evidence_id = registry.add(
                    section_number=number,
                    page=record.get("page"),
                    source_text=record.get("source_text"),
                    section_title=title,
                )
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)

            if number not in direct:
                continue
            for table_number in referenced_table_numbers(section, records, self.table_by_id):
                table = next(
                    (item for item in self.table_by_id.values() if str(item.get("table_number")) == table_number),
                    None,
                )
                if not table:
                    continue
                for record in table_records(table, number, title):
                    evidence_id = registry.add(
                        section_number=number,
                        page=record.get("page"),
                        source_text=record.get("source_text"),
                        section_title=title,
                    )
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)

        return {
            "direct_sections": direct,
            "selected_sections": selected,
            "evidence_ids": evidence_ids,
            "evidence": [block.as_dict() for block in registry._blocks if block.evidence_id in evidence_ids],
        }


class AuraReadOnlyStore:
    """Official Neo4j driver wrapper with no write-capable public method."""

    def __init__(self, uri: str, username: str, password: str, database: str, driver: Any | None = None) -> None:
        self.database = database
        self._owns_driver = driver is None
        self.driver = driver or GraphDatabase.driver(uri, auth=(username, password))

    @classmethod
    def from_environment(cls) -> "AuraReadOnlyStore":
        load_environment()
        return cls(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"], os.environ["NEO4J_DATABASE"])

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def read_cypher(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        safe_query = validate_read_only_cypher(query)
        _validate_parameters(parameters)
        with self.driver.session(database=self.database) as session:
            return [_json_safe(row) for row in session.execute_read(lambda tx: tx.run(safe_query, parameters).data())]

    def _read_fixed(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [_json_safe(row) for row in session.execute_read(lambda tx: tx.run(query, **parameters).data())]

    def preflight(self) -> bool:
        """Verify Aura connectivity with the fixed read-only probe."""

        rows = self._read_fixed("RETURN 1 AS ok")
        return bool(rows and rows[0].get("ok") == 1)

    def vector_query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        return self._read_fixed(
            """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            WHERE node.section_id <> 'section:unassigned'
              AND node.is_embedding_representative = true
              AND node.retrieval_embedding IS NOT NULL
            MATCH (s:Section)-[:STATES]->(node)
            RETURN node.id AS representative_requirement_id,
                   node.retrieval_hash AS retrieval_hash,
                   score, s.id AS section_id, s.number AS section_number,
                   s.title AS section_title, s.page AS page
            ORDER BY score DESC, node.id
            LIMIT $top_k
            """,
            index_name=VECTOR_INDEX,
            top_k=max(1, min(int(top_k), MAX_VECTOR_RESULTS)),
            embedding=embedding,
        )

    def section_context(self, section_id: str) -> list[dict[str, Any]]:
        rows = self._read_fixed(
            """
            MATCH path=(ancestor:Section)-[:CONTAINS*0..20]->(target:Section {id: $section_id})
            WHERE ancestor.id <> 'section:unassigned'
            RETURN [node IN nodes(path) | {
                id: node.id, number: node.number, title: node.title,
                chapter_id: node.chapter_id, parent_section_id: node.parent_section_id,
                order: node.order, page: node.page
            }] AS chain, length(path) AS depth
            ORDER BY depth DESC
            LIMIT 1
            """,
            section_id=section_id,
        )
        return rows[0].get("chain", []) if rows else []

    def requirements_for_hash(self, retrieval_hash: str) -> list[dict[str, Any]]:
        return self._read_fixed(
            """
            MATCH (r:Requirement {retrieval_hash: $retrieval_hash})
            WHERE r.section_id <> 'section:unassigned'
            RETURN r.id AS id, r.section_id AS section_id, r.predicate AS predicate,
                   r.subject AS subject, r.object AS object, r.value AS value,
                   r.unit AS unit, r.condition AS condition, r.exception AS exception,
                   r.source_text AS source_text, r.page AS page,
                   r.source_block_id AS source_block_id, r.char_start AS char_start,
                   r.char_end AS char_end, r.confidence AS confidence
            ORDER BY r.id LIMIT 100
            """,
            retrieval_hash=retrieval_hash,
        )

    def section_requirements(self, section_id: str) -> list[dict[str, Any]]:
        return self._read_fixed(
            """
            MATCH (s:Section {id: $section_id})-[:STATES]->(r:Requirement)
            WHERE r.section_id <> 'section:unassigned'
            RETURN r.id AS id, r.section_id AS section_id, r.predicate AS predicate,
                   r.subject AS subject, r.object AS object, r.value AS value,
                   r.unit AS unit, r.condition AS condition, r.exception AS exception,
                   r.source_text AS source_text, r.page AS page,
                   r.source_block_id AS source_block_id, r.char_start AS char_start,
                   r.char_end AS char_end, r.confidence AS confidence
            ORDER BY r.id LIMIT 100
            """,
            section_id=section_id,
        )

    def statistics(self) -> dict[str, Any]:
        rows = self._read_fixed(
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY label LIMIT 20
            """
        )
        return {str(row["label"]): row["count"] for row in rows}


class QueryEmbeddingProvider:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client
        self._cache: dict[str, list[float]] = {}

    def _get_client(self) -> Any:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()
        return self.client

    def embed(self, query: str) -> list[float]:
        key = str(query).strip()
        if not key:
            raise ValueError("Semantic query is empty")
        if key in self._cache:
            return self._cache[key]
        response = self._get_client().embeddings.create(model=EMBEDDING_MODEL, input=[key])
        data = list(getattr(response, "data", []) or [])
        if len(data) != 1:
            raise RuntimeError(f"Expected one query embedding, received {len(data)}")
        vector = [float(value) for value in data[0].embedding]
        if len(vector) != VECTOR_DIMENSIONS:
            raise RuntimeError(f"Expected {VECTOR_DIMENSIONS}-dimensional query embedding, received {len(vector)}")
        self._cache[key] = vector
        return vector


class RetrievalToolExecutor:
    def __init__(self, store: AuraReadOnlyStore, assembler: CanonicalEvidenceAssembler, embedder: QueryEmbeddingProvider) -> None:
        self.store = store
        self.assembler = assembler
        self.embedder = embedder

    def _register_sections(self, section_numbers: list[str], registry: EvidenceRegistry) -> dict[str, Any]:
        return self.assembler.assemble(section_numbers, registry)

    def cypher_search(self, arguments: dict[str, Any], registry: EvidenceRegistry) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        parameters = arguments.get("parameters") or {}
        rows = self.store.read_cypher(query, parameters)
        section_numbers: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key = str(key)
                    if key in {"section_number", "number"} and SECTION_NUMBER_RE.fullmatch(str(item or "")):
                        section_numbers.append(str(item))
                    elif key in {"section_id", "id"} and str(item or "").startswith("section:"):
                        number = str(item).split(":", 1)[1]
                        if SECTION_NUMBER_RE.fullmatch(number):
                            section_numbers.append(number)
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(rows)
        section_numbers = list(dict.fromkeys(section_numbers))
        canonical = self._register_sections(section_numbers, registry)
        return {
            "tool": "CypherSearch", "query": query, "rows": rows,
            "canonical_evidence": canonical["evidence"],
            "selected_sections": canonical["selected_sections"],
            "warnings": [] if canonical["evidence"] else ["Query returned no canonical section evidence."],
        }

    def _vector_candidates(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self.embedder.embed(query)
        rows = self.store.vector_query(vector, top_k)
        candidates: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for row in rows:
            retrieval_hash = str(row.get("retrieval_hash") or "")
            if not retrieval_hash or retrieval_hash in seen_hashes:
                continue
            section_id = str(row.get("section_id") or "")
            if section_id == "section:unassigned" or str(row.get("section_number")) == "unassigned":
                continue
            seen_hashes.add(retrieval_hash)
            candidates.append({
                "retrieval_hash": retrieval_hash,
                "representative_requirement_id": row.get("representative_requirement_id"),
                "similarity": row.get("score"),
                "section": {
                    "id": section_id, "number": row.get("section_number"),
                    "title": row.get("section_title"), "page": row.get("page"),
                },
                "ancestors": self.store.section_context(section_id),
                "requirements": self.store.requirements_for_hash(retrieval_hash),
            })
        return candidates

    def _attach_evidence(self, candidates: list[dict[str, Any]], registry: EvidenceRegistry) -> dict[str, Any]:
        direct = [str(candidate["section"]["number"]) for candidate in candidates]
        canonical = self._register_sections(direct, registry)
        evidence_by_id = {block["evidence_id"]: block for block in canonical["evidence"]}
        for candidate in candidates:
            allowed_sections = {
                str(candidate["section"]["number"]),
                *[str(item.get("number")) for item in candidate.get("ancestors", [])],
            }
            candidate["evidence_ids"] = [
                evidence_id for evidence_id in canonical["evidence_ids"]
                if evidence_by_id[evidence_id]["section_number"] in allowed_sections
            ]
        return canonical

    def vector_search(self, arguments: dict[str, Any], registry: EvidenceRegistry) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        top_k = max(1, min(int(arguments.get("top_k", 10)), MAX_VECTOR_RESULTS))
        candidates = self._vector_candidates(query, top_k)
        canonical = self._attach_evidence(candidates, registry)
        return {
            "tool": "VectorSearch", "query": query, "passages": candidates,
            "canonical_evidence": canonical["evidence"],
            "selected_sections": canonical["selected_sections"],
            "warnings": [] if candidates else ["No production vector passages found."],
        }

    def hybrid_search(self, arguments: dict[str, Any], registry: EvidenceRegistry) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        top_k = max(1, min(int(arguments.get("top_k", 10)), MAX_VECTOR_RESULTS))
        candidates = self._vector_candidates(query, top_k)
        for candidate in candidates:
            section_id = str(candidate["section"].get("id") or "")
            candidate["graph_section_requirements"] = self.store.section_requirements(section_id)
            candidate["graph_ancestor_requirements"] = []
            for ancestor in candidate.get("ancestors", []):
                ancestor_id = str(ancestor.get("id") or "")
                if ancestor_id and ancestor_id != section_id:
                    candidate["graph_ancestor_requirements"].extend(self.store.section_requirements(ancestor_id))
        canonical = self._attach_evidence(candidates, registry)
        return {
            "tool": "HybridSearch", "query": query, "passages": candidates,
            "canonical_evidence": canonical["evidence"],
            "selected_sections": canonical["selected_sections"],
            "warnings": [] if candidates else ["No production hybrid candidates found."],
        }

    def execute(self, name: str, arguments: dict[str, Any], registry: EvidenceRegistry) -> dict[str, Any]:
        dispatch: dict[str, Callable[[dict[str, Any], EvidenceRegistry], dict[str, Any]]] = {
            "CypherSearch": self.cypher_search,
            "VectorSearch": self.vector_search,
            "HybridSearch": self.hybrid_search,
        }
        if name not in dispatch:
            raise ValueError(f"Unknown retrieval tool: {name}")
        return dispatch[name](arguments, registry)


def _raw_response_item_dict(item: Any) -> dict[str, Any]:
    """Preserve the complete response item for local audit/debugging only."""

    if isinstance(item, dict):
        return _json_safe(item)
    if hasattr(item, "model_dump"):
        return _json_safe(item.model_dump(exclude_none=False))
    if hasattr(item, "__dict__"):
        return _json_safe(vars(item))
    fields = (
        "type", "id", "status", "role", "name", "arguments", "call_id",
        "content", "summary", "encrypted_content", "output",
    )
    return {
        key: _json_safe(getattr(item, key))
        for key in fields
        if hasattr(item, key)
    }


def _without_none(value: Any) -> Any:
    """Remove None-valued fields from a resume item without changing content."""

    if isinstance(value, dict):
        return {
            str(key): _without_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def serialize_response_output_for_resume(item: Any) -> dict[str, Any]:
    """Return the narrow API-compatible form of one model output item.

    The complete item is intentionally handled separately by
    :func:`_raw_response_item_dict` and never returned to ``responses.create``.
    """

    raw = _raw_response_item_dict(item)
    item_type = str(raw.get("type") or "")
    allowed_by_type = {
        "reasoning": ("type", "id", "summary", "encrypted_content", "content"),
        "function_call": ("type", "call_id", "name", "arguments", "id"),
        "function_call_output": ("type", "call_id", "output"),
    }
    # Only the explicitly supported continuation item types get their
    # type-specific fields. Unknown output types use a conservative generic
    # allowlist rather than ever forwarding a model dump.
    allowed = allowed_by_type.get(
        item_type,
        ("type", "id", "role", "content", "name", "arguments", "call_id"),
    )
    return _without_none({key: raw[key] for key in allowed if key in raw})


def serialize_function_call_output_for_resume(call_id: Any, output: Any) -> dict[str, Any]:
    """Serialize the locally-created tool result with only API input fields."""

    return {
        "type": "function_call_output",
        "call_id": _json_safe(call_id),
        "output": _json_safe(output),
    }


# Kept as a private compatibility alias for callers that used the old audit
# serializer. Runtime continuation code must use the explicit resume helper.
def _model_item_dict(item: Any) -> dict[str, Any]:
    return _raw_response_item_dict(item)


@dataclass
class ReActSession:
    """All mutable state belonging to one independent user question."""

    session_id: str
    question_id: str
    question: str
    input_items: list[Any]
    registry: EvidenceRegistry
    emitted_evidence_ids: set[str]
    checkpoint_path: Path | None = None
    tool_calls: int = 0
    turn: int = 0
    payload_accounting: list[dict[str, Any]] | None = None
    usage_checkpoints: list[dict[str, Any]] | None = None
    function_call_pairs: list[dict[str, Any]] = field(default_factory=list)
    last_response_output_items: list[dict[str, Any]] = field(default_factory=list)
    last_raw_response_output_items: list[dict[str, Any]] = field(default_factory=list)
    response_output_history: list[dict[str, Any]] = field(default_factory=list)
    raw_response_output_history: list[dict[str, Any]] = field(default_factory=list)
    request_configuration: dict[str, Any] | None = None
    per_question_input_tokens: int = 0
    evaluation_total_input_tokens: int = 0
    previous_response_id: str | None = None

    def __post_init__(self) -> None:
        if self.payload_accounting is None:
            self.payload_accounting = []
        if self.usage_checkpoints is None:
            self.usage_checkpoints = []


class ReActGraphRAG:
    """Responses API ReAct loop with exactly three retrieval tools."""

    def __init__(
        self,
        store: AuraReadOnlyStore | None = None,
        client: Any | None = None,
        assembler: CanonicalEvidenceAssembler | None = None,
        embedder: QueryEmbeddingProvider | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.assembler = assembler or CanonicalEvidenceAssembler()
        self.embedder = embedder or QueryEmbeddingProvider(client=client)
        self.max_tool_calls = max(1, min(int(max_tool_calls), MAX_TOOL_CALLS))
        self.evaluation_total_input_tokens = 0
        self.last_session: ReActSession | None = None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else ROOT / "v2_output" / "react" / "checkpoints"
        self.last_checkpoint_path: Path | None = None

    def _checkpoint_path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))
        return self.checkpoint_dir / f"{safe_id}.json"

    def start_session(
        self,
        question: str,
        question_id: str | None = None,
        session_id: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> ReActSession:
        """Create an isolated session; no prior question state is reused."""

        text = str(question or "").strip()
        resolved_session_id = str(session_id or uuid4().hex)
        input_items = self._conversation_input_items(conversation_history)
        input_items.append({"role": "user", "content": text})
        session = ReActSession(
            session_id=resolved_session_id,
            question_id=str(question_id or "question"),
            question=text,
            input_items=input_items,
            registry=EvidenceRegistry(),
            emitted_evidence_ids=set(),
            checkpoint_path=self._checkpoint_path(resolved_session_id),
            request_configuration=self._request_configuration(),
            evaluation_total_input_tokens=self.evaluation_total_input_tokens,
        )
        self.last_session = session
        self.last_checkpoint_path = session.checkpoint_path
        return session

    @staticmethod
    def _conversation_input_items(
        conversation_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Copy a small browser-session history into a new request.

        This is conversational memory only. It does not carry ReAct tool calls,
        evidence ledgers, checkpoints, or graph state between questions.
        """

        if not conversation_history:
            return []
        result: list[dict[str, str]] = []
        for item in conversation_history[-8:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            result.append({"role": role, "content": content[:4000]})
        return result

    @staticmethod
    def _request_configuration() -> dict[str, Any]:
        return {
            "model": MODEL,
            "reasoning": {"effort": "low"},
            "instructions": SYSTEM_INSTRUCTIONS,
            "tools": TOOL_SPECS,
            "parallel_tool_calls": False,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }

    def initial_request(self, session: ReActSession) -> dict[str, Any]:
        """Build the first request for a session without calling OpenAI."""

        request = dict(session.request_configuration or self._request_configuration())
        request["input"] = list(session.input_items)
        return request

    def _get_store(self) -> AuraReadOnlyStore:
        if self.store is None:
            self.store = AuraReadOnlyStore.from_environment()
        return self.store

    def _get_client(self) -> Any:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()
            self.embedder.client = self.client
        return self.client

    def _checkpoint_payload(
        self,
        session: ReActSession,
        phase: str,
        response_output_items: list[dict[str, Any]] | None = None,
        raw_response_output_items: list[dict[str, Any]] | None = None,
        response_id: str | None = None,
        response_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the complete opaque continuation state for one session."""

        evidence = session.registry.as_dicts()
        resume_output_items = list(
            response_output_items
            if response_output_items is not None
            else session.last_response_output_items
        )
        raw_output_items = list(
            raw_response_output_items
            if raw_response_output_items is not None
            else session.last_raw_response_output_items
        )
        return {
            "checkpoint_version": 2,
            "phase": str(phase),
            "session_id": session.session_id,
            "question_id": session.question_id,
            "question": session.question,
            "turn": session.turn,
            # ``raw_response_output`` is audit/debug data only. It must never
            # be used as the next Responses API input.
            "raw_response_output": _json_safe(raw_output_items),
            "raw_response_output_history": _json_safe(list(session.raw_response_output_history)),
            # This is the sole persisted continuation input history.
            "resume_input_items": _json_safe(list(session.input_items)),
            # Retain the old descriptive history field, but keep it sanitized.
            "response_output_items": _json_safe(resume_output_items),
            "response_output_history": _json_safe(list(session.response_output_history)),
            "function_call_pairs": _json_safe(list(session.function_call_pairs)),
            "last_response_id": response_id,
            "last_response_usage": _json_safe(response_usage or {}),
            "evidence_ledger": evidence,
            "evidence_id_mapping": {
                str(item["evidence_id"]): item for item in evidence
            },
            "emitted_evidence_ids": sorted(session.emitted_evidence_ids),
            "tool_call_count": session.tool_calls,
            "per_question_input_tokens": session.per_question_input_tokens,
            "evaluation_total_input_tokens": self.evaluation_total_input_tokens,
            "payload_accounting": list(session.payload_accounting or []),
            "usage_checkpoints": list(session.usage_checkpoints or []),
            "previous_response_id": session.previous_response_id,
            "request_configuration": _json_safe(
                session.request_configuration or self._request_configuration()
            ),
        }

    def _persist_checkpoint(
        self,
        session: ReActSession,
        phase: str,
        response_output_items: list[dict[str, Any]] | None = None,
        raw_response_output_items: list[dict[str, Any]] | None = None,
        response_id: str | None = None,
        response_usage: dict[str, Any] | None = None,
    ) -> None:
        """Atomically persist a local checkpoint before another request is built."""

        if session.checkpoint_path is None:
            raise RuntimeError("Session has no checkpoint path")
        path = session.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload(
            session,
            phase=phase,
            response_output_items=response_output_items,
            raw_response_output_items=raw_response_output_items,
            response_id=response_id,
            response_usage=response_usage,
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        last_error: PermissionError | None = None
        for delay in (0.0, 0.1, 0.25, 0.5):
            if delay:
                import time

                time.sleep(delay)
            try:
                os.replace(temporary, path)
                self.last_checkpoint_path = path
                return
            except PermissionError as exc:
                last_error = exc

        # Windows can briefly hold the previous checkpoint while the local
        # file watcher is reading it. Preserve the new checkpoint under a
        # versioned name rather than turning a successful answer into a
        # runtime failure.
        fallback = path.with_name(f"{path.stem}.{uuid4().hex[:8]}{path.suffix}")
        try:
            os.replace(temporary, fallback)
        except OSError:
            if last_error is not None:
                raise last_error
            raise
        session.checkpoint_path = fallback
        self.last_checkpoint_path = fallback

    @staticmethod
    def _restore_registry(payload: dict[str, Any]) -> EvidenceRegistry:
        registry = EvidenceRegistry()
        for item in payload.get("evidence_ledger", []) or []:
            if not isinstance(item, dict):
                raise ValueError("Checkpoint evidence ledger contains a non-object")
            evidence_id = registry.add(
                section_number=str(item.get("section_number") or ""),
                page=item.get("page"),
                source_text=str(item.get("source_text") or ""),
                section_title=str(item.get("section_title") or ""),
            )
            if evidence_id != str(item.get("evidence_id") or ""):
                raise ValueError("Checkpoint evidence IDs are not sequential or were altered")
        return registry

    def load_checkpoint(self, checkpoint_path: Path | str) -> ReActSession:
        """Load a ready local checkpoint without executing any completed tool."""

        path = Path(checkpoint_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("checkpoint_version", 0)) != 2:
            raise ValueError("Unsupported ReAct checkpoint version")
        # Never fall back to the raw/output history. Old checkpoints without
        # this field cannot be resumed safely because they may contain status
        # or other response-only metadata.
        input_items = payload.get("resume_input_items")
        if not isinstance(input_items, list) or not input_items:
            raise ValueError("Checkpoint does not contain sanitized resume input history")
        registry = self._restore_registry(payload)
        expected_ids = sorted(str(item.get("evidence_id")) for item in payload.get("evidence_ledger", []) or [])
        restored_ids = sorted(item.evidence_id for item in registry._blocks)
        if restored_ids != expected_ids:
            raise ValueError("Checkpoint evidence ledger could not be restored exactly")
        session = ReActSession(
            session_id=str(payload.get("session_id") or ""),
            question_id=str(payload.get("question_id") or "question"),
            question=str(payload.get("question") or ""),
            input_items=_json_safe(input_items),
            registry=registry,
            emitted_evidence_ids={str(value) for value in payload.get("emitted_evidence_ids", []) or []},
            checkpoint_path=path,
            tool_calls=int(payload.get("tool_call_count", 0)),
            turn=int(payload.get("turn", 0)),
            payload_accounting=list(payload.get("payload_accounting", []) or []),
            usage_checkpoints=list(payload.get("usage_checkpoints", []) or []),
            function_call_pairs=list(payload.get("function_call_pairs", []) or []),
            last_response_output_items=list(payload.get("response_output_items", []) or []),
            last_raw_response_output_items=list(payload.get("raw_response_output", []) or []),
            response_output_history=list(payload.get("response_output_history", []) or []),
            raw_response_output_history=list(payload.get("raw_response_output_history", []) or []),
            request_configuration=dict(payload.get("request_configuration") or self._request_configuration()),
            per_question_input_tokens=int(payload.get("per_question_input_tokens", 0)),
            evaluation_total_input_tokens=int(payload.get("evaluation_total_input_tokens", 0)),
            previous_response_id=payload.get("previous_response_id"),
        )
        self.evaluation_total_input_tokens = session.evaluation_total_input_tokens
        self.last_session = session
        self.last_checkpoint_path = path
        return session

    def resume_from_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Continue the same stateless ReAct session from its next-request state."""

        session = self.load_checkpoint(checkpoint_path)
        if session.tool_calls > self.max_tool_calls:
            return self._safe_failure(
                "Checkpoint exceeds the configured tool-call limit.",
                session.registry, [], session.tool_calls, session.payload_accounting, session,
            )
        return self._run_session(session)

    @staticmethod
    def _safe_failure(
        message: str,
        registry: EvidenceRegistry,
        trace: list[dict[str, Any]],
        calls: int,
        payload_accounting: list[dict[str, Any]] | None = None,
        session: ReActSession | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed", "answer": None, "raw_answer": None,
            "error": message, "tool_calls": calls, "tool_trace": trace,
            "evidence": registry.as_dicts(),
            "payload_accounting": payload_accounting or [],
            "question_id": session.question_id if session else None,
            "session_id": session.session_id if session else None,
            "checkpoint_path": str(session.checkpoint_path) if session and session.checkpoint_path else None,
            "usage_checkpoints": list(session.usage_checkpoints or []) if session else [],
            "per_question_input_tokens": session.per_question_input_tokens if session else 0,
            "evaluation_total_input_tokens": session.evaluation_total_input_tokens if session else 0,
        }

    def _checkpoint_response(
        self,
        session: ReActSession,
        response: Any,
        function_calls: list[Any],
        accounting: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist usage immediately after a successful response returns."""

        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(exclude_none=True)
        elif isinstance(usage, dict):
            usage = dict(usage)
        else:
            usage = {
                "input_tokens": getattr(usage, "input_tokens", None) if usage is not None else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage is not None else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
            }
        actual_input = usage.get("input_tokens")
        actual_output = usage.get("output_tokens")
        if actual_input is not None:
            actual_input = int(actual_input)
            session.per_question_input_tokens += actual_input
            self.evaluation_total_input_tokens += actual_input
        if actual_output is not None:
            actual_output = int(actual_output)
        session.evaluation_total_input_tokens = self.evaluation_total_input_tokens
        accounting["actual_input_tokens"] = actual_input
        accounting["actual_output_tokens"] = actual_output
        session.usage_checkpoints.append({
            "response_id": getattr(response, "id", None),
            "question_id": session.question_id,
            "turn": session.turn,
            "actual_input_tokens": actual_input,
            "actual_output_tokens": actual_output,
            "model": MODEL,
            "completed": (
                {
                    "type": "tool_call",
                    "tools": [
                        str(item.get("name") if isinstance(item, dict) else getattr(item, "name", ""))
                        for item in function_calls
                    ],
                }
                if function_calls
                else {"type": "answer"}
            ),
        })
        return usage

    def _run_session(self, session: ReActSession) -> dict[str, Any]:
        question = session.question
        if not question:
            return self._safe_failure("Question is empty.", session.registry, [], 0, session=session)
        store = self._get_store()
        preflight = getattr(store, "preflight", None)
        if callable(preflight):
            try:
                if not preflight():
                    return self._safe_failure(
                        "Aura connectivity preflight failed.", session.registry, [],
                        session.tool_calls, session.payload_accounting, session,
                    )
            except Exception as exc:
                return self._safe_failure(
                    f"Aura connectivity preflight failed: {exc}", session.registry, [],
                    session.tool_calls, session.payload_accounting, session,
                )
        tools = RetrievalToolExecutor(store, self.assembler, self.embedder)
        registry = session.registry
        trace: list[dict[str, Any]] = []
        input_items = session.input_items

        while session.tool_calls < self.max_tool_calls:
            session.turn += 1
            accounting = build_payload_accounting(question, input_items, registry)
            session.payload_accounting.append(accounting)
            try:
                request_configuration = dict(
                    session.request_configuration or self._request_configuration()
                )
                request_configuration["input"] = input_items
                response = self._get_client().responses.create(**request_configuration)
            except Exception as exc:
                return self._safe_failure(
                    f"Responses API request failed: {exc}", registry, trace,
                    session.tool_calls, session.payload_accounting, session,
                )
            output_items = list(getattr(response, "output", []) or [])
            # Keep the complete API output for audit/debugging, but use only
            # the narrow per-type form for the stateless continuation input.
            raw_response_output_items = [_raw_response_item_dict(item) for item in output_items]
            response_output_items = [serialize_response_output_for_resume(item) for item in output_items]
            function_calls = [
                item for item in output_items
                if getattr(item, "type", None) == "function_call"
                or (isinstance(item, dict) and item.get("type") == "function_call")
            ]
            response_usage = self._checkpoint_response(session, response, function_calls, accounting)
            response_id = getattr(response, "id", None)
            session.last_response_output_items = response_output_items
            session.last_raw_response_output_items = raw_response_output_items
            session.response_output_history.append({
                "turn": session.turn,
                "response_id": response_id,
                "output_items": response_output_items,
            })
            session.raw_response_output_history.append({
                "turn": session.turn,
                "response_id": response_id,
                "output_items": raw_response_output_items,
            })
            try:
                self._persist_checkpoint(
                    session, phase="response_received",
                    response_output_items=response_output_items,
                    raw_response_output_items=raw_response_output_items,
                    response_id=response_id,
                    response_usage=response_usage,
                )
            except Exception as exc:
                return self._safe_failure(
                    f"Continuation checkpoint persistence failed: {exc}", registry, trace,
                    session.tool_calls, session.payload_accounting, session,
                )
            input_items.extend(response_output_items)
            if not function_calls:
                try:
                    self._persist_checkpoint(
                        session, phase="answer_ready",
                        response_output_items=response_output_items,
                        raw_response_output_items=raw_response_output_items,
                        response_id=response_id,
                        response_usage=response_usage,
                    )
                except Exception as exc:
                    return self._safe_failure(
                        f"Continuation checkpoint persistence failed: {exc}", registry, trace,
                        session.tool_calls, session.payload_accounting, session,
                    )
                raw_answer = str(getattr(response, "output_text", "") or "").strip()
                if not registry.as_dicts():
                    if _likely_regulatory_question(question):
                        return self._safe_failure(
                            "I couldn’t verify that code-related answer from the canonical HVAC evidence. "
                            "I don’t want to guess; please provide a section number or describe the HVAC "
                            "requirement you want checked.",
                            registry, trace, session.tool_calls, session.payload_accounting, session,
                        )
                    return {
                        "status": "ok", "answer": raw_answer,
                        "raw_answer": raw_answer, "tool_calls": session.tool_calls,
                        "tool_trace": trace, "evidence": [],
                        "citation_validation": {
                            "valid": True, "not_required": True,
                            "cited_evidence_ids": [], "unsupported_evidence_ids": [],
                        }, "model": MODEL,
                        "payload_accounting": session.payload_accounting,
                        "question_id": session.question_id,
                        "session_id": session.session_id,
                        "checkpoint_path": str(session.checkpoint_path) if session.checkpoint_path else None,
                        "usage_checkpoints": list(session.usage_checkpoints),
                        "per_question_input_tokens": session.per_question_input_tokens,
                        "evaluation_total_input_tokens": session.evaluation_total_input_tokens,
                    }
                validation = registry.render(raw_answer)
                if not validation["valid"]:
                    result = self._safe_failure(
                        "Final citation binding failed; the raw answer was not repaired.",
                        registry, trace, session.tool_calls, session.payload_accounting, session,
                    )
                    result["raw_answer"] = raw_answer
                    result["citation_validation"] = validation
                    return result
                return {
                    "status": "ok", "answer": validation["rendered_answer"],
                    "raw_answer": raw_answer, "tool_calls": session.tool_calls,
                    "tool_trace": trace, "evidence": registry.as_dicts(),
                    "citation_validation": validation, "model": MODEL,
                    "payload_accounting": session.payload_accounting,
                    "question_id": session.question_id,
                    "session_id": session.session_id,
                    "checkpoint_path": str(session.checkpoint_path) if session.checkpoint_path else None,
                    "usage_checkpoints": list(session.usage_checkpoints),
                    "per_question_input_tokens": session.per_question_input_tokens,
                    "evaluation_total_input_tokens": session.evaluation_total_input_tokens,
                }

            for function_call in function_calls:
                if session.tool_calls >= self.max_tool_calls:
                    break
                serialized_call = _raw_response_item_dict(function_call)
                if isinstance(function_call, dict):
                    name = function_call.get("name")
                    call_id = function_call.get("call_id")
                    raw_arguments = function_call.get("arguments", "{}")
                else:
                    name = getattr(function_call, "name", None)
                    call_id = getattr(function_call, "call_id", None)
                    raw_arguments = getattr(function_call, "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    observation = tools.execute(str(name), arguments, registry)
                    model_observation = compact_tool_observation(observation, session.emitted_evidence_ids)
                    session.tool_calls += 1
                    trace_arguments: Any = arguments
                except Exception as exc:
                    session.tool_calls += 1
                    observation = {"tool": name, "error": str(exc), "canonical_evidence": []}
                    model_observation = compact_tool_observation(observation, session.emitted_evidence_ids)
                    trace_arguments = raw_arguments
                function_call_output = serialize_function_call_output_for_resume(
                    call_id, _compact_json(model_observation)
                )
                input_items.append(function_call_output)
                pair = {
                    "function_call": serialized_call,
                    "call_id": call_id,
                    "tool_name": name,
                    "arguments": trace_arguments,
                    "compact_observation": model_observation,
                    "function_call_output": function_call_output,
                }
                session.function_call_pairs.append(pair)
                trace.append({
                    "call": session.tool_calls, "tool": name, "arguments": trace_arguments,
                    "observation": observation,
                    "model_observation": model_observation,
                })
            try:
                self._persist_checkpoint(
                    session, phase="ready_for_next_request",
                    response_output_items=response_output_items,
                    raw_response_output_items=raw_response_output_items,
                    response_id=response_id,
                    response_usage=response_usage,
                )
            except Exception as exc:
                return self._safe_failure(
                    f"Continuation checkpoint persistence failed: {exc}", registry, trace,
                    session.tool_calls, session.payload_accounting, session,
                )

        return self._safe_failure(
            f"Maximum ReAct tool-call limit reached ({self.max_tool_calls}) before a grounded final answer.",
            registry, trace, session.tool_calls, session.payload_accounting, session,
        )

    def answer(
        self,
        question: str,
        question_id: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._run_session(
            self.start_session(
                question,
                question_id=question_id,
                conversation_history=conversation_history,
            )
        )


def _likely_regulatory_question(question: str) -> bool:
    """Safety gate for code claims when the model gathered no evidence.

    This is not a retrieval router or an answer generator. It only prevents a
    no-tool model response from presenting an ungrounded regulatory claim.
    """

    text = re.sub(r"\s+", " ", str(question or "").strip().casefold())
    if not text:
        return False
    return bool(re.search(
        r"\b(section|chapter|table|equation|code|requirement|shall|required|"
        r"prohibit(?:ed|s)?|allow(?:ed|s)?|permission|exception|condition|"
        r"clearance|ventilation|refrigerant|duct|furnace|boiler|combustion|"
        r"hvac|mechanical)\b|\b\d{3,4}(?:\.\d+)+\b",
        text,
    ))


def _user_facing_failure(message: str) -> str:
    """Turn runtime failures into useful, non-internal UI guidance."""

    lowered = str(message or "").casefold()
    if "preflight" in lowered or "aura" in lowered or "connect" in lowered:
        return "I’m sorry, I can’t reach the HVAC code knowledge graph right now. Please try again in a moment."
    if "citation" in lowered or "evidence" in lowered or "grounded" in lowered:
        return (
            "I’m sorry, I couldn’t safely connect that answer to the HVAC code source. "
            "Please try a specific section number or a narrower code question."
        )
    if "responses api" in lowered or "openai" in lowered:
        return "I’m sorry, the answer service could not complete that request. Please try again shortly."
    return "I’m sorry, I couldn’t complete that safely. Please try a specific HVAC code section or requirement."


def query_agent(
    question: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    runtime = ReActGraphRAG()
    try:
        result = runtime.answer(question, conversation_history=conversation_history)
        if result.get("status") == "ok":
            return str(result["answer"])
        return _user_facing_failure(str(result.get("error", "unknown runtime error")))
    finally:
        if runtime.store is not None:
            runtime.store.close()


def get_statistics() -> dict[str, Any]:
    store = AuraReadOnlyStore.from_environment()
    try:
        return store.statistics()
    finally:
        store.close()
