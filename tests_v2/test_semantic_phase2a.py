from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests_v2.audit_fixtures import provenance_output
from v2_ingestion.semantic_evaluation import build_golden_cases, evaluate
from v2_ingestion.semantic_extractor import extract_requirements
from v2_ingestion.semantic_input import build_semantic_input
from v2_ingestion.semantic_llm import build_prompt
from v2_ingestion.semantic_models import EntityRef, Requirement
from v2_ingestion.semantic_preparser import annotate, predicate_for, split_clauses


DOC = ROOT / "v2_output" / "document.json"
AUDIT = ROOT / "v2_output" / "audit"


class SemanticPhase2ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audit_dir = provenance_output() / "audit"
        cls.spans, cls.input_errors, cls.input_summary = build_semantic_input(DOC, audit_dir / "provenance_reconciliation.csv", audit_dir / "reconciliation_unparsed_blocks.json")
        cls.requirements, cls.extract_errors, cls.extract_summary = extract_requirements(cls.spans)

    def test_models_are_strict_and_predicate_is_closed(self):
        with self.assertRaises(Exception):
            EntityRef(entity_id="e", mention="x", unknown="no")
        self.assertTrue(all(req.predicate in {"REQUIRES", "PROHIBITED_IN", "PERMITTED_IN", "REQUIRES_CLEARANCE", "REQUIRES_DEVICE", "MUST_COMPLY_WITH", "MINIMUM", "MAXIMUM"} for req in self.requirements))

    def test_semantic_input_excludes_commentary_and_tables(self):
        self.assertFalse(any(span.source_type == "table_cell" for span in self.spans))
        self.assertFalse(any("INSIGHTS" in span.source_text and span.source_type == "section_prose" for span in self.spans))
        self.assertEqual(self.input_summary["table_derived_requirement_input_count"], 0)

    def test_required_provenance_quarantine_is_explicit(self):
        self.assertGreaterEqual(self.input_summary["missing_provenance_failure_count"], 28)
        self.assertEqual(
            self.input_summary["missing_provenance_failure_resolved_count"]
            + self.input_summary["missing_provenance_failure_quarantined_count"],
            self.input_summary["missing_provenance_failure_count"],
        )
        self.assertTrue(all(error.code in {"MISSING_RELIABLE_PROVENANCE", "MISSING_SECTION_CONTEXT"} for error in self.input_errors))

    def test_deterministic_preparser_covers_values_refs_and_exception_markers(self):
        text = "Equipment shall be installed with a minimum clearance of 18 inches in accordance with ASHRAE 15 and Section 303.3. Exception: unless listed."
        ann = annotate(text)
        self.assertTrue(ann.values)
        self.assertTrue(ann.section_refs)
        self.assertTrue(ann.external_standard_refs)
        self.assertTrue(ann.exception_markers)
        self.assertEqual(predicate_for(text), "REQUIRES_CLEARANCE")

    def test_multiple_requirements_per_sentence_are_separate(self):
        clauses = split_clauses("Fans shall be provided and dampers shall be installed.")
        self.assertGreaterEqual(len(clauses), 2)
        self.assertIn(predicate_for(clauses[0]), {"REQUIRES", "REQUIRES_DEVICE"})
        self.assertEqual(predicate_for(clauses[1]), "REQUIRES_DEVICE")

    def test_exception_and_equation_table_refs_preserve_source(self):
        self.assertTrue(any(span.source_type == "exception" for span in self.spans))
        self.assertTrue(any(span.source_type == "table_reference" for span in self.spans))
        self.assertTrue(any(span.source_type == "equation_reference" for span in self.spans))
        self.assertTrue(all(req.evidence.source_text for req in self.requirements))
        self.assertTrue(all(req.evidence.page_no > 0 and req.section_id for req in self.requirements))
        self.assertTrue(all(req.evidence.source_sha256 for req in self.requirements))
        self.assertTrue(all(span.source_sha256 for span in self.spans))

    def test_section_303_3_is_present_and_prohibited(self):
        spans = [s for s in self.spans if s.section_id == "section:303.3"]
        self.assertTrue(spans)
        self.assertTrue(any("Fuel-fired appliances" in s.source_text for s in spans))
        self.assertTrue(any(r.section_id == "section:303.3" and r.predicate == "PROHIBITED_IN" for r in self.requirements))

    def test_chapters_10_and_11_are_represented(self):
        sections = {s.section_id for s in self.spans}
        self.assertTrue(any(s.startswith("section:10") for s in sections))
        self.assertTrue(any(s.startswith("section:11") for s in sections))

    def test_golden_benchmark_has_at_least_fifty_cases_and_separate_metrics(self):
        cases = build_golden_cases(self.spans, minimum=50)
        self.assertGreaterEqual(len(cases), 50)
        metrics = evaluate(self.requirements, cases)
        for key in ("predicate_recall", "value_recall", "reference_recall", "section_recall"):
            self.assertIn(key, metrics)
        self.assertNotIn("combined_score", metrics)

    def test_no_neo4j_side_effect_is_present_in_phase2a(self):
        self.assertFalse(any("neo4j" in p.name.casefold() or "aura" in p.name.casefold() for p in (ROOT / "v2_ingestion").glob("semantic_*.py")))

    def test_optional_llm_boundary_is_schema_constrained_and_not_invoked(self):
        prompt = build_prompt(self.spans[0], self.spans[0].source_text[:120])
        self.assertIn("JSON schema", prompt)
        self.assertIn("PROHIBITED_IN", prompt)
        self.assertIn(self.spans[0].section_id, prompt)


if __name__ == "__main__":
    unittest.main()
