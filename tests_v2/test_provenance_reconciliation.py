from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "v2_output"
AUDIT = OUTPUT / "audit"


def read_csv(name: str) -> list[dict[str, str]]:
    with (AUDIT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ProvenanceReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((AUDIT / "provenance_audit_summary.json").read_text(encoding="utf-8"))
        cls.document = json.loads((OUTPUT / "document.json").read_text(encoding="utf-8"))
        cls.reconciliation = read_csv("provenance_reconciliation.csv")
        cls.spans = read_csv("provenance_spans.csv")
        cls.classified = read_csv("unmatched_line_classification.csv")
        cls.excluded = read_csv("unmatched_excluded_source_lines.csv")
        cls.tables = read_csv("provenance_table_coverage.csv")
        cls.preserved = json.loads((AUDIT / "reconciliation_unparsed_blocks.json").read_text(encoding="utf-8"))

    def test_legacy_heading_inventory_is_fully_linked(self) -> None:
        headings = self.summary["legacy_heading_provenance"]
        self.assertEqual(headings["section_source_headings"], 584)
        self.assertEqual(headings["section_linked_headings"], 584)
        self.assertEqual(headings["chapter_source_headings"], 8)
        self.assertEqual(headings["chapter_linked_headings"], 8)
        self.assertEqual(headings["coverage"], 1.0)

    def test_equation_occurrence_provenance_is_complete(self) -> None:
        equations = self.summary["equation_provenance"]
        self.assertEqual(equations["source_occurrences"], 27)
        self.assertEqual(equations["linked_occurrences"], 27)
        self.assertEqual(equations["unique_source_identifiers"], 15)
        self.assertEqual(equations["unique_linked_identifiers"], 15)
        self.assertEqual(equations["coverage"], 1.0)

    def test_normative_acceptance_has_no_unexplained_spans_or_silent_drops(self) -> None:
        self.assertEqual(self.summary["status"], "PASS")
        self.assertEqual(self.summary["unmatched_normative_spans"], 0)
        self.assertEqual(self.summary["unassigned_normative_tokens"], 0)
        self.assertEqual(self.summary["silent_drop_count"], 0)
        self.assertEqual(len(self.spans), self.summary["normative_span_count"])
        self.assertTrue(all(row["target_id"] for row in self.spans))

    def test_original_643_breakdown_is_exhaustive(self) -> None:
        breakdown = self.summary["original_643_breakdown"]
        self.assertEqual(self.summary["original_643_candidate_lines"], 643)
        self.assertTrue(self.summary["original_643_breakdown_sum_check"])
        self.assertEqual(sum(item["count"] for item in breakdown.values()), 643)
        self.assertEqual(breakdown["existing-object alignment failures"]["count"], 446)
        self.assertEqual(breakdown["table representation differences"]["count"], 1)
        self.assertEqual(breakdown["list-marker representation differences"]["count"], 1)
        self.assertEqual(breakdown["genuine parser omissions"]["count"], 193)
        self.assertEqual(breakdown["false audit classifications"]["count"], 2)

    def test_page_120_markers_use_layout_proximity(self) -> None:
        markers = [row for row in self.reconciliation if row["source_kind"] == "list_marker"]
        self.assertGreaterEqual(len(markers), 5)
        self.assertTrue(all(row["page_no"] == "120" for row in markers))
        self.assertTrue(all("layout proximity" in row["match_strategy"] for row in markers))
        self.assertTrue(all(row["target_id"] for row in markers))

    def test_page_174_lines_are_not_navigation_exclusions(self) -> None:
        rows = [
            row for row in self.classified
            if row["page_no"] == "174" and row["legacy_audit_category"] == "navigation_artifact"
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["category"] == "normative code text" for row in rows))
        self.assertTrue(all(row["blocker"].casefold() == "true" for row in rows))
        self.assertTrue(all(row["legacy_audit_category"] == "navigation_artifact" for row in rows))
        self.assertFalse(any(row["page_no"] == "174" for row in self.excluded))

    def test_table_fragment_and_footnote_accounting_is_explicit(self) -> None:
        self.assertEqual(self.summary["table_fragment_count"], 56)
        self.assertLess(self.summary["table_cell_token_coverage"], 1.0)
        self.assertEqual(self.summary["table_cell_accounted_coverage"], 1.0)
        self.assertLess(self.summary["table_footnote_coverage"], 1.0)
        self.assertEqual(self.summary["table_footnote_accounted_coverage"], 1.0)
        self.assertEqual(self.summary["unassigned_table_cell_tokens"], 0)
        self.assertEqual(self.summary["unassigned_table_footnote_tokens"], 0)
        self.assertTrue(any("table footnote" in row["reason"] for row in self.preserved))

    def test_multi_page_table_has_one_canonical_object(self) -> None:
        tables = [table for table in self.document["tables"] if table.get("table_number") == "403.3.1.1"]
        self.assertEqual(len(tables), 1)
        self.assertEqual([fragment["page_no"] for fragment in tables[0]["fragments"]], [44, 45, 46, 47, 48])
        self.assertEqual(len(tables[0]["rows"]), 114)
        coverage = next(row for row in self.tables if row["table_number"] == "403.3.1.1")
        self.assertEqual(coverage["fragment_count"], "5")

    def test_preservation_records_are_explicit_unparsed_blocks(self) -> None:
        self.assertTrue(self.preserved)
        self.assertTrue(all(row["block_type"] == "unparsed" for row in self.preserved))
        self.assertEqual(len(self.preserved), self.summary["preservation_record_count"])
        self.assertTrue(any(row["id"].startswith("unparsed:reconciliation-span:") for row in self.preserved))

    def test_reconciliation_artifacts_exist(self) -> None:
        for filename in (
            "provenance_reconciliation.csv", "provenance_spans.csv",
            "provenance_category_summary.csv", "provenance_page_summary.csv",
            "provenance_table_coverage.csv", "reconciliation_unparsed_blocks.json",
        ):
            self.assertTrue((AUDIT / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
