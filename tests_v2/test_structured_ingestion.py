from __future__ import annotations

import json
import copy
import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from v2_ingestion.models import ListBlock
from v2_ingestion.parser import DoclingStructuralParser
from v2_ingestion.source_inventory import build_source_inventory
from v2_ingestion.writer import write_outputs
from tests_v2.audit_fixtures import audit_output


REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "HVAC-Codes.pdf"
OUTPUT = REPO / "v2_output"


def prov(page: int = 10, top: float = 20, bottom: float = 40) -> list[dict]:
    return [{"page_no": page, "bbox": {"l": 20, "t": top, "r": 580, "b": bottom}, "charspan": [0, 20]}]


def fake_export() -> dict:
    return {
        "schema_name": "DoclingDocument",
        "version": "2.0.0",
        "body": [
            {"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/groups/0"},
            {"$ref": "#/texts/3"}, {"$ref": "#/tables/0"}, {"$ref": "#/texts/4"},
            {"$ref": "#/formulas/0"}, {"$ref": "#/texts/5"},
        ],
        "texts": [
            {"self_ref": "#/texts/0", "label": "chapter", "text": "CHAPTER 3 GENERAL REGULATIONS", "prov": prov(2)},
            {"self_ref": "#/texts/1", "label": "section_header", "text": "303.3 Prohibited locations.", "prov": prov(10)},
            {"self_ref": "#/texts/2", "label": "list_item", "text": "Sleeping rooms", "prov": prov(10, 50, 60)},
            {"self_ref": "#/texts/3", "label": "text", "text": "Exception: This section shall not apply to the following appliances: 1. Direct-vent appliances that obtain all combustion air directly from the outdoors.", "prov": prov(11, 70, 90)},
            {"self_ref": "#/texts/4", "label": "text", "text": "Reference to the source", "target": "https://codes.iccsafe.org/", "prov": prov(10, 100, 120)},
            {"self_ref": "#/texts/5", "label": "text", "text": "Table note", "prov": prov(19, 200, 220)},
        ],
        "groups": [
            {"self_ref": "#/groups/0", "label": "list", "children": [{"$ref": "#/texts/2"}], "prov": prov(10, 45, 65)},
        ],
        "formulas": [
            {"self_ref": "#/formulas/0", "label": "formula", "text": "Q = m c_p ΔT", "latex": "Q = m c_p \\Delta T", "prov": prov(19, 230, 250)},
        ],
        "tables": [
            {
                "self_ref": "#/tables/0", "label": "table", "captions": ["#/texts/5"],
                "footnotes": ["#/texts/5"], "prov": prov(19, 20, 180),
                "data": {"num_rows": 2, "num_cols": 2, "table_cells": [
                    {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "text": "Nominal size", "column_header": True, "bbox": {"l": 20, "t": 30, "r": 200, "b": 50}},
                    {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "text": "Maximum spacing", "column_header": True, "bbox": {"l": 200, "t": 30, "r": 400, "b": 50}},
                    {"start_row_offset_idx": 1, "start_col_offset_idx": 0, "text": "1 inch", "bbox": {"l": 20, "t": 60, "r": 200, "b": 80}},
                    {"start_row_offset_idx": 1, "start_col_offset_idx": 1, "text": "6 feet", "bbox": {"l": 200, "t": 60, "r": 400, "b": 80}},
                ]},
            }
        ],
    }


def nested_list_export() -> dict:
    export = copy.deepcopy(fake_export())
    export["groups"].append({
        "self_ref": "#/groups/1", "label": "list", "children": [{"$ref": "#/texts/2"}], "prov": prov(10, 45, 65),
    })
    export["groups"][0]["children"] = [{"$ref": "#/groups/1"}]
    return export


class StructuralIngestionTests(unittest.TestCase):
    def test_difficult_section_preserves_list_and_exception_without_semantics(self) -> None:
        document = DoclingStructuralParser(PDF).parse_export(fake_export())
        section = next(section for section in document.sections if section.number == "303.3")
        self.assertEqual(section.title, "Prohibited locations")
        self.assertTrue(any(block.block_type == "list" for block in section.blocks))
        self.assertTrue(any(block.block_type == "exception" for block in section.blocks))
        list_block = next(block for block in section.blocks if block.block_type == "list")
        self.assertEqual(list_block.items[0].text, "Sleeping rooms")
        self.assertFalse(hasattr(list_block.items[0], "equipment"))

    def test_table_rows_footnotes_and_cell_provenance_are_retained(self) -> None:
        document = DoclingStructuralParser(PDF).parse_export(fake_export())
        self.assertEqual(len(document.tables), 1)
        table = document.tables[0]
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.rows[1].cells, ["1 inch", "6 feet"])
        self.assertEqual(table.rows[0].cell_provenance[0].page_no, 19)
        self.assertEqual(table.title, "Table note")
        self.assertEqual(table.footnotes[0].text, "Table note")

    def test_missing_provenance_is_reported(self) -> None:
        export = {"body": [{"$ref": "#/texts/0"}], "texts": [{"self_ref": "#/texts/0", "label": "text", "text": "orphan"}]}
        document = DoclingStructuralParser(PDF).parse_export(export)
        self.assertTrue(any(failure.code == "MISSING_PROVENANCE" for failure in document.failures))

    def test_requested_artifacts_round_trip(self) -> None:
        document = DoclingStructuralParser(PDF).parse_export(fake_export())
        with tempfile.TemporaryDirectory() as temp:
            paths = write_outputs(document, temp)
            self.assertEqual(set(paths), {"document", "sections", "tables", "table_rows", "references", "equations"})
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertEqual(len(pd.read_parquet(paths["tables"])), 1)
            self.assertEqual(len(pd.read_parquet(paths["table_rows"])), 2)
            payload = json.loads(paths["document"].read_text(encoding="utf-8"))
            self.assertIn("failures", payload)
            self.assertEqual(payload["source_sha256"], document.source_sha256)

    def test_nested_lists_are_not_flattened(self) -> None:
        document = DoclingStructuralParser(PDF).parse_export(nested_list_export())
        outer = next(block for section in document.sections for block in section.blocks if block.block_type == "list")
        self.assertIsInstance(outer.items[0], ListBlock)
        self.assertEqual(outer.items[0].items[0].text, "Sleeping rooms")


class SourceCoverageGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads((OUTPUT / "document.json").read_text(encoding="utf-8"))
        cls.summary = json.loads((OUTPUT / "audit" / "audit_summary.json").read_text(encoding="utf-8"))

    def section(self, number: str) -> dict:
        return next(section for section in self.document["sections"] if section.get("number") == number)

    def table(self, number: str) -> dict:
        return next(table for table in self.document["tables"] if table.get("table_number") == number)

    def equation(self, number: str) -> dict:
        return next(equation for equation in self.document["equations"] if equation.get("equation_number") == number)

    def test_independent_source_inventory_is_present(self) -> None:
        inventory = build_source_inventory(PDF)
        self.assertEqual(inventory.pages, 306)
        self.assertEqual(len(inventory.section_headings), 897)
        self.assertEqual(len(inventory.unique_table_numbers), 30)
        self.assertEqual(len(inventory.unique_equation_numbers), 15)
        self.assertEqual(len(inventory.exception_blocks()), 125)

    def test_section_303_3_title_and_complete_exception_structure(self) -> None:
        section = self.section("303.3")
        self.assertEqual(section["title"], "Prohibited locations")
        exception = next(block for block in section["blocks"] if block.get("block_type") == "exception")
        self.assertEqual(exception["content_class"], "Exception")
        self.assertEqual(len(exception["items"]), 3)
        self.assertIn("Direct-vent appliances", exception["items"][0]["text"])
        self.assertIn("Solid fuel-fired appliances", exception["items"][1]["text"])
        self.assertIn("dedicated enclosure", exception["items"][2]["text"])
        self.assertIn("approved self-closing device", exception["text"])
        self.assertNotIn("303.4 Protection from damage", exception["text"])

    def test_chapter_10_and_11_hierarchy(self) -> None:
        chapters = {chapter["number"]: chapter for chapter in self.document["chapters"]}
        self.assertIn("10", chapters)
        self.assertIn("11", chapters)
        section_1001 = self.section("1001")
        section_1001_1 = self.section("1001.1")
        section_1101 = self.section("1101")
        section_1101_1 = self.section("1101.1")
        self.assertEqual(section_1001["chapter_id"], "chapter:10")
        self.assertEqual(section_1001_1["chapter_id"], "chapter:10")
        self.assertEqual(section_1001_1["parent_section_id"], "section:1001")
        self.assertEqual(section_1001_1["level"], 2)
        self.assertEqual(section_1101["chapter_id"], "chapter:11")
        self.assertEqual(section_1101_1["chapter_id"], "chapter:11")
        self.assertEqual(section_1101_1["parent_section_id"], "section:1101")

    def test_table_305_4_columns_rows_footnotes_and_provenance(self) -> None:
        table = self.table("305.4")
        self.assertIn("PIPING SUPPORT SPACING", table["title"])
        self.assertEqual(table["num_cols"], 3)
        self.assertEqual(len(table["rows"]), 19)
        self.assertGreaterEqual(len(table["footnotes"]), 1)
        self.assertEqual(table["provenance"]["page_no"], 19)
        self.assertIsNotNone(table["provenance"]["bbox"])

    def test_table_403_3_1_1_is_one_logical_multi_page_table(self) -> None:
        matches = [table for table in self.document["tables"] if table.get("table_number") == "403.3.1.1"]
        self.assertEqual(len(matches), 1)
        table = matches[0]
        self.assertEqual([fragment["page_no"] for fragment in table["fragments"]], [44, 45, 46, 47, 48])
        self.assertEqual(len(table["rows"]), 114)
        self.assertGreater(len(table["fragments"]), 1)

    def test_equation_4_1_and_variables(self) -> None:
        equation = self.equation("4-1")
        self.assertEqual(equation["provenance"]["page_no"], 49)
        self.assertEqual(set(["Vbz", "Rp", "Pz", "Ra", "Az"]), set(equation["variables"]))
        self.assertIn("Vbz", equation["text"])

    def test_equation_4_6_and_later_chapter_equation(self) -> None:
        self.assertIn("Vou", self.equation("4-6")["text"])
        self.assertIn("11-2", self.equation("11-2")["text"])
        self.assertEqual(self.equation("11-2")["provenance"]["page_no"], 282)

    def test_typed_references_and_separate_commentary_classes(self) -> None:
        references = self.document["references"]
        self.assertTrue(any(reference["reference_type"] == "section" and "301.3" in reference["target"] for reference in references))
        self.assertTrue(any(reference["reference_type"] == "table" and "403.3.1.1" in reference["target"] for reference in references))
        self.assertTrue(any(reference["reference_type"] == "chapter" and "Chapter 7" in reference["target"] for reference in references))
        self.assertTrue(any(reference["reference_type"] == "external_standard" and any(token in reference["target"].upper() for token in ("ASTM", "NFPA", "ANSI")) for reference in references))
        classes = {
            block.get("content_class")
            for section in self.document["sections"]
            for block in section.get("blocks", [])
        }
        self.assertIn("Insights", classes)
        self.assertIn("UserNote", classes)
        self.assertIn("Exception", classes)
        self.assertIn("UnparsedBlock", classes)

    def test_duplicate_detection_and_zero_silent_data_loss(self) -> None:
        for key in ("sections", "tables", "equations"):
            ids = [item["id"] for item in self.document[key]]
            self.assertEqual(len(ids), len(set(ids)), key)
        numbered_tables = [table["table_number"] for table in self.document["tables"] if table.get("table_number")]
        equation_numbers = [equation["equation_number"] for equation in self.document["equations"]]
        self.assertEqual(len(numbered_tables), len(set(numbered_tables)))
        self.assertEqual(len(equation_numbers), len(set(equation_numbers)))
        self.assertGreater(self.summary["coverage"]["unmatched_content_record_count"], 0)
        self.assertGreater(len(self.document["unparsed_blocks"]), 0)
        self.assertEqual(self.summary["coverage"]["section_heading_recall"], 1.0)
        self.assertEqual(self.summary["coverage"]["table_caption_recall"], 1.0)
        self.assertEqual(self.summary["coverage"]["equation_identifier_recall"], 1.0)
        for filename in ("section_coverage.csv", "table_coverage.csv", "equation_coverage.csv", "reference_coverage.csv", "unmatched_content.csv"):
            self.assertTrue((audit_output() / "audit" / filename).exists())
        with (audit_output() / "audit" / "unmatched_content.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(any(row["status"] == "unparsed_output" for row in rows))


if __name__ == "__main__":
    unittest.main()
