import copy
import json
import unittest
from pathlib import Path
from v2_ingestion.equation_links import equation_owner, equation_source, reconcile_equation_records
from v2_ingestion.react_runtime import CanonicalEvidenceAssembler, EvidenceRegistry
from v2_ingestion.source_inventory import build_source_inventory

ROOT = Path(__file__).resolve().parents[1]


class EquationLinkTests(unittest.TestCase):
    def test_pdf_fallback_parser_links_formulas_to_source_headings(self):
        from v2_ingestion.parser import DoclingStructuralParser
        from tests_v2.test_structured_ingestion import fake_export
        document = DoclingStructuralParser(ROOT / 'HVAC-Codes.pdf').parse_export(fake_export())
        equation = next(e for e in document.equations if e.equation_number == '4-2')
        self.assertEqual(equation.section_id, 'section:403.3.1.1.1.3')
        owner = next(s for s in document.sections if s.id == equation.section_id)
        self.assertIn(equation.id, owner.equation_ids)

    def test_formula_before_first_heading_continues_previous_page(self):
        headings = [{'number': '301.1', 'page_no': 1, 'bbox': {'top': 500}},
                    {'number': '301.2', 'page_no': 2, 'bbox': {'top': 300}}]
        source = {'page_no': 2, 'bbox': {'top': 70}}
        self.assertEqual(equation_owner(source, headings, {'section:301.1', 'section:301.2'}), 'section:301.1')
        source['bbox']['top'] = 350
        self.assertEqual(equation_owner(source, headings, {'section:301.1', 'section:301.2'}), 'section:301.2')

    def test_formula_line_wins_over_whole_page_fallback(self):
        whole_page = {'text': 'Equation 4-9 formula=x', 'bbox': {'top': 0}}
        line = {'text': 'Equation 4-9 formula=x', 'bbox': {'top': 650}}
        self.assertIs(equation_source([whole_page, line]), line)

    def test_committed_equation_ownership_and_reverse_links(self):
        document = json.loads((ROOT / 'v2_output/document.json').read_text(encoding='utf-8'))
        equations = {e['equation_number']: e for e in document['equations']}
        expected = {'4-2': '403.3.1.1.1.3', '4-3': '403.3.1.1.2.1', '4-4': '403.3.1.1.2.2',
                    '4-5': '403.3.1.1.2.3.1', '4-7': '403.3.1.1.2.3.3', '4-8': '403.3.1.1.2.3.4',
                    '5-1': '502.6.2', '5-3': '513.10.1', '11-1': '1105.6', '11-2': '1105.6.3.2'}
        sections = {s['id']: s for s in document['sections']}
        for number, section in expected.items():
            with self.subTest(equation=number):
                e = equations[number]
                self.assertEqual(e['section_id'], f'section:{section}')
                self.assertIn(e['id'], sections[e['section_id']]['equation_ids'])
        for e in equations.values():
            self.assertEqual(sum(e['id'] in s['equation_ids'] for s in sections.values()), 1)
        before = copy.deepcopy(document)
        self.assertEqual(reconcile_equation_records(document, build_source_inventory(ROOT / 'HVAC-Codes.pdf')), [])
        self.assertEqual(document, before)

    def test_equation_is_evidence_for_its_physical_owner(self):
        assembler = CanonicalEvidenceAssembler()
        right = assembler.assemble(['513.10.1'], EvidenceRegistry())['evidence']
        wrong = assembler.assemble(['513.10.2'], EvidenceRegistry())['evidence']
        self.assertTrue(any('Ts=(Qc/mc)+(Ta)' in e['source_text'] for e in right))
        self.assertFalse(any('Ts=(Qc/mc)+(Ta)' in e['source_text'] for e in wrong))
