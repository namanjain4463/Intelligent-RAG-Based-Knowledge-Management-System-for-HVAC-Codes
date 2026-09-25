import json, unittest
from pathlib import Path

class EvaluationIntegrityTests(unittest.TestCase):
    def test_dataset_has_disjoint_splits_and_valid_source_hashes(self):
        root=Path(__file__).resolve().parents[1]
        data=json.loads((root/'evals/cases.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['cases']),140)
        ids=[c['id'] for c in data['cases']];self.assertEqual(len(set(ids)),len(ids))
        groups={name:{s for c in data['cases'] if c['split']==name for s in c['expected_sections']} for name in ['heldout','development']}
        self.assertFalse(groups['heldout']&groups['development'])
        document=json.loads((root/'v2_output/document.json').read_text(encoding='utf-8'))
        self.assertEqual(data['source_sha256'],document['source_sha256'])
        for case in data['cases']:
            if case['expected_sections']:
                self.assertTrue(case['source_spans'])
                self.assertTrue(set(case['expected_sections'])<={s['section_number'] for s in case['source_spans']})
