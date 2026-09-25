import unittest
from v2_ingestion.inspector import build_inspection, citation_impact

class InspectorTests(unittest.TestCase):
    def test_removing_shared_reference_exposes_dependencies_not_truth(self):
        claims=[{'text':'A','evidence_ids':['E1','E2']},{'text':'B','evidence_ids':['E1']}]
        self.assertEqual([r['status'] for r in citation_impact(claims,['E1'])],['partial','uncovered'])
        self.assertEqual([r['status'] for r in citation_impact(claims,['unknown'])],['unchanged','unchanged'])

    def test_graph_uses_real_parent_edges_and_export_omits_internal_data(self):
        sections=[{'id':'p','number':'3'},{'id':'c','number':'3.1','parent_section_id':'p'}]
        result={'checkpoint_path':'private','raw_answer':'private','tool_trace':[{'tool':'CypherSearch','arguments':{'password':'private'},'observation':{'selected_sections':['3.1']}}],
                'claims':[{'text':'A','evidence_ids':['E1']}], 'evidence':[{'evidence_id':'E1','section_number':'3.1','page':2}]}
        report=build_inspection(result,sections)
        self.assertIn('contains',report['graph_dot'])
        self.assertEqual(report['steps'][0]['sections'],['3.1'])
        self.assertNotIn('private',str(report))
        sections[0]['parent_section_id']='c'
        self.assertTrue(build_inspection(result,sections)['graph_dot'])
