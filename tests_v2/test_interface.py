import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
    def test_offline_library_and_empty_search_work_without_credentials(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'bot.py'),default_timeout=20).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual([t.label for t in app.tabs],['Chat','Sections','PDF'])
        app.text_input[0].set_value('no-such-section-xyz').run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('No matching section' in item.value for item in app.info))
        app.text_input[0].set_value('303.7').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.selectbox[0].value,'303.7')

    def test_inspector_reference_removal_is_offline(self):
        from unittest.mock import patch
        result={'status':'ok','answer':'Example answer','claims':[{'text':'Example claim','evidence_ids':['E1']}],
                'evidence':[{'evidence_id':'E1','section_number':'303.7','section_title':'Example','source_text':'Example passage for UI fixture only.','page':12}],
                'tool_trace':[{'tool':'VectorSearch','observation':{'selected_sections':['303.7']}}]}
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'bot.py'),default_timeout=20)
        app.session_state['messages']=[{'role':'assistant','content':result['answer'],'result':result}]
        with patch('v2_ingestion.react_runtime.query_agent_result',side_effect=AssertionError('Inspector must stay offline')):
            app.run()
            self.assertEqual(len(app.exception),0)
            app.multiselect[0].set_value(['E1']).run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('All citations removed' in m.value for m in app.markdown))
