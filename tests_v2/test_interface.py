import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
    def test_offline_library_and_empty_search_work_without_credentials(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'bot.py'),default_timeout=20).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(list(app.radio[0].options),['Chat','Sections','PDF'])
        app.radio[0].set_value('Sections').run()
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

    def test_three_turns_keep_one_pair_per_submission(self):
        from unittest.mock import patch
        def answer(question, history, on_progress=None):
            on_progress('Checking HVAC conversation scope')
            return {'status':'general','answer':'HVAC answer','claims':[],'evidence':[]}
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'bot.py'),default_timeout=20).run()
        with patch('v2_ingestion.react_runtime.query_agent_result',side_effect=answer) as query:
            for question in ['What is HVAC?','What is a heat pump?','What is ventilation?']:
                app.chat_input[0].set_value(question).run()
                self.assertEqual(len(app.exception),0)
            self.assertEqual(query.call_count,3)
            self.assertEqual(len(app.chat_message),6)
            self.assertEqual(len(app.status),0)
            self.assertEqual(len(query.call_args_list[2].args[1]),4)

    def test_failed_question_retry_and_chat_recall(self):
        from unittest.mock import Mock, patch
        runtime=Mock()
        runtime.answer.side_effect=[{'status':'failed','answer':'Connection failed'}, {'status':'ok','answer':'Recovered test answer','claims':[],'evidence':[]}]
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'bot.py'),default_timeout=20).run()
        with patch('v2_ingestion.react_runtime.ReActGraphRAG',return_value=runtime), patch.dict('os.environ',{'HVAC_SPEND_LEDGER':''}):
            for question in ['What does Section 303.7 require?','try again','What did you just say?']:
                app.chat_input[0].set_value(question).run()
                self.assertEqual(len(app.exception),0)
        self.assertEqual(runtime.answer.call_count,2)
        self.assertTrue(all(c.args[0]=='What does Section 303.7 require?' for c in runtime.answer.call_args_list))
        self.assertEqual(len(app.chat_message),6)
        self.assertTrue(any('Recovered test answer' in m.value for m in app.markdown))
        self.assertFalse(any('proof of concept' in c.value.lower() for c in app.caption))
