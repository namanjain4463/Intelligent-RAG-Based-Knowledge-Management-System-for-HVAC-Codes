import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
    def test_offline_library_and_empty_search_work_without_credentials(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'bot.py'),default_timeout=20).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual([t.label for t in app.tabs],['Ask the source','Section index','PDF reader'])
        app.text_input[0].set_value('no-such-section-xyz').run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('No matching section' in item.value for item in app.info))
        app.text_input[0].set_value('303.7').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.selectbox[0].value,'303.7')
