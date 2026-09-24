"""Regression tests for bounded retrieval and final-turn synthesis; no services."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from v2_ingestion.react_runtime import (
    AuraReadOnlyStore, QUERY_TIMEOUT_SECONDS, ReActGraphRAG,
    validate_read_only_cypher,
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_limit_bypasses_are_rejected(self):
        for query in (
            "MATCH (s:Section) RETURN 'LIMIT 1' AS label, s",
            "MATCH (s:Section) RETURN s LIMIT 1 + 100000",
            "WITH 1 AS x LIMIT 1 MATCH (s:Section) RETURN s",
            "MATCH (s:Section) RETURN s LIMIT 1 UNION MATCH (s:Section) RETURN s",
            "MATCH (s:Section) RETURN s UNION MATCH (s:Section) RETURN s LIMIT 1",
            "MATCH (s:Section) RETURN s AS `LIMIT 1`",
            "MATCH (s:Section) RETURN s LIMIT 1000",
        ):
            with self.subTest(query=query), self.assertRaises(ValueError):
                validate_read_only_cypher(query)

    def test_normal_bounded_queries_remain_supported(self):
        for query in (
            "MATCH (s:Section) RETURN s.number LIMIT 10;",
            "MATCH (s:Section) WHERE s.number = $number RETURN s.number LIMIT 1",
            "MATCH (s:Section) RETURN count(s) AS total_sections LIMIT 1",
        ):
            self.assertEqual(validate_read_only_cypher(query), query)

    def test_generated_query_uses_transaction_timeout(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = Mock()
        tx.run.return_value.data.return_value = [{"section_number": "303.3"}]
        def execute(read):
            self.assertEqual(read.timeout, QUERY_TIMEOUT_SECONDS)
            return read(tx)
        session.execute_read.side_effect = execute
        store = AuraReadOnlyStore("unused", "unused", "unused", "test", driver=driver)
        self.assertEqual(store.read_cypher("RETURN $number AS section_number LIMIT 1", {"number": "303.3"}), [{"section_number": "303.3"}])

    def run_last_tool_scenario(self, ignore_tool_choice=False, interrupt=False):
        call = {"type": "function_call", "name": "CypherSearch", "call_id": "call_1",
                "arguments": json.dumps({"query": "MATCH (s:Section) RETURN s.number AS section_number LIMIT 1", "parameters": {}})}
        responses = Mock()
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(output=[call], output_text="")
            self.assertEqual(kwargs["tool_choice"], "none")
            if interrupt and len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return SimpleNamespace(output=[call] if ignore_tool_choice else [], output_text="The rule applies. [E1]")
        responses.create.side_effect = create
        store = Mock(spec=["read_cypher"])
        store.read_cypher.return_value = [{"section_number": "303.3"}]
        with tempfile.TemporaryDirectory() as directory:
            runtime = ReActGraphRAG(store=store, client=SimpleNamespace(responses=responses), max_tool_calls=1, checkpoint_dir=Path(directory))
            result = runtime.answer("What does Section 303.3 prohibit?")
            if interrupt:
                self.assertEqual(result["status"], "failed")
                result = runtime.resume_from_checkpoint(result["checkpoint_path"])
        self.assertEqual(store.read_cypher.call_count, 1)
        self.assertEqual(len(calls), 3 if interrupt else 2)
        return result

    def test_last_allowed_tool_can_be_used_to_answer(self):
        self.assertEqual(self.run_last_tool_scenario()["status"], "ok")

    def test_model_cannot_exceed_tool_budget(self):
        self.assertEqual(self.run_last_tool_scenario(ignore_tool_choice=True)["status"], "failed")

    def test_resume_at_tool_budget_synthesizes_without_retrieval(self):
        self.assertEqual(self.run_last_tool_scenario(interrupt=True)["status"], "ok")
