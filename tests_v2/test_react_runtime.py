import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2_ingestion.react_runtime import (
    build_payload_accounting,
    CanonicalEvidenceAssembler,
    compact_tool_observation,
    EvidenceRegistry,
    ReActGraphRAG,
    serialize_function_call_output_for_resume,
    serialize_response_output_for_resume,
    TOOL_SPECS,
    validate_read_only_cypher,
)


class ReactRuntimeSafetyTests(unittest.TestCase):
    def test_exactly_three_runtime_tools_are_exposed(self):
        self.assertEqual(
            [tool["name"] for tool in TOOL_SPECS],
            ["CypherSearch", "VectorSearch", "HybridSearch"],
        )

    def test_read_only_cypher_requires_bounded_query(self):
        query = "MATCH (s:Section {number: $number}) RETURN s.number AS section_number LIMIT 10"
        self.assertEqual(validate_read_only_cypher(query), query)
        with self.assertRaisesRegex(ValueError, "CREATE"):
            validate_read_only_cypher("MATCH (s:Section) CREATE (x:Section) RETURN x LIMIT 1")
        with self.assertRaisesRegex(ValueError, "LIMIT"):
            validate_read_only_cypher("MATCH (s:Section) RETURN s")
        with self.assertRaisesRegex(ValueError, "CALL"):
            validate_read_only_cypher("CALL db.index.vector.queryNodes('x', 1, $v) YIELD node RETURN node LIMIT 1")

    def test_evidence_registry_binds_and_rejects_citations(self):
        registry = EvidenceRegistry()
        first = registry.add("303.3", 10, "Sleeping rooms.", "Combustion air")
        duplicate = registry.add("303.3", 10, "Sleeping rooms.", "Combustion air")
        self.assertEqual(first, "E1")
        self.assertEqual(duplicate, first)
        rendered = registry.render("The location is prohibited. [E1]")
        self.assertTrue(rendered["valid"])
        self.assertIn("[Section 303.3, p. 10]", rendered["rendered_answer"])
        repeated = registry.render("The location is prohibited. [E1][E1]")
        self.assertEqual(repeated["rendered_answer"].count("[Section 303.3, p. 10]"), 1)
        rejected = registry.render("The location is prohibited. [E99]")
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["unsupported_evidence_ids"], ["E99"])
        forbidden = registry.render("The location is prohibited. [Section 303.3, p. 10]")
        self.assertFalse(forbidden["valid"])

    def test_canonical_assembler_preserves_prohibition_list_and_excludes_artifacts(self):
        assembler = CanonicalEvidenceAssembler()
        registry = EvidenceRegistry()
        result = assembler.assemble(["1109.2.3"], registry)
        text = "\n".join(block["source_text"] for block in result["evidence"])
        for item in (
            "Exposed within a fire-resistance-rated exit access corridor.",
            "Within an interior exit stairway.",
            "Within an interior exit ramp.",
            "Within an exit passageway.",
            "Within an elevator, dumbwaiter or other shaft containing a moving object.",
        ):
            self.assertIn(item, text)
        self.assertNotIn("INSIGHTS", text)
        self.assertNotIn("section:unassigned", text)

    def test_canonical_assembler_emits_complete_303_7_own_body(self):
        assembler = CanonicalEvidenceAssembler()
        result = assembler.assemble(["303.7"], EvidenceRegistry())
        text = "\n".join(block["source_text"] for block in result["evidence"] if block["section_number"] == "303.7")
        self.assertIn("not less than 3 inches (76 mm) above the pit floor", text)

    def test_canonical_assembler_emits_complete_302_3_1_own_body(self):
        assembler = CanonicalEvidenceAssembler()
        result = assembler.assemble(["302.3.1"], EvidenceRegistry())
        text = "\n".join(block["source_text"] for block in result["evidence"] if block["section_number"] == "302.3.1")
        self.assertIn(
            "Holes bored in joists shall not be within 2 inches (51 mm) of the top or bottom of the joist",
            text,
        )

    def test_lexical_fallback_recovers_structural_ventilation_clause(self):
        assembler = CanonicalEvidenceAssembler()
        matches = assembler.lexical_section_matches(
            "Are multiple fans allowed to provide the emergency ventilation rate?",
            limit=5,
        )
        self.assertTrue(matches)
        self.assertEqual(matches[0]["number"], "1105.6.3")

    def test_compaction_excludes_rich_requirement_and_graph_fields(self):
        registry = EvidenceRegistry()
        evidence_id = registry.add(
            "1105.6.3", 281,
            "Multiple fans or multispeed fans shall be allowed to produce the emergency ventilation rate.",
            "Ventilation rate",
        )
        raw = {
            "tool": "HybridSearch",
            "passages": [{
                "retrieval_hash": "hash-secret",
                "representative_requirement_id": "requirement-secret",
                "similarity": 0.99,
                "section": {"number": "1105.6.3", "title": "Ventilation rate", "page": 281},
                "requirements": [{"id": "requirement-secret", "source_text": "internal"}],
                "graph_section_requirements": [{"predicate": "shall", "object": "internal"}],
                "graph_ancestor_requirements": [{"retrieval_hash": "hash-secret"}],
            }],
            "canonical_evidence": registry.as_dicts(),
        }
        compact = compact_tool_observation(raw, set())
        text = json.dumps(compact, ensure_ascii=False)
        self.assertEqual(compact["new_evidence"][0]["evidence_id"], evidence_id)
        self.assertIn("Multiple fans or multispeed fans", text)
        for forbidden in (
            "retrieval_hash", "representative_requirement_id", "similarity",
            "requirements", "graph_section_requirements", "graph_ancestor_requirements",
        ):
            self.assertNotIn(forbidden, text)

    def test_evidence_ledger_deduplicates_and_keeps_ids_stable(self):
        registry = EvidenceRegistry()
        first_id = registry.add("303.3", 10, "1. Sleeping rooms.", "Prohibited locations")
        second_id = registry.add("303.3", 11, "Exception: Direct-vent appliances.", "Prohibited locations")
        emitted = set()
        first = compact_tool_observation(
            {"tool": "VectorSearch", "passages": [], "canonical_evidence": registry.as_dicts()[:1]},
            emitted,
        )
        second = compact_tool_observation(
            {"tool": "CypherSearch", "selected_sections": ["303.3"], "canonical_evidence": registry.as_dicts()},
            emitted,
        )
        self.assertEqual(first["new_evidence"][0]["evidence_id"], first_id)
        self.assertEqual(second["existing_evidence"], [first_id])
        self.assertEqual(second["new_evidence"][0]["evidence_id"], second_id)
        self.assertEqual(registry.render("The rule applies. [E1] [E2]")["valid"], True)

    def test_compaction_preserves_complete_lists_and_exceptions(self):
        assembler = CanonicalEvidenceAssembler()
        registry = EvidenceRegistry()
        result = assembler.assemble(["303.3"], registry)
        compact = compact_tool_observation(
            {"tool": "CypherSearch", "selected_sections": result["selected_sections"], "canonical_evidence": result["evidence"]},
            set(),
        )
        text = "\n".join(item["source_text"] for item in compact["new_evidence"])
        self.assertIn("Sleeping rooms", text)
        self.assertIn("Surgical rooms", text)
        self.assertIn("Exception:", text)
        self.assertIn("Direct-vent appliances", text)

    def test_simulated_multitool_observation_can_reuse_and_add_evidence(self):
        registry = EvidenceRegistry()
        first_id = registry.add("303.3", 10, "Prohibited locations.", "Prohibited locations")
        second_id = registry.add("303.3", 11, "Exception: Direct-vent appliances.", "Prohibited locations")
        emitted = set()
        first = compact_tool_observation(
            {"tool": "VectorSearch", "passages": [{"section": {"number": "303.3", "title": "Prohibited locations", "page": 10}}], "canonical_evidence": [registry.as_dicts()[0]]},
            emitted,
        )
        second = compact_tool_observation(
            {"tool": "HybridSearch", "passages": [{"section": {"number": "303.3", "title": "Prohibited locations", "page": 10}}], "canonical_evidence": registry.as_dicts()},
            emitted,
        )
        self.assertEqual(first["new_evidence"][0]["evidence_id"], first_id)
        self.assertEqual(second["existing_evidence"], [first_id])
        self.assertEqual(second["new_evidence"][0]["evidence_id"], second_id)
        self.assertEqual(registry.render("See [E2] and [E1].")["valid"], True)

    def test_payload_accounting_reports_compact_observation_components(self):
        registry = EvidenceRegistry()
        registry.add("1105.6.3", 281, "Multiple fans are allowed.", "Ventilation rate")
        input_items = [
            {"role": "user", "content": "Are multiple fans allowed?"},
            {"type": "function_call", "name": "HybridSearch", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": '{"tool":"HybridSearch"}'},
        ]
        accounting = build_payload_accounting(input_items[0]["content"], input_items, registry)
        self.assertGreater(accounting["system_instruction_tokens"], 0)
        self.assertGreater(accounting["tool_schema_tokens"], 0)
        self.assertGreater(accounting["new_tool_observation_tokens"], 0)
        self.assertGreater(accounting["cumulative_unique_evidence_tokens"], 0)
        self.assertEqual(
            accounting["approximate_total_input_tokens"],
            accounting["system_instruction_tokens"]
            + accounting["tool_schema_tokens"]
            + accounting["question_tokens"]
            + accounting["prior_compact_history_tokens"]
            + accounting["new_tool_observation_tokens"],
        )

    def test_new_question_resets_ledger_ids_history_and_tool_counter(self):
        runtime = ReActGraphRAG()
        first = runtime.start_session("Q1: Section 303.3", question_id="Q1")
        first.registry.add("303.3", 10, "Q1 evidence", "Prohibited locations")
        first.input_items.append({"type": "function_call_output", "output": "Q1 tool output"})
        first.tool_calls = 2
        first.previous_response_id = "response-q1"
        runtime.evaluation_total_input_tokens = 17

        second = runtime.start_session("Q2: emergency ventilation", question_id="Q2")
        self.assertEqual(second.registry.as_dicts(), [])
        self.assertEqual(second.registry.add("1105.6.3", 281, "Q2 evidence", "Ventilation rate"), "E1")
        self.assertEqual(second.tool_calls, 0)
        self.assertIsNone(second.previous_response_id)
        self.assertEqual(second.per_question_input_tokens, 0)
        self.assertEqual(second.evaluation_total_input_tokens, 17)
        self.assertEqual(second.input_items, [{"role": "user", "content": "Q2: emergency ventilation"}])
        self.assertNotIn("Q1", json.dumps(second.input_items))
        self.assertNotIn("Q1 evidence", json.dumps(second.input_items))

    def test_initial_requests_for_independent_questions_have_stable_size(self):
        runtime = ReActGraphRAG()
        questions = {
            "Q1": "What does Section 303.3 prohibit, and what exceptions apply?",
            "Q2": "Are multiple fans allowed to provide the emergency ventilation rate?",
            "Q3": "Where is refrigerant piping prohibited, and are there any relevant exceptions or qualifications?",
        }
        requests = [runtime.initial_request(runtime.start_session(text, question_id=qid)) for qid, text in questions.items()]
        serialized = [json.dumps(request, ensure_ascii=False, separators=(",", ":")) for request in requests]
        sizes = [len(item) for item in serialized]
        self.assertLess(max(sizes) - min(sizes), 200)
        q3_input = json.dumps(requests[2]["input"], ensure_ascii=False)
        self.assertNotIn("Q1 evidence", q3_input)
        self.assertNotIn("Q2 evidence", q3_input)
        self.assertNotIn("Q1: Section 303.3", q3_input)
        self.assertNotIn("Q2: emergency ventilation", q3_input)
        self.assertNotIn("previous_response_id", requests[2])

    def test_same_question_history_survives_across_turns(self):
        runtime = ReActGraphRAG()
        session = runtime.start_session("same question", question_id="same")
        session.input_items.extend([
            {"type": "function_call", "name": "HybridSearch", "call_id": "c1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": '{"new_evidence":[{"evidence_id":"E1"}]}'},
        ])
        self.assertIn("HybridSearch", json.dumps(session.input_items))
        self.assertIn("E1", json.dumps(session.input_items))
        fresh = runtime.start_session("new question", question_id="new")
        self.assertNotIn("HybridSearch", json.dumps(fresh.input_items))
        self.assertNotIn("E1", json.dumps(fresh.input_items))

    def test_usage_checkpoint_is_available_when_next_request_is_blocked(self):
        class FailingResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("budget stop")
                return SimpleNamespace(
                    id="resp-q1-1",
                    usage=SimpleNamespace(input_tokens=17, output_tokens=5),
                    output=[SimpleNamespace(
                        type="function_call", name="CypherSearch", call_id="call-1",
                        arguments=json.dumps({
                            "query": "MATCH (s:Section {number: $number}) RETURN s.number AS section_number LIMIT 1",
                            "parameters": {"number": "303.3"},
                        }),
                    )],
                    output_text="",
                )

        class FakeClient:
            def __init__(self):
                self.responses = FailingResponses()

        class FakeStore:
            def read_cypher(self, query, parameters):
                return [{"section_number": parameters["number"]}]

            def close(self):
                return None

        runtime = ReActGraphRAG(client=FakeClient(), store=FakeStore())
        result = runtime.answer("What does Section 303.3 prohibit?", question_id="Q1")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["usage_checkpoints"][0]["response_id"], "resp-q1-1")
        self.assertEqual(result["usage_checkpoints"][0]["question_id"], "Q1")
        self.assertEqual(result["usage_checkpoints"][0]["actual_input_tokens"], 17)
        self.assertEqual(result["usage_checkpoints"][0]["actual_output_tokens"], 5)
        self.assertEqual(result["usage_checkpoints"][0]["completed"], {"type": "tool_call", "tools": ["CypherSearch"]})
        self.assertEqual(result["per_question_input_tokens"], 17)
        self.assertEqual(result["evaluation_total_input_tokens"], 17)

    def test_checkpoint_retains_call_pair_reasoning_and_request_configuration(self):
        call = {
            "type": "function_call", "id": "fc_item_1", "call_id": "call_1",
            "name": "CypherSearch",
            "arguments": json.dumps({
                "query": "MATCH (s:Section) RETURN s.number AS section_number LIMIT 1",
                "parameters": {},
            }),
        }
        reasoning = {
            "type": "reasoning", "id": "rs_1",
            "encrypted_content": "opaque-encrypted-reasoning",
        }

        class Responses:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return SimpleNamespace(
                        id="resp_1", usage=SimpleNamespace(input_tokens=10, output_tokens=3),
                        output=[reasoning, call], output_text="",
                    )
                return SimpleNamespace(
                    id="resp_2", usage=SimpleNamespace(input_tokens=12, output_tokens=4),
                    output=[], output_text="The rule applies. [E1]",
                )

        class Client:
            def __init__(self):
                self.responses = Responses()

        class Store:
            def read_cypher(self, query, parameters):
                return [{"section_number": "303.3"}]

        with tempfile.TemporaryDirectory() as directory:
            runtime = ReActGraphRAG(
                client=Client(), store=Store(), checkpoint_dir=Path(directory),
            )
            result = runtime.answer("What applies?", question_id="checkpoint-1")
            self.assertEqual(result["status"], "ok")
            checkpoint = json.loads(Path(result["checkpoint_path"]).read_text(encoding="utf-8"))

        self.assertEqual(checkpoint["phase"], "answer_ready")
        self.assertEqual(checkpoint["function_call_pairs"][0]["call_id"], "call_1")
        self.assertEqual(
            checkpoint["function_call_pairs"][0]["function_call_output"]["call_id"],
            "call_1",
        )
        self.assertEqual(
            checkpoint["response_output_history"][0]["output_items"][0]["encrypted_content"],
            "opaque-encrypted-reasoning",
        )
        self.assertIn("call_1", json.dumps(checkpoint["resume_input_items"]))
        self.assertIn("function_call_output", json.dumps(checkpoint["resume_input_items"]))
        self.assertEqual(checkpoint["request_configuration"]["store"], False)
        self.assertEqual(checkpoint["request_configuration"]["include"], ["reasoning.encrypted_content"])

    def test_resume_serializer_reasoning_excludes_status_and_metadata(self):
        item = SimpleNamespace(
            type="reasoning", id="rs_1", summary=[{"text": "opaque summary"}],
            encrypted_content="opaque-reasoning", status="completed",
            extra_metadata="must-not-resume",
        )
        serialized = serialize_response_output_for_resume(item)
        self.assertEqual(
            serialized,
            {
                "type": "reasoning", "id": "rs_1",
                "summary": [{"text": "opaque summary"}],
                "encrypted_content": "opaque-reasoning",
            },
        )
        self.assertNotIn("status", serialized)
        self.assertNotIn("extra_metadata", serialized)

    def test_resume_serializer_function_call_excludes_status_and_debug_fields(self):
        item = {
            "type": "function_call", "id": "fc_item", "call_id": "call_1",
            "name": "CypherSearch", "arguments": "{}", "status": "completed",
            "parsed": {"query": "debug"}, "debug": "must-not-resume",
        }
        serialized = serialize_response_output_for_resume(item)
        self.assertEqual(
            set(serialized), {"type", "call_id", "name", "arguments", "id"}
        )
        self.assertEqual(serialized["call_id"], "call_1")
        self.assertNotIn("status", serialized)
        self.assertNotIn("parsed", serialized)
        self.assertNotIn("debug", serialized)

    def test_checkpoint_round_trip_preserves_resume_order_and_call_pairing(self):
        runtime = ReActGraphRAG()
        session = runtime.start_session("checkpoint round trip", question_id="round-trip")
        reasoning = serialize_response_output_for_resume({
            "type": "reasoning", "id": "rs_1", "summary": [],
            "encrypted_content": "opaque", "status": "completed",
        })
        function_call = serialize_response_output_for_resume({
            "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "VectorSearch", "arguments": "{\"query\":\"x\",\"top_k\":1}",
            "status": "completed",
        })
        function_output = serialize_function_call_output_for_resume(
            "call_1", '{"tool":"VectorSearch"}'
        )
        session.input_items.extend([reasoning, function_call, function_output])
        session.turn = 1
        session.tool_calls = 1
        session.last_response_output_items = [reasoning, function_call]
        session.last_raw_response_output_items = [{
            **reasoning, "status": "completed",
        }, {
            **function_call, "status": "completed", "parsed": {"debug": True},
        }]
        session.raw_response_output_history = [{
            "turn": 1, "response_id": "resp_1",
            "output_items": session.last_raw_response_output_items,
        }]
        session.response_output_history = [{
            "turn": 1, "response_id": "resp_1",
            "output_items": session.last_response_output_items,
        }]
        session.function_call_pairs = [{
            "function_call": session.last_raw_response_output_items[1],
            "call_id": "call_1", "tool_name": "VectorSearch",
            "arguments": {"query": "x", "top_k": 1},
            "compact_observation": {"tool": "VectorSearch"},
            "function_call_output": function_output,
        }]
        with tempfile.TemporaryDirectory() as directory:
            session.checkpoint_path = Path(directory) / "checkpoint.json"
            runtime._persist_checkpoint(
                session, "ready_for_next_request",
                response_output_items=session.last_response_output_items,
                raw_response_output_items=session.last_raw_response_output_items,
                response_id="resp_1",
                response_usage={"input_tokens": 1, "output_tokens": 1},
            )
            loaded = runtime.load_checkpoint(session.checkpoint_path)

        self.assertEqual(
            [item["type"] for item in loaded.input_items[1:]],
            ["reasoning", "function_call", "function_call_output"],
        )
        self.assertEqual(loaded.input_items[2]["call_id"], "call_1")
        self.assertEqual(loaded.input_items[3]["call_id"], "call_1")
        self.assertNotIn("status", loaded.input_items[1])
        self.assertNotIn("status", loaded.input_items[2])
        self.assertEqual(loaded.function_call_pairs[0]["call_id"], "call_1")
        self.assertEqual(loaded.function_call_pairs[0]["function_call_output"]["call_id"], "call_1")
        self.assertIn("status", loaded.last_raw_response_output_items[0])
        self.assertIn("parsed", loaded.last_raw_response_output_items[1])

    def test_aura_preflight_failure_prevents_first_openai_request(self):
        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise AssertionError("OpenAI request must not be constructed")

        class Client:
            def __init__(self):
                self.responses = Responses()

        class UnavailableAura:
            def __init__(self):
                self.preflight_calls = 0

            def preflight(self):
                self.preflight_calls += 1
                return False

        client = Client()
        store = UnavailableAura()
        runtime = ReActGraphRAG(client=client, store=store)
        result = runtime.answer("Can this be answered?", question_id="preflight")
        self.assertEqual(result["status"], "failed")
        self.assertIn("preflight", result["error"].lower())
        self.assertEqual(store.preflight_calls, 1)
        self.assertEqual(client.responses.calls, 0)

    def test_multiple_tool_turns_resume_without_rerunning_completed_tools(self):
        def tool_call(call_id):
            return {
                "type": "function_call", "id": f"item_{call_id}", "call_id": call_id,
                "name": "CypherSearch",
                "arguments": json.dumps({
                    "query": "MATCH (s:Section) RETURN s.number AS section_number LIMIT 1",
                    "parameters": {},
                }),
            }

        class FirstResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls <= 2:
                    return SimpleNamespace(
                        id=f"resp_{self.calls}",
                        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                        output=[tool_call(f"call_{self.calls}")], output_text="",
                    )
                raise RuntimeError("budget stop")

        class FirstClient:
            def __init__(self):
                self.responses = FirstResponses()

        class CountingStore:
            def __init__(self, fail=False):
                self.calls = 0
                self.fail = fail

            def read_cypher(self, query, parameters):
                self.calls += 1
                if self.fail:
                    raise AssertionError("completed tool was rerun during resume")
                return [{"section_number": "303.3"}]

        class ResumeResponses:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    id="resp_final", usage=SimpleNamespace(input_tokens=15, output_tokens=5),
                    output=[], output_text="The rule applies. [E1]",
                )

        class ResumeClient:
            def __init__(self):
                self.responses = ResumeResponses()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            first_store = CountingStore()
            first_runtime = ReActGraphRAG(
                client=FirstClient(), store=first_store, checkpoint_dir=checkpoint_dir,
            )
            failed = first_runtime.answer("What applies?", question_id="resume-1")
            self.assertEqual(failed["status"], "failed")
            checkpoint_path = Path(failed["checkpoint_path"])
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(len(checkpoint["function_call_pairs"]), 2)
            self.assertEqual(
                [pair["call_id"] for pair in checkpoint["function_call_pairs"]],
                ["call_1", "call_2"],
            )
            self.assertEqual(
                [block["evidence_id"] for block in checkpoint["evidence_ledger"]],
                [f"E{i}" for i in range(1, len(checkpoint["evidence_ledger"]) + 1)],
            )

            resume_client = ResumeClient()
            resume_store = CountingStore(fail=True)
            resumed_runtime = ReActGraphRAG(
                client=resume_client, store=resume_store, checkpoint_dir=checkpoint_dir,
            )
            resumed = resumed_runtime.resume_from_checkpoint(checkpoint_path)
            self.assertEqual(resumed["status"], "ok")
            self.assertEqual(resume_store.calls, 0)
            request = resume_client.responses.calls[0]
            serialized_input = json.dumps(request["input"], ensure_ascii=False)
            self.assertIn("call_1", serialized_input)
            self.assertIn("call_2", serialized_input)
            self.assertEqual(request["include"], ["reasoning.encrypted_content"])
            self.assertEqual(resumed["citation_validation"]["cited_evidence_ids"], ["E1"])

    def test_checkpoint_resume_preserves_question_isolation(self):
        runtime = ReActGraphRAG()
        first = runtime.start_session("Q1 Section 303.3", question_id="Q1", session_id="s1")
        first.registry.add("303.3", 10, "Q1 evidence", "Prohibited locations")
        first.input_items.extend([
            {"type": "function_call", "call_id": "q1-call"},
            {"type": "function_call_output", "call_id": "q1-call", "output": "Q1"},
        ])
        second = runtime.start_session("Q2 ventilation", question_id="Q2", session_id="s2")
        self.assertEqual(second.registry.as_dicts(), [])
        self.assertEqual(second.input_items, [{"role": "user", "content": "Q2 ventilation"}])
        self.assertEqual(second.tool_calls, 0)
        self.assertIsNone(second.previous_response_id)
        self.assertNotIn("Q1", json.dumps(second.input_items))
        self.assertNotIn("q1-call", json.dumps(second.input_items))

    def test_conversation_memory_is_copied_without_react_state(self):
        runtime = ReActGraphRAG()
        session = runtime.start_session(
            "Does that exception apply here?",
            question_id="follow-up",
            conversation_history=[
                {"role": "user", "content": "What does Section 303.3 prohibit?"},
                {"role": "assistant", "content": "It identifies prohibited locations. [Section 303.3, p. 10]"},
                {"role": "function_call", "content": "must not be copied"},
            ],
        )
        self.assertEqual(
            session.input_items,
            [
                {"role": "user", "content": "What does Section 303.3 prohibit?"},
                {"role": "assistant", "content": "It identifies prohibited locations. [Section 303.3, p. 10]"},
                {"role": "user", "content": "Does that exception apply here?"},
            ],
        )
        self.assertEqual(session.tool_calls, 0)
        self.assertEqual(session.registry.as_dicts(), [])

    def test_guardrail_distinguishes_conversation_from_code_claims(self):
        from v2_ingestion.react_runtime import _likely_regulatory_question

        self.assertFalse(_likely_regulatory_question("hi, what is your name?"))
        self.assertFalse(_likely_regulatory_question("Can you help me understand this topic?"))
        self.assertFalse(_likely_regulatory_question(
            "What all can you do? Give me the code for 1D analysis of a vapour chamber."
        ))
        self.assertTrue(_likely_regulatory_question("What does Section 303.3 prohibit?"))
        self.assertTrue(_likely_regulatory_question("Are multiple fans allowed for ventilation?"))

    def test_non_regulatory_answer_without_evidence_is_allowed(self):
        class Responses:
            def create(self, **kwargs):
                return SimpleNamespace(
                    id="resp-chat", usage=SimpleNamespace(input_tokens=3, output_tokens=5),
                    output=[], output_text="Hello! I can help you explore the HVAC code corpus.",
                )

        class Client:
            responses = Responses()

        class Store:
            def preflight(self):
                return True

        with tempfile.TemporaryDirectory() as directory:
            runtime = ReActGraphRAG(
                client=Client(), store=Store(), checkpoint_dir=Path(directory),
            )
            result = runtime.answer("Hi, what can you help me with?", question_id="chat")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["citation_validation"]["not_required"])
        self.assertIn("HVAC", result["answer"])


if __name__ == "__main__":
    unittest.main()
