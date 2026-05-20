import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app.graph as graph_module
import app.nodes as nodes
from app.state import GraphState


RAG_HISTORY = [
    "User: Tell me about RAG in detail",
    "Assistant: RAG combines retrieval with generation to answer using external context.",
]


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self.response)


class RoutingTests(unittest.TestCase):
    def test_short_contextual_followup_routes_without_llm(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("LLM classifier should not run for heuristic followups")

        with patch.object(nodes, "ChatGroq", fail_if_called):
            result = nodes.decide_retrieval_node(
                GraphState(query="Explain it simply", chat_history=RAG_HISTORY)
            )

        self.assertEqual(result, {"intent": "followup", "needs_retrieval": False})

    def test_above_concept_followup_routes_without_llm(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("LLM classifier should not run for heuristic followups")

        with patch.object(nodes, "ChatGroq", fail_if_called):
            result = nodes.decide_retrieval_node(
                GraphState(
                    query="Explain above concept in simple terms",
                    chat_history=RAG_HISTORY,
                )
            )

        self.assertEqual(result["intent"], "followup")
        self.assertFalse(result["needs_retrieval"])

    def test_new_topic_with_history_can_route_to_knowledge(self):
        with patch.object(nodes, "ChatGroq", lambda **kwargs: FakeLLM("knowledge")):
            result = nodes.decide_retrieval_node(
                GraphState(query="Tell me about Redis persistence", chat_history=RAG_HISTORY)
            )

        self.assertEqual(result, {"intent": "knowledge", "needs_retrieval": True})

    def test_short_ambiguous_query_without_history_is_not_forced_to_followup(self):
        with patch.object(nodes, "ChatGroq", lambda **kwargs: FakeLLM("knowledge")):
            result = nodes.decide_retrieval_node(
                GraphState(query="Explain it simply", chat_history=[])
            )

        self.assertEqual(result, {"intent": "knowledge", "needs_retrieval": True})

    def test_greeting_classifier_output_routes_to_conversation(self):
        with patch.object(nodes, "ChatGroq", lambda **kwargs: FakeLLM("thanks.")):
            result = nodes.decide_retrieval_node(
                GraphState(query="thanks", chat_history=RAG_HISTORY)
            )

        self.assertEqual(result, {"intent": "greeting", "needs_retrieval": False})

    def test_invalid_classifier_output_defaults_safely(self):
        with patch.object(nodes, "ChatGroq", lambda **kwargs: FakeLLM("not sure")):
            result = nodes.decide_retrieval_node(
                GraphState(query="Tell me about Redis persistence", chat_history=RAG_HISTORY)
            )

        self.assertEqual(result, {"intent": "knowledge", "needs_retrieval": True})

    def test_followup_route_does_not_call_retrieval(self):
        def fail_retrieval(state):
            raise AssertionError("Retrieval should not run for followup intent")

        with (
            patch.object(
                graph_module,
                "decide_retrieval_node",
                lambda state: {"intent": "followup", "needs_retrieval": False},
            ),
            patch.object(
                graph_module,
                "conversational_fallback_node",
                lambda state: {
                    "answer": "Simplified answer",
                    "chat_history": state.chat_history,
                },
            ),
            patch.object(graph_module, "retrieve_node", fail_retrieval),
        ):
            graph = graph_module.build_graph()
            result = graph.invoke(
                GraphState(query="Explain it simply", chat_history=RAG_HISTORY)
            )

        self.assertEqual(result["intent"], "followup")
        self.assertEqual(result["answer"], "Simplified answer")


if __name__ == "__main__":
    unittest.main()
