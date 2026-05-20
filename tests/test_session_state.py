import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app.main as main
from app.state import GraphState


class FakeGraph:
    def __init__(self):
        self.input_state = None

    def invoke(self, state: GraphState):
        self.input_state = state
        return {
            "query": state.query,
            "retrieved_docs": state.retrieved_docs,
            "answer": "Simplified answer",
            "needs_retrieval": False,
            "intent": "followup",
            "needs_web_search": state.needs_web_search,
            "chat_history": state.chat_history
            + [
                f"User: {state.query}",
                "Assistant: Simplified answer",
            ],
        }


class SessionStateTests(unittest.TestCase):
    def test_run_chat_clears_transient_state_and_preserves_history(self):
        loaded_state = GraphState(
            query="Tell me about RAG in detail",
            retrieved_docs=["stale doc"],
            answer="Old answer",
            needs_retrieval=True,
            intent="knowledge",
            needs_web_search=True,
            chat_history=[
                "User: Tell me about RAG in detail",
                "Assistant: RAG combines retrieval with generation.",
            ],
        )
        fake_graph = FakeGraph()
        saved_states = []

        with (
            patch.object(main, "load_session", return_value=loaded_state),
            patch.object(main, "save_session", lambda session_id, state: saved_states.append(state)),
            patch.object(main, "graph", fake_graph),
            patch("builtins.print"),
        ):
            final_state = main.run_chat("session-1", "Explain it simply")

        self.assertEqual(fake_graph.input_state.query, "Explain it simply")
        self.assertIsNone(fake_graph.input_state.retrieved_docs)
        self.assertIsNone(fake_graph.input_state.answer)
        self.assertIsNone(fake_graph.input_state.needs_retrieval)
        self.assertIsNone(fake_graph.input_state.intent)
        self.assertIsNone(fake_graph.input_state.needs_web_search)

        self.assertIsNone(final_state.retrieved_docs)
        self.assertEqual(final_state.intent, "followup")
        self.assertIn("User: Tell me about RAG in detail", final_state.chat_history)
        self.assertIn("User: Explain it simply", final_state.chat_history)
        self.assertEqual(saved_states, [final_state])


if __name__ == "__main__":
    unittest.main()
