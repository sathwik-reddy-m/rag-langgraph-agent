from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

from app.graph import build_graph
from app.state import GraphState

if __name__ == "__main__":
    # 1. Build the graph
    graph = build_graph()

    print("\n🤖 RAG Assistant Started!")
    print("Type your question below.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    # 2. Create initial state
    state = GraphState(
        query="",
        chat_history=[]
    )

    while True:

        query= input("\nYou: ")
        if query.lower() in ["exit","quit"]:
            break

        state.query = query

        final_state = graph.invoke(state)

        print("\nAssistant:", final_state["answer"])
        
        state.chat_history = final_state.get("chat_history",state.chat_history)

