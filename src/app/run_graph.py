from app.graph import build_graph
from app.state import GraphState

if __name__ == "__main__":
    # 1. Build the graph
    graph = build_graph()

    # 2. Create initial state
    initial_state = GraphState(
        query="Who is the longest serving Oracle CEO?"
    )

    # 3. Run the graph
    final_state = graph.invoke(initial_state)

    # 4. Print results
    print("\nQuestion:", initial_state.query)
    print("\nAnswer:\n", final_state["answer"])

