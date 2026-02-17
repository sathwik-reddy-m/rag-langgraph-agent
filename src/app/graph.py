from langgraph.graph import StateGraph, END 

from app.state import GraphState
from app.nodes import retrieve_node, generate_node, decide_retrieval_node, conversational_fallback_node

def build_graph():
    """
    Build and compile the LangGraph RAG agent.
    """

    # 1. Create a graph with shared state
    graph = StateGraph(GraphState)

    # 2. Register nodes
    graph.add_node("decide", decide_retrieval_node)
    graph.add_node("fallback", conversational_fallback_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # 3. Define execution order
    graph.set_entry_point("decide")

    # conditional routing
    graph.add_conditional_edges(
        "decide", 
        lambda state: (
            "retrieve" if state.intent == "knowledge"
            else "fallback" if state.intent == "greeting"
            else "generate"
        ),
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    # 4. Compile the graph
    return graph.compile()
