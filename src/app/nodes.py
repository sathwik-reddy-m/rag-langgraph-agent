from app.state import GraphState
from app.retriever import retrieve_documents
from app.generator import generate_answer

def decide_retrieval_node(state: GraphState) -> dict:
    """
    Decide whether retrieval is needed for the given query.
    """

    query = state.query.lower()

    # Simple heuristic (can be replaced with LLM later)
    needs_retrieval = not any(
        phrase in query
        for phrase in ["hi", "hello", "who are you", "what can you do"]
    )

    return {
        "needs_retrieval": needs_retrieval
    }

def retrieve_node(state: GraphState) -> dict:
    """
    LangGraph node that retrieves relevant documents
    based on the User's query.
    """

    # 1. Read from the state
    query = state.query

    # 2. Call existing retrieval logic
    docs = retrieve_documents(query)

    # 3. Extract raw text content
    contents = [doc.page_content for doc in docs]

    # 4. Return partial state update
    return {
        "retrieved_docs": contents
    }

def generate_node(state: GraphState) -> dict:
    """
    LangGraph node that generates an answer
    using retrieved documents as context.
    """

    # 1. Read from state
    query = state.query
    documents = state.retrieved_docs or []

    # 2. Call existing generation logic
    answer = generate_answer(query, documents)

    # 3. Return partial state update
    return {
        "answer": answer
    }

