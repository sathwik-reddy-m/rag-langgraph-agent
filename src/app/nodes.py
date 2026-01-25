from app.state import GraphState
from app.retriever import retrieve_documents

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
