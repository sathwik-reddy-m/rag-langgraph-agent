from app.state import GraphState
from app.retriever import retrieve_documents
from app.generator import generate_answer
from langchain_groq import ChatGroq

def decide_retrieval_node(state: GraphState) -> dict:
    """
    Decide user intent:
    - knowledge
    - conversation
    - greeting
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    prompt = f"""
Classify the user's query into ONE category.

Categories:
- greeting (hi, hello, thanks, etc.)
- conversation (who are you, how are you, general chat)
- knowledge (requires external knowledge)

Reply with ONLY one word:
greeting
conversation
knowledge

Query:
{state.query}
"""

    intent = llm.invoke(prompt).content.strip().lower()

    return {
        "intent": intent,
        "needs_retrieval": intent == "knowledge",
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

def conversational_fallback_node(state: GraphState) -> dict:
    """
    Handle greetings and small talk.
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
    )

    prompt = f"""
You are a friendly AI assistant.
Respond naturally and briefly to the user.

User:
{state.query}
"""

    answer = llm.invoke(prompt).content.strip()

    return {
        "answer": answer
    }