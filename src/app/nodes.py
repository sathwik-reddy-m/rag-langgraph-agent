from app.state import GraphState
from app.retriever import retrieve_documents
from app.generator import generate_answer
from langchain_groq import ChatGroq
from tavily import TavilyClient

def decide_retrieval_node(state: GraphState) -> dict:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    history = "\n".join((state.chat_history or [])[-6:])

    prompt = f"""
You are an intent classifier for a RAG system.

Classify the user query into ONE of the following categories:

1. greeting → simple hello, thanks, etc.
2. followup → refers to something mentioned earlier in the conversation
3. knowledge → requires external factual knowledge or new topic

Rules:
- If the query depends on previous conversation context, classify as followup.
- If the query introduces a new topic or entity, classify as knowledge.
- If unsure, prefer knowledge.

Conversation so far:
{history}

User query:
{state.query}

Reply with only one word:
greeting
followup
knowledge
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
    docs_with_scores = retrieve_documents(query)

    # 3. Extract raw text content
    threshold = 1.0
    filtered_docs = []
    scores = []

    for doc, score in docs_with_scores:
        scores.append(score)
        if score < threshold:
            filtered_docs.append(doc.page_content)

    # 4. Return partial state update
    return {
        "retrieved_docs": filtered_docs,
        "needs_web_search": len(filtered_docs) == 0,
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

    history = state.chat_history or []  

    updated_history = history + [
        f"User: {query}",
        f"Assistant: {answer}"
    ]

    # 3. Return partial state update
    return {
        "answer": answer,
        "chat_history": updated_history
    }

def conversational_fallback_node(state: GraphState) -> dict:
    """
    Handle greetings and small talk.
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
    )

    history = state.chat_history or []
    formatted_history = "\n".join(history[-6:]) #small window

    prompt = f"""
You are a helpful conversational AI assistant.

Conversation so far:
{formatted_history}

User:
{state.query}

If the question refers to something earlier, infer it from the conversation.
Respond naturally and briefly.
"""

    answer = llm.invoke(prompt).content.strip()

    updated_history = history + [
        f"User: {state.query}",
        f"Assistant: {answer}"
    ]

    return {
        "answer": answer,
        "chat_history": updated_history
    }

def web_search_node(state: GraphState) -> dict:
    """
    Perform web search when local knowledge is insufficient.
    """

    client = TavilyClient()

    results = client.search(
        query=state.query,
        search_depth="basic",
        max_results=5,
    )

    # Extract text content
    contents = [
        item["content"]
        for item in results.get("results", [])
        if "content" in item
    ]

    return {
        "retrieved_docs": contents
    }