import re

from app.state import GraphState
from app.retriever import retrieve_documents
from app.generator import generate_answer
from langchain_groq import ChatGroq
from tavily import TavilyClient

VALID_INTENTS = {"greeting", "followup", "knowledge"}


def normalize_intent(intent: str) -> str | None:
    """
    Normalize classifier output into one of the supported route labels.
    """
    normalized = re.sub(r"[^a-z]", "", intent.lower())
    return normalized if normalized in VALID_INTENTS else None


def is_greeting(query: str) -> bool:
    """
    Detect common greetings and acknowledgements without using retrieval.
    """
    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    greeting_patterns = [
        r"^(hi|hello|hey|yo)$",
        r"^(thanks|thank you|thx|ty)$",
        r"^(thanks|thank you) .+",
        r"^(good morning|good afternoon|good evening)$",
    ]

    return any(re.search(pattern, normalized_query) for pattern in greeting_patterns)


def is_contextual_followup(query: str, chat_history: list[str] | None) -> bool:
    """
    Detect short, high-confidence follow-ups before calling the LLM router.
    """
    if not chat_history:
        return False

    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    if not normalized_query:
        return False

    contextual_references = [
        "it",
        "this",
        "that",
        "above",
        "previous",
        "earlier",
        "last answer",
        "same",
        "concept",
    ]
    followup_actions = [
        "explain",
        "simplify",
        "summarize",
        "continue",
        "elaborate",
        "expand",
        "clarify",
        "give examples",
    ]
    short_followups = {
        "continue",
        "summarize",
        "simplify",
        "more details",
        "explain simply",
        "explain in simple terms",
        "explain it simply",
        "explain this simply",
        "explain that simply",
    }

    words = re.findall(r"\b\w+\b", normalized_query)

    if normalized_query in short_followups:
        return True

    has_reference = any(
        re.search(rf"\b{re.escape(reference)}\b", normalized_query)
        for reference in contextual_references
    )
    has_action = any(action in normalized_query for action in followup_actions)

    if has_reference and has_action:
        return True

    if has_reference and len(words) <= 6:
        return True

    return False


def decide_retrieval_node(state: GraphState) -> dict:
    if is_greeting(state.query):
        return {
            "intent": "greeting",
            "needs_retrieval": False,
        }

    if is_contextual_followup(state.query, state.chat_history):
        return {
            "intent": "followup",
            "needs_retrieval": False,
        }

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
- If there is prior conversation and the query is short, pronoun-based, or not standalone-answerable, prefer followup.
- Prefer knowledge only when the user introduces a standalone new topic or asks for external factual information.

Examples:
- Previous topic: RAG. User query: Explain it simply. Answer: followup
- User query: Explain above concept in simple terms. Answer: followup
- User query: Tell me about Redis persistence. Answer: knowledge
- User query: thanks. Answer: greeting

Conversation so far:
{history}

User query:
{state.query}

Reply with only one word:
greeting
followup
knowledge
"""

    raw_intent = llm.invoke(prompt).content.strip()
    intent = normalize_intent(raw_intent)

    if intent is None:
        intent = (
            "followup"
            if is_contextual_followup(state.query, state.chat_history)
            else "knowledge"
        )

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

    # --- Intelligent Routing Logic ---
    time_sensitive_keywords = [
        "current",
        "latest",
        "today",
        "recent",
        "breaking",
        "now",
        "news",
        "present",
        "who is current",
    ]

    is_time_sensitive = any(word in query for word in time_sensitive_keywords)



    # ---- Confidence-based fallback ----
    if not scores:
        best_score = None
        # No vector knowledge available
        if is_time_sensitive:
            needs_web_search = True
        else:
            needs_web_search = False
    else:
        best_score = min(scores)

        if best_score <= 0.95:
            # Strong local match
            needs_web_search = False
        else:
            # Weak local match
            if is_time_sensitive:
                needs_web_search = True
            else:
                # Let LLM answer directly
                needs_web_search = False

    print("DEBUG → best_score:", best_score)
    print("DEBUG → time_sensitive:", is_time_sensitive)
    print("DEBUG → needs_web_search:", needs_web_search)

    return {
        "retrieved_docs": filtered_docs,
        "needs_web_search": needs_web_search,
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

    updated_history = (history + [
        f"User: {query}",
        f"Assistant: {answer}"
    ])[-10:] # keep last 10 turns only
 
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

    updated_history = (history + [
        f"User: {state.query}",
        f"Assistant: {answer}"
    ])[-10:] # keep last 10 turns only

    return {
        "answer": answer,
        "chat_history": updated_history
    }

def web_search_node(state: GraphState) -> dict:
    """
    Perform web search when local knowledge is insufficient.
    """
    print("DEBUG → Web search triggered")

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
        "retrieved_docs": contents,
    }
