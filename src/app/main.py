from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4

from app.graph import build_graph
from app.state import GraphState

app = FastAPI(title="RAG Assistant API")

graph = build_graph()

# Simple in-memory session store
sessions = {}

# -------- Request Models --------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str

# -------- Health Check --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Chat Endpoint --------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # 1. Create new session if needed
    if request.session_id is None:
        session_id = str(uuid4())
        sessions[session_id] = GraphState(query= "",chat_history=[])
    else:
        session_id = request.session_id

        if session_id not in sessions:
            sessions[session_id] = GraphState(query= "",chat_history=[])

    state = sessions[session_id]

    # 2. Update query
    state.query = request.message

    # 3. Run graph
    final_state = graph.invoke(state)

    # 4. Save updated memory
    state.chat_history = final_state.get("chat_history", state.chat_history)

    # 5. Return JSON response
    return ChatResponse(
        session_id = session_id,
        answer = final_state["answer"],
        source = final_state.get("source")
    )

# -------- Reset Endpoint --------
@app.post("/reset")
def reset(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "reset successful"}

