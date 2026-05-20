import redis
import time
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
from app.generator import stream_answer

from fastapi.responses import StreamingResponse
from app.retriever_store import initialize_retriever
import json

app = FastAPI(title="RAG Assistant API")

redis_client = redis.Redis(
    host="localhost",
    port="6379",
    decode_responses=True,
)

def save_session(session_id: str, state: GraphState):
    redis_client.set(
        session_id,
        state.model_dump_json()
    )


def load_session(session_id: str) -> GraphState | None:
    data = redis_client.get(session_id)

    if data is None:
        return None

    return GraphState.model_validate_json(data)


def reset_turn_state(state: GraphState) -> None:
    """
    Clear fields that belong to the current graph turn only.
    """
    state.retrieved_docs = None
    state.answer = None
    state.needs_retrieval = None
    state.intent = None
    state.needs_web_search = None


graph = build_graph()


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

def run_chat(session_id: str, message: str) -> GraphState:
    """
    Shared chat execution pipeline.
    Handles:
    - session loading
    - graph execution
    - state persistence
    """

    # 1. Load session
    state = load_session(session_id)

    # 2. Create new state if missing
    if state is None:
        state = GraphState(
            query="",
            chat_history=[]
        )

    # 3. Update query
    state.query = message
    reset_turn_state(state)

    # 4. Run graph
    final_state = graph.invoke(state)

    print("DEBUG → final_state:", final_state)
    print("DEBUG → final chat_history:", final_state.get("chat_history"))

    # 5. Persist full state
    updated_state = GraphState(**final_state)

    save_session(session_id, updated_state)

    return updated_state


def fake_stream(text: str):
    """
    Simulate streaming response chunk-by-chunk.
    """

    words = text.split()

    for word in words:
        yield word + " "
        time.sleep(0.03)

@app.on_event("startup")
def startup_event():
    """
    Initialize shared application resources.
    """

    initialize_retriever()

# -------- Chat Endpoint --------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    session_id = request.session_id or str(uuid4())

    final_state = run_chat(
        session_id=session_id,
        message=request.message
    )

    return ChatResponse(
        session_id=session_id,
        answer=final_state.answer
    )

# -------- Chat Stream Endpoint --------
@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    session_id = request.session_id or str(uuid4())

    final_state = run_chat(
        session_id=session_id,
        message=request.message
    )

    answer = final_state.answer

    return StreamingResponse(
        fake_stream(answer),
        media_type="text/plain",
        headers={
            "X-Session-Id": session_id
        }
    )

# -------- Reset Endpoint --------
@app.post("/reset")
def reset(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "reset successful"}
