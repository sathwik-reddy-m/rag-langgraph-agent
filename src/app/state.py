from typing import List, Optional
from pydantic import BaseModel

class GraphState(BaseModel):
    """
    Shared state that flows through the LangGraph
    """

    query: str
    retrieved_docs: Optional[list[str]] = None
    answer: Optional[str] = None
    needs_retrieval: Optional[bool] = None
    intent: Optional[str] = None 
    needs_web_search: Optional[bool] = None

