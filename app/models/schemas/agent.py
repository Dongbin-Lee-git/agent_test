from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class AddKnowledgeRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None

class KnowledgeResponse(BaseModel):
    status: str
    message: str

class StatsResponse(BaseModel):
    name: str
    count: int
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    user_query: str
    process_status: Optional[str] = None
    loop_count: Optional[int] = None
    build_logs: Optional[List[Message]] = None
    augment_logs: Optional[List[Message]] = None
    extract_logs: Optional[List[Message]] = None
    answer_logs: Optional[List[Message]] = None
    eval_logs: Optional[List[Message]] = None

class AgentRunRequest(BaseModel):
    inputs: Dict[str, Any]
    session_id: Optional[str] = None
