from typing import TypedDict, Annotated, List, Optional, Union
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class InfoBuildAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class InfoExtractAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class AnswerGenAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class EvaluateAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class MainState(TypedDict):
    user_query: str
    build_logs: List[BaseMessage]
    augment_logs: List[BaseMessage]
    extract_logs: List[BaseMessage]
    answer_logs: Annotated[List[BaseMessage], add_messages]
    eval_logs: List[BaseMessage]
    process_status: str
    loop_count: int
