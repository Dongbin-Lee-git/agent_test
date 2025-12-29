from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json

from app.models.schemas import (
    AddKnowledgeRequest, 
    KnowledgeResponse, 
    StatsResponse, 
    ChatRequest,
    ChatResponse,
    AgentRunRequest
)
from app.deps import get_agent_service
from app.service.agent_service import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])




@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
):
    try:
        inputs = {"user_query": request.query, "process_status": "start"}
        result = agent_service.run_agent("super", inputs, session_id=request.session_id)
        
        # Serialize result for response (handling BaseMessage objects)
        serializable_result = {}
        # Ensure we capture logs from the agents
        relevant_logs = ["build_logs", "extract_logs", "answer_logs", "eval_logs", "augment_logs"]
        for k, v in result.items():
            if k in relevant_logs and isinstance(v, list):
                serializable_result[k] = []
                for m in v:
                    msg_dict = {"role": getattr(m, 'type', 'unknown'), "content": getattr(m, 'content', str(m))}
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        msg_dict["tool_calls"] = m.tool_calls
                    serializable_result[k].append(msg_dict)
            else:
                serializable_result[k] = v
        
        return serializable_result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
):
    async def event_generator():
        try:
            inputs = {"user_query": request.query, "process_status": "start"}
            async for event in agent_service.stream_agent("super", inputs, session_id=request.session_id):
                # Handle tuple events when subgraphs=True
                if isinstance(event, tuple):
                    event = event[1]
                
                # LangGraph update event serialization
                serializable_event = {}
                for node_name, state_update in event.items():
                    serializable_node_update = {}
                    for k, v in state_update.items():
                        if isinstance(v, list):
                            serializable_node_update[k] = []
                            for m in v:
                                if hasattr(m, 'content') or isinstance(m, str):
                                    msg_dict = {
                                        "role": getattr(m, 'type', 'unknown'), 
                                        "content": getattr(m, 'content', str(m))
                                    }
                                    # Include tool calls if present
                                    if hasattr(m, 'tool_calls') and m.tool_calls:
                                        msg_dict["tool_calls"] = m.tool_calls
                                    serializable_node_update[k].append(msg_dict)
                        else:
                            serializable_node_update[k] = v
                    serializable_event[node_name] = serializable_node_update
                
                yield f"data: {json.dumps(serializable_event, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_msg = {"error": str(e)}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/knowledge", response_model=KnowledgeResponse)
async def add_knowledge(
    request: AddKnowledgeRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    try:
        result = agent_service.add_knowledge(
            documents=request.documents, metadatas=request.metadatas
        )
        return KnowledgeResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Adding knowledge failed: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse)
async def get_knowledge_stats(agent_service: AgentService = Depends(get_agent_service)):
    try:
        stats = agent_service.get_knowledge_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.delete("/knowledge/{doc_id}")
async def delete_knowledge(
    doc_id: str, agent_service: AgentService = Depends(get_agent_service)
):
    try:
        agent_service.vector_service.delete_document(doc_id)
        return {"status": "success", "message": f"Document {doc_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Agent service is running"}


@router.post("/{name}")
async def run_individual_agent(
    name: str,
    request: AgentRunRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    try:
        result = agent_service.run_agent(name, request.inputs, session_id=request.session_id)
        
        # Simple serialization
        serializable_result = {}
        for k, v in result.items():
            if isinstance(v, list):
                serializable_result[k] = [
                    {"role": getattr(m, 'type', 'unknown'), "content": getattr(m, 'content', str(m))}
                    for m in v
                ]
            else:
                serializable_result[k] = v
                
        return serializable_result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Agent '{name}' execution failed: {str(e)}"
        )
