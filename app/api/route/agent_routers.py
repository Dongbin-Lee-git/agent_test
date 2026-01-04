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
from app.exceptions import BaseAppException
from app.deps import get_agent_service
from app.service.agent_service import AgentService
from app.core.seed import get_seed_status

router = APIRouter(prefix="/agent", tags=["agent"])

@router.get("/seed-status")
async def seed_status():
    return get_seed_status()



@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
):
    try:
        inputs = {"user_query": request.query, "process_status": "start"}
        result = agent_service.run_agent("super", inputs, session_id=request.session_id)
        
        # Serialize result for response
        serializable_result = {
            "history": [],
            "reasoning": {},
            "answer": ""
        }
        
        # answer_logs contains full history [Human, AI, Human, AI, ...]
        answer_logs = result.get("answer_logs", [])
        if answer_logs:
            # Last message is usually the current AI response
            last_msg = answer_logs[-1]
            if getattr(last_msg, 'type', '') == 'ai':
                serializable_result["answer"] = last_msg.content
                history_to_process = answer_logs[:-1]
            else:
                history_to_process = answer_logs
            
            for m in history_to_process:
                serializable_result["history"].append({
                    "role": getattr(m, 'type', 'unknown'),
                    "content": getattr(m, 'content', str(m))
                })

        # Reasoning logs (per-turn)
        reasoning_keys = ["extract_logs", "augment_logs", "build_logs"]
        for k in reasoning_keys:
            if k in result and isinstance(result[k], list):
                serializable_result["reasoning"][k] = []
                for m in result[k]:
                    msg_dict = {"role": getattr(m, 'type', 'unknown'), "content": getattr(m, 'content', str(m))}
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        msg_dict["tool_calls"] = m.tool_calls
                    serializable_result["reasoning"][k].append(msg_dict)
        
        # Add other metadata
        for k, v in result.items():
            if k not in (["answer_logs"] + reasoning_keys):
                serializable_result[k] = v
        
        return serializable_result
    except BaseAppException as e:
        raise e
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
                serializable_event = {
                    "history": [],
                    "reasoning": {},
                    "answer": ""
                }
                for node_name, state_update in event.items():
                    serializable_node_update = {}
                    for k, v in state_update.items():
                        if k == "answer_logs" and isinstance(v, list):
                            # In streaming, answer_logs might be the full list or just the new message
                            # depending on how it's yielded. LangGraph 'updates' mode usually gives the new part.
                            for m in v:
                                msg_dict = {
                                    "role": getattr(m, 'type', 'unknown'),
                                    "content": getattr(m, 'content', str(m))
                                }
                                if msg_dict["role"] == "ai":
                                    serializable_event["answer"] = msg_dict["content"]
                                else:
                                    serializable_event["history"].append(msg_dict)
                        
                        elif k in ["extract_logs", "augment_logs", "build_logs"] and isinstance(v, list):
                            # Ensure we don't duplicate logs in the UI state
                            serializable_event["reasoning"][k] = []
                            for m in v:
                                msg_dict = {
                                    "role": getattr(m, 'type', 'unknown'),
                                    "content": getattr(m, 'content', str(m)),
                                    "node": node_name # Add node name to logs
                                }
                                if hasattr(m, 'tool_calls') and m.tool_calls:
                                    msg_dict["tool_calls"] = m.tool_calls
                                
                                # Add summary for tool results as requested
                                if msg_dict["role"] == "tool":
                                    content = msg_dict["content"]
                                    if "Source 1:" in content: # search_medical_qa result
                                        sources = content.split("Source ")[1:]
                                        msg_dict["summary"] = {
                                            "count": len(sources),
                                            "snippets": [s.split("\n", 1)[1][:20].strip() if "\n" in s else s[:20].strip() for s in sources]
                                        }
                                    else: # google_search or other tool results
                                        msg_dict["summary"] = content[:20] + "..." if len(content) > 20 else content

                                serializable_event["reasoning"][k].append(msg_dict)
                        else:
                            # Other state updates
                            if isinstance(v, list):
                                serializable_node_update[k] = [
                                    {"role": getattr(m, 'type', 'unknown'), "content": getattr(m, 'content', str(m))}
                                    for m in v
                                ]
                            else:
                                serializable_node_update[k] = v
                    
                    if serializable_node_update:
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
    except BaseAppException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Agent '{name}' execution failed: {str(e)}"
        )
