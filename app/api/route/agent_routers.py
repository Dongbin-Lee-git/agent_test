from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.core.logger import logger
from app.models import (
    AddKnowledgeRequest,
    KnowledgeResponse,
    StatsResponse,
    ChatRequest,
    ChatResponse,
    TokenStreamEvent,
    LogStreamEvent,
    ErrorStreamEvent
)
from app.exceptions import AgentException, KnowledgeBaseException
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
        result = agent_service.run_agent(inputs, session_id=request.session_id)

        # Serialize result for response
        serializable_result = {
            "answer": ""
        }

        # answer_logs contains full history [Human, AI, Human, AI, ...]
        answer_logs = result.get("answer_logs", [])
        if answer_logs:
            # Last message is usually the current AI response
            last_msg = answer_logs[-1]
            if getattr(last_msg, 'type', '') == 'ai':
                serializable_result["answer"] = last_msg.content

        # Add other basic metadata if needed for ChatResponse
        for k in ["user_query", "process_status", "loop_count"]:
            if k in result:
                serializable_result[k] = result[k]

        return serializable_result
    except (AgentException, KnowledgeBaseException) as e:
        raise e
    except Exception as e:
        raise AgentException(f"Chat processing failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
        request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
):
    async def event_generator():
        try:
            inputs = {"user_query": request.query, "process_status": "start"}
            current_node = ""
            async for event in agent_service.stream_agent(inputs, session_id=request.session_id):
                kind = event.get("event")
                name = event.get("name", "")

                # 1. 노드 시작/종료 이벤트 처리 (진행 상태 표시용 및 current_node 추적)
                if kind == "on_chain_start":
                    # workflow 관련 노드나 super_graph 진입 시 current_node 업데이트
                    if name and ("workflow" in name or name == "super_graph"):
                        current_node = name

                    # Log 이벤트 전송 (주요 단계만)
                    if name == "info_extract_agent_workflow":
                        event_data = LogStreamEvent(log="내부 지식 검색 중...")
                        yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"
                    elif name == "knowledge_augment_workflow":
                        event_data = LogStreamEvent(log="외부 지식 검색 중 (Google Search)...")
                        yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"
                    elif name == "answer_gen_agent_workflow":
                        event_data = LogStreamEvent(log="답변 생성 중...")
                        yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"

                # 2. LLM 토큰 스트리밍 이벤트 처리
                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # 정보 추출이나 지식 확장 단계가 아니라면 답변 생성으로 간주하고 스트리밍
                        # 이렇게 하면 정확한 노드 이름을 모르더라도 답변 단계의 스트림을 놓치지 않음
                        if current_node not in ["info_extract_agent_workflow", "knowledge_augment_workflow"]:
                            event_data = TokenStreamEvent(answer=content)
                            yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"

                # 3. Tool 실행 로그
                elif kind == "on_tool_start":
                    tool_name = event.get("name")
                    if tool_name == "search_medical_qa":
                        event_data = LogStreamEvent(log="내부 DB 검색 실행...")
                        yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"
                    elif tool_name == "google_search":
                        event_data = LogStreamEvent(log="Google 검색 실행...")
                        yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    pass

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            event_data = ErrorStreamEvent(error=str(e))
            yield f"data: {event_data.model_dump_json(ensure_ascii=False)}\n\n"

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
    except (AgentException, KnowledgeBaseException) as e:
        raise e
    except Exception as e:
        raise KnowledgeBaseException(f"Adding knowledge failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_knowledge_stats(agent_service: AgentService = Depends(get_agent_service)):
    try:
        stats = agent_service.get_knowledge_stats()
        return StatsResponse(**stats)
    except (AgentException, KnowledgeBaseException) as e:
        raise e
    except Exception as e:
        raise KnowledgeBaseException(f"Failed to get stats: {str(e)}")


@router.delete("/knowledge/{doc_id}")
async def delete_knowledge(
        doc_id: str, agent_service: AgentService = Depends(get_agent_service)
):
    try:
        agent_service.vector_service.delete_document(doc_id)
        return {"status": "success", "message": f"Document {doc_id} deleted"}
    except (AgentException, KnowledgeBaseException) as e:
        raise e
    except Exception as e:
        raise KnowledgeBaseException(f"Deletion failed: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Agent service is running"}

