import os
from typing import List, Dict, Any

from openai import OpenAI  # openai==1.52.2
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from app.service.vector_service import VectorService
from app.service.agents.info_extractor_service import InfoExtractorService
from app.service.agents.knowledge_augmentor_service import KnowledgeAugmentorService
from app.service.agents.answer_gen_service import AnswerGenService
from app.agents import super_graph

load_dotenv()


class AgentService:
    def __init__(
            self,
            vector_service: VectorService,
            info_extractor_service: InfoExtractorService,
            knowledge_augmentor_service: KnowledgeAugmentorService,
            answer_gen_service: AnswerGenService
    ):
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
        self.vector_service = vector_service

        # Injected sub-services
        self.info_extractor_service = info_extractor_service
        self.knowledge_augmentor_service = knowledge_augmentor_service
        self.answer_gen_service = answer_gen_service

    def add_knowledge(
            self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        try:
            # documents와 metadatas를 사용하여 MedicalQA 리스트를 생성하는 방식으로 확장 가능하지만
            # 현재는 단순 전달 구조 유지. 필요시 엔티티 변환 로직 추가 가능.
            self.vector_service.add_documents(documents, metadatas)
            return {
                "status": "success",
                "message": f"Added {len(documents)} documents to knowledge base",
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to add documents: {str(e)}"}

    def get_knowledge_stats(self) -> Dict[str, Any]:
        return self.vector_service.get_collection_info()

    def run_agent(self, inputs: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        # Ensure state is fresh for each run
        # answer_logs will be handled by LangGraph's checkpointer if session_id is provided.
        # We only provide the new user query.
        full_inputs = {
            "user_query": inputs["user_query"],
            "answer_logs": [HumanMessage(content=inputs["user_query"])],
            "build_logs": [],
            "augment_logs": [],
            "extract_logs": [],
            "loop_count": 0
        }
        inputs = full_inputs

        config = {
            "configurable": {
                "vector_service": self.vector_service,
                "info_extractor_service": self.info_extractor_service,
                "knowledge_augmentor_service": self.knowledge_augmentor_service,
                "answer_gen_service": self.answer_gen_service,
            }
        }
        if session_id:
            config["configurable"]["thread_id"] = session_id

        result = super_graph.invoke(inputs, config=config)
        return result

    async def stream_agent(self, inputs: Dict[str, Any], session_id: str = None):
        # answer_logs will be handled by LangGraph's checkpointer if session_id is provided.
        # We only provide the new user query.
        full_inputs = {
            "user_query": inputs["user_query"],
            "answer_logs": [HumanMessage(content=inputs["user_query"])],
            "build_logs": [],
            "augment_logs": [],
            "extract_logs": [],
            "loop_count": 0
        }
        inputs = full_inputs

        config = {
            "configurable": {
                "vector_service": self.vector_service,
                "info_extractor_service": self.info_extractor_service,
                "knowledge_augmentor_service": self.knowledge_augmentor_service,
                "answer_gen_service": self.answer_gen_service,
            }
        }
        if session_id:
            config["configurable"]["thread_id"] = session_id

        # graph.astream_events uses the async event streaming interface of LangGraph
        # This allows capturing token-level events from LLMs within the graph
        async for event in super_graph.astream_events(inputs, config=config, version="v2"):
            yield event
