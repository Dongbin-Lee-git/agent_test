import os
from typing import List, Dict, Any

from openai import OpenAI  # openai==1.52.2
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from app.service.vector_service import VectorService
from app.agents import (
    super_graph, 
    info_extract_graph, 
    knowledge_augment_graph,
    answer_gen_graph, 
    evaluate_graph
)

load_dotenv()


class AgentService:
    def __init__(self, vector_service: VectorService):
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
        self.vector_service = vector_service
        self.graphs = {
            "super": super_graph,
            "extractor": info_extract_graph,
            "augmentor": knowledge_augment_graph,
            "answerer": answer_gen_graph,
            "evaluator": evaluate_graph
        }


    def add_knowledge(
        self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        try:
            self.vector_service.add_documents(documents, metadatas)
            return {
                "status": "success",
                "message": f"Added {len(documents)} documents to knowledge base",
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to add documents: {str(e)}"}

    def get_knowledge_stats(self) -> Dict[str, Any]:
        return self.vector_service.get_collection_info()

    def run_agent(self, agent_name: str, inputs: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        graph = self.graphs.get(agent_name)
        if not graph:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        # Ensure state is fresh for each run if it's the main graph
        if agent_name == "super":
            full_inputs = {
                "answer_logs": [HumanMessage(content=inputs["user_query"])],
                "build_logs": [],
                "augment_logs": [],
                "extract_logs": [],
                "eval_logs": [],
                "loop_count": 0
            }
            full_inputs.update(inputs)
            inputs = full_inputs

        config = {"configurable": {"vector_service": self.vector_service}}
        if session_id:
            config["configurable"]["thread_id"] = session_id
            
        result = graph.invoke(inputs, config=config)
        return result

    async def stream_agent(self, agent_name: str, inputs: Dict[str, Any], session_id: str = None):
        graph = self.graphs.get(agent_name)
        if not graph:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        if agent_name == "super":
            full_inputs = {
                "answer_logs": [HumanMessage(content=inputs["user_query"])],
                "build_logs": [],
                "augment_logs": [],
                "extract_logs": [],
                "eval_logs": [],
                "loop_count": 0
            }
            full_inputs.update(inputs)
            inputs = full_inputs

        config = {"configurable": {"vector_service": self.vector_service}}
        if session_id:
            config["configurable"]["thread_id"] = session_id
        
        # graph.astream uses the async streaming interface of LangGraph
        # subgraphs=True allows capturing events from internal nodes of subgraphs
        async for event in graph.astream(inputs, config=config, stream_mode="updates", subgraphs=True):
            yield event

