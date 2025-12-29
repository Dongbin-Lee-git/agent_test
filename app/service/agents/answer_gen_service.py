import json
import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from app.agents.subgraphs.answer_gen import answer_gen_graph

class AnswerGenService:
    def _clean_and_parse_json(self, text: str):
        try:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match: text = match.group(1)
            else:
                match = re.search(r"(\{.*\})", text, re.DOTALL)
                if match: text = match.group(1)
            return json.loads(text)
        except:
            return None

    def run(self, user_query: str, extract_logs: List[BaseMessage], config: RunnableConfig = None, history: List[BaseMessage] = None) -> Dict[str, Any]:
        if not extract_logs:
            return {"answer_logs": [AIMessage(content="Failed to extract info.")], "process_status": "fail"}
        
        last_extract_msg = extract_logs[-1]
        parsed_result = self._clean_and_parse_json(last_extract_msg.content)
        
        status = parsed_result.get("status") if parsed_result else "unknown"
        medical_context = parsed_result.get("medical_context", "") if parsed_result else ""
        
        if status == "out_of_domain":
            # 도메인을 벗어난 경우의 전용 프롬프트
            prompt = f"""User Query: "{user_query}"\n\nTask: You are a medical AI assistant. The user's query is unrelated to medical or health topics. Explain that you are specialized in medical advice and cannot answer this specific non-medical query, but offer to help with any health-related questions. Keep it polite and professional in Korean. You should also refer to previous conversation history if it's helpful to maintain context (e.g. if the user previously introduced themselves)."""
        else:
            prompt = f"""User Query: "{user_query}"\n\nRetrieved Medical Context:\n===\n{medical_context}\n===\nTask: Provide a medical consultation based on the context. Refer to the previous conversation history if needed to maintain continuity."""
        
        messages = []
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=prompt))
        
        sub_result = answer_gen_graph.invoke({"messages": messages}, config=config)
        
        # Filter messages
        new_messages = [msg for msg in sub_result["messages"] if not isinstance(msg, HumanMessage) and not isinstance(msg, SystemMessage)]
        
        return {
            "answer_logs": new_messages,
            "process_status": "success"
        }
