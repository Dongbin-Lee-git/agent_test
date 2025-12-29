import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from app.agents.subgraphs.evaluator import evaluate_graph

class EvaluatorService:
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

    def run(self, user_query: str, answer_logs: List[BaseMessage], extract_logs: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        final_answer = answer_logs[-1].content if answer_logs else ""
        
        context_text = ""
        if extract_logs:
            parsed = self._clean_and_parse_json(extract_logs[-1].content)
            context_text = parsed.get("medical_context", "") if parsed else ""

        input_prompt = f"[User Query]:\n{user_query}\n\n[Retrieved Context]:\n{context_text}\n\n[Generated Answer]:\n{final_answer}\n\nEvaluate now."
        sub_result = evaluate_graph.invoke({"messages": [HumanMessage(content=input_prompt)]})
        
        # Filter messages
        new_messages = [msg for msg in sub_result["messages"] if not isinstance(msg, HumanMessage) and not isinstance(msg, SystemMessage)]
        
        return {
            "eval_logs": new_messages,
            "process_status": "audit_complete"
        }
