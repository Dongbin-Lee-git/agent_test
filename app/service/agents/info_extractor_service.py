from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.agents.subgraphs.info_extractor import info_extract_graph


class InfoExtractorService:
    def run(self, user_query: str, build_logs: List[BaseMessage] = None, config: RunnableConfig = None,
            history: List[BaseMessage] = None) -> Dict[str, Any]:
        handoff_msg = f"Original User Query: \"{user_query}\"\n\nPlease search the internal database first. Refer to the previous conversation history if it helps to understand the context of the user's query."
        if build_logs:
            last_build_msg = build_logs[-1]
            handoff_msg += f"\nPrevious context: {last_build_msg.content}"

        messages = []
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=handoff_msg))

        sub_result = info_extract_graph.invoke({"messages": messages}, config=config)

        # sub_result["messages"] contains the full conversation history + new messages.
        # We only want the AI messages produced *after* our handoff message.
        history_len = len(messages)
        new_messages = [
            msg for msg in sub_result["messages"][history_len:]
            if isinstance(msg, AIMessage)
        ]

        return {
            "extract_logs": new_messages,
            "process_status": "success"
        }
