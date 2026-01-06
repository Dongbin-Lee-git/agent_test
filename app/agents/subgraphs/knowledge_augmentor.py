from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from app.agents.state import InfoBuildAgentState
from app.agents.tools import google_search, add_to_medical_qa, solar_chat
from app.agents.utils import get_current_time_str
from app.core.logger import log_agent_step

instruction_augment = """
You are the 'MedicalKnowledgeAugmentor'. Your goal is to search Google for medical information and add it to our knowledge base.

# Workflow
1. **Search Google**: Use `google_search(query)` to find relevant medical info.
2. **Add to DB**: Use `add_to_medical_qa(content, metadata)` to save the found info.
3. **Termination**: 
   - Once you have found and added SUFFICIENT information to answer the original query, you MUST stop.
   - Do NOT repeat the same search or add redundant information.
   - If you have added at least one or two high-quality pieces of information, that is usually enough.
4. **Final Answer**: Return strictly JSON: `{"status": "success", "info_added": "..."}`.
"""

augment_tools = [google_search, add_to_medical_qa]
llm_augment = solar_chat.bind_tools(augment_tools)


def augment_agent(state: InfoBuildAgentState):
    messages = state["messages"]

    # Check for recent tool results and log summary
    if messages and isinstance(messages[-1], ToolMessage):
        last_msg = messages[-1]
        content = last_msg.content
        # If it's from google_search (we don't have the tool name here easily, 
        # but google_search usually returns a string that doesn't look like our DB Source format)
        if "Source 1:" not in content and len(content) > 0:
            summary = content[:20] + "..." if len(content) > 20 else content
            log_agent_step("KnowledgeAugmentor", "Google 검색 결과 요약", {"summary": summary})

    if not messages or not isinstance(messages[0], SystemMessage):
        current_time = get_current_time_str()
        system_content = f"현재 시각: {current_time}\n\n{instruction_augment}"
        messages = [SystemMessage(content=system_content)] + messages

    # 도구 호출 횟수 체크
    tool_call_count = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
    if tool_call_count > 3:
        log_agent_step("KnowledgeAugmentor", "최대 도구 호출 횟수 도달 -> 강제 종료")
        return {"messages": [AIMessage(content='{"status": "success", "info_added": "Maximum tool calls reached"}')]}

    log_agent_step("KnowledgeAugmentor", "구글 검색 및 DB 추가 시작")
    response = llm_augment.invoke(messages)

    log_agent_step("KnowledgeAugmentor", "응답 수신", {"content": response.content, "tool_calls": response.tool_calls})
    return {"messages": [response]}


def should_continue(state: InfoBuildAgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        log_agent_step("KnowledgeAugmentor", "도구 사용", {"tools": [tc['name'] for tc in last_message.tool_calls]})
        return "tools"
    return END


workflow = StateGraph(InfoBuildAgentState)
workflow.add_node("augment_agent", augment_agent)
workflow.add_node("augment_tools", ToolNode(augment_tools))
workflow.set_entry_point("augment_agent")
workflow.add_conditional_edges("augment_agent", should_continue, {"tools": "augment_tools", END: END})
workflow.add_edge("augment_tools", "augment_agent")

knowledge_augment_graph = workflow.compile()
