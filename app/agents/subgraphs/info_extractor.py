from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, ToolMessage
from app.agents.state import InfoExtractAgentState
from app.agents.tools import search_medical_qa, solar_chat
from app.agents.utils import clean_and_parse_json
from app.core.logger import log_agent_step

instruction_info_extract = """
You are the 'MedicalInfoExtractor'. Your goal is to gather medical context for the user's query from our internal Korean-language medical knowledge base.

# Workflow
1. **Internal Search**: Use `search_medical_qa(query)` to find relevant information from our knowledge base.
2. **Review**: Look at the results from the tool and decide if you need more search or if you have enough raw information.
3. **Finish**: When you have gathered enough raw information, stop and let the 'MedicalInfoVerifier' evaluate it.
"""

instruction_info_verify = """
You are the 'MedicalInfoVerifier'. Your goal is to evaluate the medical information gathered by the extractor and determine if it's sufficient to answer the user's query.

# Input
- User's original query.
- Retrieved documents from the internal database.

# Evaluation Criteria
1. **Domain Check**: Is the user's query related to medical or health topics?
   - If NOT medical-related (e.g., space, cooking, sports), set "status" to "out_of_domain".
2. **Sufficiency Check**: If it IS a medical query, does the information directly address it?
   - If sufficient, set "status" to "success".
   - If insufficient or missing, set "status" to "insufficient".

# Output Format
- Return strictly JSON format: `{"status": "success" | "insufficient" | "out_of_domain", "medical_context": "...", "key_points": ["point1", "point2", ...]}`
- Do NOT output anything else.
"""

info_extract_tools = [search_medical_qa]
llm_info_extract = solar_chat.bind_tools(info_extract_tools)


def info_extractor(state: InfoExtractAgentState):
    messages = state["messages"]
    
    # Check for recent tool results and log summary
    if messages and isinstance(messages[-1], ToolMessage):
        last_msg = messages[-1]
        content = last_msg.content
        if "Source 1:" in content:
            sources = content.split("Source ")[1:]
            summary = {
                "count": len(sources),
                "snippets": [s.split("\n", 1)[1][:20].strip() if "\n" in s else s[:20].strip() for s in sources]
            }
            log_agent_step("MedicalInfoExtractor", "VectorDB 검색 결과 요약", summary)

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=instruction_info_extract)] + messages
    
    log_agent_step("MedicalInfoExtractor", "검색 에이전트 시작", {"input_messages_count": len(messages)})
    response = llm_info_extract.invoke(messages)
    
    # 여러 개의 툴 호출이 들어올 경우 첫 번째만 수행하도록 제한
    if response.tool_calls and len(response.tool_calls) > 1:
        print(f"\n[MedicalInfoExtractor] Multiple tool calls detected. Keeping only the first one: {response.tool_calls[0]['name']}")
        response.tool_calls = response.tool_calls[:1]

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n[MedicalInfoExtractor] Tool Call: {tool_call['name']}({tool_call['args']})")
        log_agent_step("MedicalInfoExtractor", "도구 호출 응답 수신", {"tool_calls": response.tool_calls})
    else:
        log_agent_step("MedicalInfoExtractor", "검색 및 추출 완료", {"content": response.content[:100] + "..." if response.content else "None"})
    
    return {"messages": [response]}

def info_verifier(state: InfoExtractAgentState):
    messages = state["messages"]
    # 검증을 위한 시스템 메시지 추가
    verify_messages = [SystemMessage(content=instruction_info_verify)] + messages
    
    log_agent_step("MedicalInfoVerifier", "검증 시작")
    response = solar_chat.invoke(verify_messages)
    
    # 결과 파싱 및 로깅
    parsed = clean_and_parse_json(response.content)
    if parsed:
        log_agent_step("MedicalInfoVerifier", "검증 완료", {
            "status": parsed.get("status")
        })
    else:
        log_agent_step("MedicalInfoVerifier", "검증 완료 (파싱 실패)", {"content": response.content})
        
    return {"messages": [response]}

def no_results_handler(state: InfoExtractAgentState):
    """검색 결과가 없을 때 verifier를 건너뛰지 않고, 대신 도메인 판단만 수행"""
    log_agent_step("MedicalInfoExtractor", "내부 검색 결과 없음 -> 도메인 확인 시작")
    
    # 도메인 판단을 위한 전용 프롬프트
    domain_check_prompt = f"""
    Evaluate if the following user query is related to medical or health topics.
    Query: {state['messages'][1].content if len(state['messages']) > 1 else ""}
    
    Output strictly in JSON:
    {"status": "out_of_domain" | "insufficient"}
    
    If it IS medical but no info was found, use "insufficient".
    If it is NOT medical, use "out_of_domain".
    """
    
    response = solar_chat.invoke(domain_check_prompt)
    
    parsed = clean_and_parse_json(response.content)
    
    if parsed:
        log_agent_step("MedicalInfoExtractor", "도구 결과 없음 - 도메인 확인 결과", {
            "status": parsed.get("status")
        })
    
    return {"messages": [response]}

def should_continue(state: InfoExtractAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Tool call limit
    tool_call_count = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
    if tool_call_count > 2:
        log_agent_step("MedicalInfoExtractor", "도구 호출 횟수 초과로 강제 종료")
        return "verify"

    if last_message.tool_calls:
        log_agent_step("MedicalInfoExtractor", "도구 호출 결정", {"tools": [tc['name'] for tc in last_message.tool_calls]})
        return "tools"
    
    # 검색 도구의 결과를 확인하여 결과가 0건인지 체크
    # messages를 역순으로 확인하여 가장 최근의 ToolMessage(검색 결과)를 찾음
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # search_medical_qa의 결과가 비어있거나 "Search Error"인 경우
            if not msg.content or msg.content.strip() == "" or msg.content.startswith("Search Error"):
                return "no_results"
            break # 가장 최근 도구 응답만 확인
    
    return "verify"

workflow = StateGraph(InfoExtractAgentState)
workflow.add_node("info_extractor", info_extractor)
workflow.add_node("info_extract_tools", ToolNode(info_extract_tools))
workflow.add_node("info_verifier", info_verifier)
workflow.add_node("no_results_handler", no_results_handler)

workflow.set_entry_point("info_extractor")

workflow.add_conditional_edges(
    "info_extractor",
    should_continue,
    {
        "tools": "info_extract_tools",
        "verify": "info_verifier",
        "no_results": "no_results_handler"
    }
)
workflow.add_edge("info_extract_tools", "info_extractor")
workflow.add_edge("info_verifier", END)
workflow.add_edge("no_results_handler", END)

info_extract_graph = workflow.compile()
