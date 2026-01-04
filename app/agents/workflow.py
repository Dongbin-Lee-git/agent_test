from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from app.agents.state import MainState
from app.core.logger import log_agent_step
from app.agents.utils import clean_and_parse_json


# Functions to call agents via services
def call_info_extractor(state: MainState, config: RunnableConfig):
    log_agent_step("Workflow", "Step 1: MedicalInfoExtractor 시작 (RAG)")
    print(f"\n[Workflow] Step 1: MedicalInfoExtractor 시작 (Query: {state['user_query']})")

    # Get service from config
    info_extractor_service = config["configurable"].get("info_extractor_service")

    # loop_count 초기화 및 증가
    current_count = state.get("loop_count", 0) + 1

    # InfoExtractor 실행
    result = info_extractor_service.run(
        state["user_query"],
        state.get("augment_logs", []),
        config=config,
        history=state.get("answer_logs", [])
    )

    # loop_count 업데이트 포함
    result["loop_count"] = current_count

    # history 업데이트: 새로 추가된 extract_logs를 로그에 반영
    if "extract_logs" in result:
        last_msg = result["extract_logs"][-1].content
        parsed = clean_and_parse_json(last_msg)
        status = parsed.get("status") if parsed else "unknown"
        log_agent_step("Workflow", f"Step 1 완료 (반복: {current_count})", {"status": status})
        print(f"[Workflow] Step 1 완료. Status: {status}, Iteration: {current_count}")

    return result


def call_knowledge_augmentor(state: MainState, config: RunnableConfig):
    log_agent_step("Workflow", "Step 2: MedicalKnowledgeAugmentor 시작 (Google Search)")
    print(f"\n[Workflow] Step 2: MedicalKnowledgeAugmentor 시작")

    # Get service from config
    knowledge_augmentor_service = config["configurable"].get("knowledge_augmentor_service")

    result = knowledge_augmentor_service.run(
        state["user_query"],
        config=config,
        history=state.get("answer_logs", [])
    )

    log_agent_step("Workflow", "Step 2 완료")
    print(f"[Workflow] Step 2 완료. 지식 보강됨.")
    return result


def call_answer_gen(state: MainState, config: RunnableConfig):
    log_agent_step("Workflow", "Step 3: MedicalConsultant 시작")
    print(f"\n[Workflow] Step 3: MedicalConsultant (AnswerGen) 시작")

    # Get service from config
    answer_gen_service = config["configurable"].get("answer_gen_service")

    result = answer_gen_service.run(
        state["user_query"],
        state.get("extract_logs", []),
        config=config,
        history=state.get("answer_logs", [])
    )

    log_agent_step("Workflow", "Step 3 완료")
    if "answer_logs" in result:
        print(f"[Workflow] Step 3 완료. 답변 생성됨.")
    return result


def check_extract_status(state: MainState):
    if not state.get("extract_logs"): return "augment"
    last_msg = state["extract_logs"][-1].content
    parsed = clean_and_parse_json(last_msg)

    status = parsed.get("status") if parsed else "unknown"
    loop_count = state.get("loop_count", 1)

    # 1. 도메인을 벗어나는 경우 -> 즉시 답변 생성으로 이동 (안내 메시지 목적)
    if status == "out_of_domain":
        log_agent_step("Workflow", "도메인 외 질문 판단 -> 답변 생성 이동 (안내 메시지)")
        return "continue"

    # 2. "success"이면 정보를 충분히 찾은 것이므로 답변 생성으로 이동
    if status == "success":
        return "continue"

    # 3. 반복 횟수 체크 (최대 2회)
    if loop_count >= 2:
        log_agent_step("Workflow", f"최대 반복 횟수({loop_count}) 도달 -> 답변 생성 이동", {"reason": "Iteration limit reached"})
        return "continue"

    # 4. "insufficient"이거나 파싱 실패 시 구글 검색(augment)으로 이동
    log_agent_step("Workflow", "내부 지식 부족 판단 -> Google 검색 이동", {
        "reason": parsed.get("reason") if parsed else "parse error",
        "iteration": loop_count
    })
    return "augment"


def router_node(state: MainState):
    return "medical"


super_workflow = StateGraph(MainState)
super_workflow.add_node("info_extract_agent_workflow", call_info_extractor)
super_workflow.add_node("knowledge_augment_workflow", call_knowledge_augmentor)
super_workflow.add_node("answer_gen_agent_workflow", call_answer_gen)

super_workflow.set_conditional_entry_point(
    router_node,
    {
        "medical": "info_extract_agent_workflow"
    }
)

super_workflow.add_conditional_edges(
    "info_extract_agent_workflow",
    check_extract_status,
    {
        "continue": "answer_gen_agent_workflow",
        "augment": "knowledge_augment_workflow"
    }
)
# Augment 이후 다시 추출을 시도
super_workflow.add_edge("knowledge_augment_workflow", "info_extract_agent_workflow")
super_workflow.add_edge("answer_gen_agent_workflow", END)

# 메모리 기반 체크포인터 추가 (대화 기록 보존용)
memory = MemorySaver()
super_graph = super_workflow.compile(checkpointer=memory)
