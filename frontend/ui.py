import streamlit as st
import requests
import json
import pandas as pd
import httpx
import uuid

# FastAPI 백엔드 URL
BACKEND_URL = "http://localhost:1234"

st.set_page_config(
    page_title="의료 QA 에이전트",
    page_icon="🏥",
    layout="wide"
)

# 세션 ID 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("🏥 의료 QA 에이전트 시스템")
st.markdown("""
이 시스템은 Upstage Solar LLM과 LangGraph를 사용하여 구축된 의료 전문 질의응답 시스템입니다.
질문을 입력하면 에이전트가 지식 베이스를 검색하고 답변을 생성합니다.
""")

# 사이드바: 시스템 정보 및 통계
with st.sidebar:
    st.header("📊 시스템 상태")
    try:
        stats_response = requests.get(f"{BACKEND_URL}/agent/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            st.metric("저장된 문서 수", f"{stats.get('count', 0)}개")
            st.info(f"컬렉션 이름: {stats.get('name', 'N/A')}")
        else:
            st.warning("백엔드에서 통계를 가져올 수 없습니다.")
    except Exception as e:
        st.error(f"백엔드 연결 실패: {e}")

    st.markdown("---")
    st.markdown("### 설정")
    api_url = st.text_input("백엔드 API URL", value=BACKEND_URL)
    
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4()) # 새로운 세션 ID 생성
        st.rerun()

# 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "logs" in message:
            with st.expander("추론 로그 보기"):
                st.json(message["logs"])

# 사용자 입력
if prompt := st.chat_input("의료 관련 질문을 입력하세요."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 에이전트 답변 생성
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        log_placeholder = st.empty()
        
        with st.status("🤔 에이전트가 생각 중입니다...", expanded=True) as status:
            full_response_data = {
                "extract_logs": [],
                "augment_logs": [],
                "answer_logs": [],
                "eval_logs": []
            }
            
            try:
                # httpx를 사용하여 스트리밍 요청
                with httpx.stream(
                    "POST", 
                    f"{api_url}/agent/chat/stream", 
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=None
                ) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                data_str = line[len("data: "):]
                                if data_str == "[DONE]":
                                    status.update(label="✅ 답변 생성 완료", state="complete", expanded=False)
                                    break
                                
                                try:
                                    event = json.loads(data_str)
                                    if "error" in event:
                                        st.error(f"에러 발생: {event['error']}")
                                        break
                                    
                                    # 이벤트 처리 및 UI 업데이트
                                    for node_name, update in event.items():
                                        # 한글 노드 명칭 맵핑
                                        node_display_names = {
                                            "info_extract_agent_workflow": "🔍 지식 추출 프로세스",
                                            "info_extractor": "🔎 내부 지식 검색 중",
                                            "info_extract_tools": "🛠️ 검색 도구 실행",
                                            "info_verifier": "⚖️ 검색 결과 검증",
                                            "knowledge_augment_workflow": "🌐 외부 지식 보강 (Google)",
                                            "answer_gen_agent_workflow": "✍️ 답변 작성",
                                            "evaluate_agent_workflow": "⚖️ 답변 검증 및 평가"
                                        }
                                        display_name = node_display_names.get(node_name, node_name)
                                        
                                        # 툴 호출 정보 표시
                                        if "messages" in update:
                                            for msg in update["messages"]:
                                                if "tool_calls" in msg:
                                                    for tc in msg["tool_calls"]:
                                                        status.write(f"🛠️ **도구 호출**: `{tc['name']}` ({tc['args']})")
                                        
                                        # 노드별 상세 정보 추출
                                        detail_info = ""
                                        if node_name == "info_extract_agent_workflow" and "extract_logs" in update:
                                            last_log = update["extract_logs"][-1].get("content", "")
                                            if "out_of_domain" in last_log:
                                                detail_info = " (도메인 외 질문으로 판단됨)"
                                            elif "success" in last_log:
                                                detail_info = " (관련 정보 탐색 성공)"
                                            elif "insufficient" in last_log:
                                                detail_info = " (내부 정보 부족, 보강 필요)"
                                                
                                        elif node_name == "evaluate_agent_workflow" and "eval_logs" in update:
                                            last_log = update["eval_logs"][-1].get("content", "")
                                            if "final_score" in last_log:
                                                detail_info = " (평가 완료)"

                                        status.update(label=f"⏳ {display_name} 진행 중...")
                                        if detail_info:
                                            status.write(f"✅ **{display_name}** 완료{detail_info}")
                                        
                                        # 로그 업데이트
                                        for log_key in full_response_data.keys():
                                            if log_key in update:
                                                full_response_data[log_key].extend(update[log_key])
                                        
                                        # 실시간 답변 표시 (answer_placeholder는 status 외부)
                                        if "answer_logs" in update and update["answer_logs"]:
                                            answer = update["answer_logs"][-1].get("content", "")
                                            answer_placeholder.markdown(answer)
                                            
                                except json.JSONDecodeError:
                                    continue
                    else:
                        st.error(f"오류가 발생했습니다 (상태 코드: {response.status_code})")
                        status.update(label="❌ 오류 발생", state="error")
            except Exception as e:
                st.error(f"연결 오류: {str(e)}")
                status.update(label="❌ 연결 오류", state="error")

        # 최종 답변 정리 및 저장
        final_answer = ""
        if full_response_data["answer_logs"]:
            final_answer = full_response_data["answer_logs"][-1].get("content", "")
        
        if not final_answer:
            final_answer = "답변을 생성하지 못했습니다."
        
        answer_placeholder.markdown(final_answer)
        
        # 로그 표시
        logs_to_show = {k: v for k, v in full_response_data.items() if v}
        if logs_to_show:
            with log_placeholder.expander("추론 로그 보기"):
                st.json(logs_to_show)
        
        # 세션 상태에 저장
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer,
            "logs": logs_to_show
        })

# 하단 정보
st.markdown("---")
st.caption("Powered by Upstage Solar LLM & LangGraph")
