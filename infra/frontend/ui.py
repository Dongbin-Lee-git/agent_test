import streamlit as st
import requests
import json
import httpx
import uuid

import os

# FastAPI 백엔드 URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:1234")

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
        if "reasoning" in message and message["reasoning"]:
            with st.expander("추론 로그 보기"):
                # 노드별 한글 명칭 맵핑 (UI용)
                node_names = {
                    "info_extract_agent_workflow": "🔍 내부 지식 검색 & 검증",
                    "knowledge_augment_workflow": "🌐 외부 지식 보강 (Google)",
                    "answer_gen_agent_workflow": "✍️ 답변 생성"
                }
                
                for log_type, logs in message["reasoning"].items():
                    if not logs: continue
                    
                    # 해당 로그의 노드 정보 확인
                    node_id = logs[0].get("node", "unknown")
                    display_node_name = node_names.get(node_id, node_id)
                    
                    st.write(f"### 📍 {display_node_name} ({node_id})")
                    st.json(logs)

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
                "history": [],
                "reasoning": {},
                "answer": ""
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
                                    
                                    # 히스토리 업데이트 (참고용)
                                    if "history" in event and event["history"]:
                                        full_response_data["history"].extend(event["history"])

                                    # 답변 업데이트 및 실시간 표시
                                    if "answer" in event and event["answer"]:
                                        full_response_data["answer"] += event["answer"]
                                        answer_placeholder.markdown(full_response_data["answer"])

                                    # 추론 로그 업데이트: 현재 턴의 로그만 유지하도록 개선
                                    if "reasoning" in event and event["reasoning"]:
                                        for k, v in event["reasoning"].items():
                                            if k not in full_response_data["reasoning"]:
                                                full_response_data["reasoning"][k] = []
                                            # v가 리스트인 경우만 처리
                                            if isinstance(v, list):
                                                for log_entry in v:
                                                    # 중복 체크
                                                    is_duplicate = False
                                                    for existing in full_response_data["reasoning"][k]:
                                                        if existing.get("content") == log_entry.get("content") and \
                                                           existing.get("role") == log_entry.get("role") and \
                                                           existing.get("tool_calls") == log_entry.get("tool_calls"):
                                                            is_duplicate = True
                                                            break
                                                    
                                                    if not is_duplicate:
                                                        full_response_data["reasoning"][k].append(log_entry)

                                    # 노드 상태 업데이트 (기존 로직 유지)
                                    for node_name, update in event.items():
                                        if node_name in ["history", "reasoning", "answer"]: continue
                                        
                                        # 한글 노드 명칭 맵핑
                                        node_display_names = {
                                            "info_extract_agent_workflow": "🔍 지식 추출 프로세스",
                                            "info_extractor": "🔎 내부 지식 검색 중",
                                            "info_extract_tools": "🛠️ 검색 도구 실행",
                                            "info_verifier": "⚖️ 검색 결과 검증",
                                            "knowledge_augment_workflow": "🌐 외부 지식 보강 (Google)",
                                            "answer_gen_agent_workflow": "✍️ 답변 작성"
                                        }
                                        display_name = node_display_names.get(node_name, node_name)
                                        
                                        # 툴 호출 정보 표시
                                        if isinstance(update, dict) and "messages" in update:
                                            for msg in update["messages"]:
                                                if "tool_calls" in msg:
                                                    for tc in msg["tool_calls"]:
                                                        status.write(f"🛠️ **도구 호출**: `{tc['name']}`")
                                        
                                        # 노드별 상세 정보 추출
                                        detail_info = ""
                                        if node_name == "info_extract_agent_workflow" and "extract_logs" in update:
                                            # (Note: Backwards compatibility for raw update format if needed)
                                            pass

                                        status.update(label=f"⏳ {display_name} 진행 중...")
                                            
                                except json.JSONDecodeError:
                                    continue
                    else:
                        st.error(f"오류가 발생했습니다 (상태 코드: {response.status_code})")
                        status.update(label="❌ 오류 발생", state="error")
            except Exception as e:
                st.error(f"연결 오류: {str(e)}")
                status.update(label="❌ 연결 오류", state="error")

        # 최종 답변 정리 및 저장
        final_answer = full_response_data["answer"]
        if not final_answer:
            final_answer = "답변을 생성하지 못했습니다."
        
        answer_placeholder.markdown(final_answer)
        
        # 로그 표시
        if full_response_data["reasoning"]:
            with log_placeholder.expander("추론 로그 보기"):
                # 노드별 한글 명칭 맵핑 (UI용)
                node_names = {
                    "info_extract_agent_workflow": "🔍 내부 지식 검색 & 검증",
                    "knowledge_augment_workflow": "🌐 외부 지식 보강 (Google)",
                    "answer_gen_agent_workflow": "✍️ 답변 생성"
                }
                
                for log_type, logs in full_response_data["reasoning"].items():
                    if not logs: continue
                    
                    # 해당 로그의 노드 정보 확인
                    node_id = logs[0].get("node", "unknown")
                    display_node_name = node_names.get(node_id, node_id)
                    
                    st.write(f"### 📍 {display_node_name} ({node_id})")
                    st.json(logs)
        
        # 세션 상태에 저장
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer,
            "reasoning": full_response_data["reasoning"]
        })

# 하단 정보
st.markdown("---")
st.caption("Powered by Upstage Solar LLM & LangGraph")
