import streamlit as st
import json
import httpx
import uuid
import os

# FastAPI backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

st.set_page_config(
    page_title="의료 QA 에이전트",
    page_icon="🏥",
    layout="wide"
)

# Initialize Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🏥 의료 QA 에이전트 시스템")
st.markdown("""
이 시스템은 Upstage Solar LLM과 LangGraph를 사용하여 구축된 의료 전문 질의응답 시스템입니다.
질문을 입력하면 에이전트가 지식 베이스를 검색하고 답변을 생성합니다.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def response_generator(prompt, session_id):
    """
    Connects to the backend and yields chunks of text for st.write_stream.
    """
    try:
        with httpx.stream(
                "POST",
                f"{BACKEND_URL}/agent/chat/stream",
                json={
                    "query": prompt,
                    "session_id": session_id
                },
                timeout=None
        ) as response:
            if response.status_code != 200:
                yield f"오류가 발생했습니다 (상태 코드: {response.status_code})"
                return

            # [수정] st.status 객체 생성
            status = st.status("에이전트가 분석 중입니다...", expanded=True)

            is_answering = False

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                        if "error" in event:
                            yield f"\n\n**에러 발생**: {event['error']}"
                            break

                        # 1. 로그 처리 (복구)
                        # Spinner 내부(status)에 주요 단계를 표시합니다.
                        if "log" in event:
                            status.write(event['log'])
                            continue

                        # 2. 중간 생각(Thought) 처리 로직 제거
                        # 사용자 요청에 따라 백엔드에서 보내지 않으므로 처리 로직도 삭제함

                        # 3. 답변(Answer) 처리
                        # Spinner 외부(메인 채팅창)에 작성되어야 하므로 yield를 사용합니다.
                        if "answer" in event and event["answer"]:
                            if not is_answering:
                                # 답변 시작 시 Spinner 상태 업데이트 (접기)
                                status.update(label="분석 완료", state="complete", expanded=False)
                                is_answering = True

                            # 여기서 yield하면 with status: 블록 밖이므로
                            # st.write_stream이 호출된 위치(assistant message)에 바로 찍힙니다.
                            yield event["answer"]

                    except json.JSONDecodeError:
                        continue

            # 루프가 끝날 때까지 answer가 없었다면 status 강제 종료
            if not is_answering:
                status.update(label="작업 완료", state="complete", expanded=False)

    except Exception as e:
        yield f"연결 오류: {str(e)}"


# User Input handling
if prompt := st.chat_input("의료 관련 질문을 입력하세요."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response logic
    with st.chat_message("assistant"):
        # st.write_stream은 generator에서 yield되는 답변 부분만 화면에 실시간으로 그림
        full_response = st.write_stream(response_generator(prompt, st.session_state.session_id))

        # Save complete response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

# Footer information
st.markdown("---")
st.caption("Powered by Upstage Solar LLM & LangGraph")
