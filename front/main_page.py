import streamlit as st
from backend.llm_service import get_huggingface_response
from backend.chat_storage import save_chat_history

# 알고리즘별 시스템 프롬프트
ALGORITHM_PROMPTS = {
    "Depth-First Search(DFS)": """
            당신은 알고리즘 구현 전문가입니다. 
            사용자는 깊이 우선 탐색(DFS, Depth-First Search) 알고리즘을 구현하고자 합니다. 
            다음 규칙을 따르세요:

            1. DFS 알고리즘은 Python으로 작성합니다.
            2. DFS 알고리즘은 인접 리스트를 사용한 그래프 표현을 기반으로 동작합니다.
            3. 주어진 시작 노드에서 탐색을 시작하며, 방문한 노드를 순서대로 반환하는 함수를 작성하세요.
            4. 함수 이름은 `dfs_traversal`로 하고, 다음과 같은 매개변수를 받습니다:
            - `graph`: 인접 리스트 형태의 그래프 (딕셔너리)
            - `start`: 탐색을 시작할 노드
            5. 함수는 방문 순서대로 노드가 저장된 리스트를 반환해야 합니다.
            6. 입력 그래프는 연결 그래프 또는 비연결 그래프일 수 있습니다.
            7. 코드의 주요 단계에 대해 간단한 주석을 추가합니다.

            추가적으로, 다음 사항을 고려하세요:
            - 방문된 노드는 중복으로 처리되지 않도록 관리합니다.
            - 입력 그래프가 비어 있는 경우, 빈 리스트를 반환합니다.
            - 재귀와 스택을 활용한 두 가지 구현 방식 중 하나를 선택할 수 있습니다.
            """,
    "BFS": "너비 우선 탐색(BFS) 관련 시스템 프롬프트",
    "Sort": "정렬 알고리즘 관련 시스템 프롬프트",
    "Greedy": "그리디 알고리즘 관련 시스템 프롬프트",
    "Binary": "이진 탐색 관련 시스템 프롬프트",
    "최단 경로": "최단 경로 알고리즘 관련 시스템 프롬프트"
}

def render_main_page():
    """중앙 메인 페이지 구현"""
    # 안내문구 표시 (처음 한 번만)
    # if not st.session_state.greetings:
    #     with st.chat_message("assistant"):
    #         intro = "안녕하세요! 알고리즘 대화 인터페이스에 오신 것을 환영합니다. 아래 버튼을 눌러 원하는 알고리즘의 시스템 프롬프트를 선택하세요!"
    #         st.markdown(intro)
    #         st.session_state.messages.append({"role": "assistant", "content": intro})  # 대화 기록에 추가
    #     st.session_state.greetings = True  # 상태 업데이트
    st.title(f"Conversation {st.session_state['current_page']}")

    

    # 알고리즘 버튼
    st.subheader("Choose an Algorithm")
    cols = st.columns(3)  # 3열 레이아웃
    for idx, (algo, prompt) in enumerate(ALGORITHM_PROMPTS.items()):
        with cols[idx % 3]:
            if st.button(algo):
                st.session_state["current_prompt"] = prompt  # 선택된 프롬프트 저장
                st.rerun()

    # 대화 메시지 출력
    st.subheader("Conversation")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 프롬프트 입력 창
    user_input = st.text_input("Your prompt:", key="user_input")
    if st.button("Submit"):
        if user_input.strip():
            # LLM 응답 생성
            response = get_huggingface_response(st.session_state["model"], user_input)

            # 메시지 기록 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # 대화 기록 저장
            chat_history = st.session_state["chat_history"]
            current_page = st.session_state["current_page"] - 1

            # 기존 페이지 업데이트
            if current_page < len(chat_history):
                chat_history[current_page] = {"messages": st.session_state["messages"]}
            else:
                chat_history.append({"messages": st.session_state["messages"]})

            save_chat_history(chat_history)

            # UI 업데이트
            st.rerun()
