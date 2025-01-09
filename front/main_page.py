import streamlit as st
from back.llm_service import get_huggingface_response
from back.chat_storage import save_chat_history

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
    "Breadth-First Search(BFS)": "너비 우선 탐색(BFS) 관련 시스템 프롬프트",
    "Sort Algorithm": "정렬 알고리즘 관련 시스템 프롬프트",
    "Greedy Algorithm": "그리디 알고리즘 관련 시스템 프롬프트",
    "Dynamic Programming(DP)": "DP 관련 시스템 프롬프트",
    "최단 경로 알고리즘": "최단 경로 알고리즘 관련 시스템 프롬프트"
}

def render_main_page():
    """중앙 메인 페이지 구현"""
    st.header(f"Conversation {st.session_state['current_page']}")

    # 안내문구 표시 (처음 한 번만)
    # if not st.session_state.greetings:
    #     with st.chat_message("assistant"):
    #         intro = "안녕하세요! 알고리즘 대화 인터페이스에 오신 것을 환영합니다. 아래 버튼을 눌러 원하는 알고리즘의 시스템 프롬프트를 선택하세요!"
    #         st.markdown(intro)
    #         st.session_state.messages.append({"role": "assistant", "content": intro})  # 대화 기록에 추가
    #     st.session_state.greetings = True  # 상태 업데이트

    # 상태 관리: 버튼이 눌리지 않았을 때
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None

    # 버튼이 눌리지 않았을 때 알고리즘 버튼 출력
    if st.session_state.button_pressed is None:
        # st.subheader("Choose an Algorithm")
        cols = st.columns(3)  # 3열 레이아웃
        for idx, (algo, prompt) in enumerate(ALGORITHM_PROMPTS.items()):
            with cols[idx % 3]:
                if st.button(algo):
                    st.session_state.button_pressed = algo  # 선택된 버튼을 상태로 저장
                    st.session_state.system_prompt = prompt  # 해당 시스템 프롬프트 저장
                    st.rerun()  # 페이지를 리프레시하여 새로운 버튼 상태 반영

    # 버튼이 눌린 후 선택된 알고리즘의 프롬프트 표시
    else:
        st.sidebar.title("System Prompt")
        st.sidebar.info(st.session_state.system_prompt)  # 사이드바에 시스템 프롬프트 표시

        # 선택된 알고리즘의 상태에서 "다시 선택" 버튼을 만들어 상태 초기화
        col = st.columns(1)[0]  # 중앙 정렬을 위해 1열 사용
        with col:
            if st.button(st.session_state.button_pressed, key="selected_button"):
                st.session_state.button_pressed = None  # 상태 초기화
                st.session_state.system_prompt = None

        # "다시 선택" 버튼을 눌러 선택을 초기화하는 기능
        if st.button("다시 선택"):
            st.session_state.button_pressed = None  # 상태 초기화
            st.session_state.system_prompt = None
            st.rerun()  # 리프레시하여 다시 처음 상태로 돌아가기
    # 대화 메시지 출력
    # st.subheader("Conversation")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 프롬프트 입력 창 (st.chat_input() 사용)
    user_input = st.chat_input("Your prompt:")
    if user_input:  # 사용자가 입력을 하면
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
    