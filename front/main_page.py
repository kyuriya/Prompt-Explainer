import streamlit as st
from back.llm_service import get_huggingface_response
from back.chat_storage import save_chat_history
# from back.explainability import compute_lime_values
# from front.visualization import display_lime_visualization

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
    "Breadth-First Search(BFS)": """
            당신은 알고리즘 구현 전문가입니다. 
            사용자는 너비 우선 탐색(BFS, Breadth-First Search) 알고리즘을 구현하고자 합니다. 
            다음 규칙을 따르세요:

            1. BFS 알고리즘은 Python으로 작성합니다.
            2. BFS 알고리즘은 인접 리스트를 사용한 그래프 표현을 기반으로 동작합니다.
            3. 주어진 시작 노드에서 탐색을 시작하며, 방문한 노드를 순서대로 반환하는 함수를 작성하세요.
            4. 함수 이름은 `bfs_traversal`로 하고, 다음과 같은 매개변수를 받습니다:
            - `graph`: 인접 리스트 형태의 그래프 (딕셔너리)
            - `start`: 탐색을 시작할 노드
            5. 함수는 방문 순서대로 노드가 저장된 리스트를 반환해야 합니다.
            6. 입력 그래프는 연결 그래프 또는 비연결 그래프일 수 있습니다.
            7. 코드의 주요 단계에 대해 간단한 주석을 추가합니다.

            추가적으로, 다음 사항을 고려하세요:
            - 방문된 노드는 중복으로 처리되지 않도록 관리합니다.
            - 입력 그래프가 비어 있는 경우, 빈 리스트를 반환합니다.
            - 큐(queue)를 사용하여 탐색 과정을 구현합니다.
            """ ,
    "Sort Algorithm": """
            당신은 알고리즘 구현 전문가입니다. 
            사용자는 정렬 알고리즘을 구현하고자 합니다. 
            다음 규칙을 따르세요:

            1. 정렬 알고리즘은 Python으로 작성합니다.
            2. 사용자는 특정 정렬 알고리즘(예: 버블 정렬, 퀵 정렬, 병합 정렬 등)을 구현하려고 합니다.
            3. 구현하려는 정렬 알고리즘은 다음과 같은 규칙을 따릅니다:
            - 함수 이름은 `sort_algorithm`으로 하고, 정렬하려는 리스트를 매개변수로 받습니다.
            - 정렬된 리스트를 반환해야 합니다.
            4. 함수는 오름차순 정렬을 기본으로 합니다.
            5. 알고리즘에 따라 주요 단계를 간단한 주석으로 설명합니다.

            추가적으로, 다음 사항을 고려하세요:
            - 입력 리스트가 비어 있거나 원소가 하나뿐인 경우, 그대로 반환합니다.
            - 적절한 정렬 알고리즘을 선택하고 구현합니다 (예: `버블 정렬` -> 기본 반복문 사용).
            """,
    "Greedy Algorithm": """
            당신은 알고리즘 구현 전문가입니다. 
            사용자는 탐욕 알고리즘(Greedy Algorithm)을 구현하고자 합니다. 
            다음 규칙을 따르세요:

            1. 탐욕 알고리즘은 Python으로 작성합니다.
            2. 사용자가 원하는 문제 유형(예: 활동 선택 문제, 최소 스패닝 트리, 동전 거스름 문제)을 해결하는 코드를 작성합니다.
            3. 작성한 함수는 다음을 포함해야 합니다:
            - 함수 이름과 매개변수는 문제에 따라 유동적으로 설정합니다.
            - 탐욕 알고리즘이 선택하는 기준(예: 최대, 최소 등)을 명확히 설명합니다.
            4. 주요 단계에 대한 주석을 포함합니다.

            추가적으로, 다음 사항을 고려하세요:
            - 탐욕 알고리즘으로 최적 해를 보장할 수 있는 경우를 명확히 설명합니다.
            - 문제 입력값에 대한 예외 처리를 포함합니다.
            """,
    "Dynamic Programming(DP)": """
            당신은 알고리즘 구현 전문가입니다. 
            사용자는 동적 계획법(Dynamic Programming, DP)을 사용하여 문제를 해결하고자 합니다. 
            다음 규칙을 따르세요:

            1. 동적 계획법은 Python으로 작성합니다.
            2. 사용자가 원하는 문제 유형(예: 피보나치 수열, 배낭 문제, 최소 경로 등)을 해결하는 코드를 작성합니다.
            3. 작성한 함수는 다음을 포함해야 합니다:
            - 함수 이름과 매개변수는 문제에 따라 유동적으로 설정합니다.
            - 동적 계획법의 `메모이제이션` 또는 `타뷸레이션` 접근법 중 하나를 사용합니다.
            4. 주요 단계와 DP 테이블 구조를 설명하는 주석을 포함합니다.

            추가적으로, 다음 사항을 고려하세요:
            - 반복적(recurrence relation)인 풀이 과정을 명확히 구현합니다.
            - 최적화된 시간 복잡도를 고려합니다.
            - 기본 입력 데이터가 비어 있을 때의 예외 처리를 포함합니다.
            """,
    "최단 경로 알고리즘": """
            당신은 알고리즘 구현 전문가입니다. 
            사용자는 최단 경로 알고리즘을 구현하고자 합니다. 
            다음 규칙을 따르세요:

            1. 최단 경로 알고리즘은 Python으로 작성합니다.
            2. 구현하려는 알고리즘은 다익스트라, 플로이드-와샬, 또는 벨만-포드 알고리즘 중 하나입니다.
            3. 구현할 함수는 다음과 같은 규칙을 따릅니다:
            - 함수 이름은 `shortest_path`로 하고, 입력 그래프와 시작 노드를 매개변수로 받습니다.
            - 그래프는 인접 리스트 또는 가중치 행렬로 표현됩니다.
            - 각 노드에 대한 최단 거리 값을 반환합니다.
            4. 주요 단계에 대한 주석을 포함합니다.

            추가적으로, 다음 사항을 고려하세요:
            - 그래프가 비어 있을 경우, 적절한 예외 처리를 수행합니다.
            - 구현한 알고리즘의 시간 복잡도를 최적화합니다.
            """
}

def render_main_page():
    """중앙 메인 페이지 구현"""
    st.header(f"Conversation {st.session_state['current_page']}")

    # 안내문구 표시 (처음 한 번만)
    if not st.session_state.get("greetings", False):
        with st.chat_message("assistant"):
            intro = "안녕하세요! 알고리즘 대화 인터페이스에 오신 것을 환영합니다. 아래 버튼을 눌러 원하는 알고리즘의 시스템 프롬프트를 선택하세요!"
            st.markdown(intro)  # 사용자에게 보이는 안내문구
            st.session_state.messages.append({"role": "assistant", "content": intro})  # 대화 기록에 추가
        st.session_state.greetings = True  # 안내문구가 한 번만 표시되도록 상태 업데이트
        st.rerun()

    # 상태 관리: 버튼이 눌리지 않았을 때
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None

    # 대화 메시지 출력
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 버튼이 눌리지 않았을 때 알고리즘 버튼 출력
    if st.session_state.button_pressed is None:
        cols = st.columns(3)  # 3열 레이아웃
        for idx, (algo, prompt) in enumerate(ALGORITHM_PROMPTS.items()):
            with cols[idx % 3]:
                if st.button(algo):
                    st.session_state.button_pressed = algo
                    st.session_state.system_prompt = prompt
                    st.rerun()

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

    # 프롬프트 입력 창 (st.chat_input() 사용)
    user_input = st.chat_input("Your prompt:")
    if user_input:  # 사용자가 입력을 하면
        if user_input.strip():
            # LLM 응답 생성
            response = get_huggingface_response(st.session_state["model"], user_input)
            
            # 메시지 기록 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # prompt 기여도 계산
            
            # 기여도 시각화

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
    