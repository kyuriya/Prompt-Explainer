import streamlit as st
from back.llm_service import get_huggingface_response
from back.chat_storage import save_chat_history
# from back.explainability import compute_lime_values
# from front.visualization import display_lime_visualization

# 알고리즘별 시스템 프롬프트
ALGORITHM_PROMPTS = {
    "Depth-First Search(DFS)": """
            You are an algorithm implementation expert.
            The user wants to implement the Depth-First Search (DFS) algorithm.
            Follow these rules:
                1.	Write the DFS algorithm in Python.
                2.	The DFS algorithm should operate based on a graph represented using an adjacency list.
                3.	Start the traversal from the given start node and return the visited nodes in order.
                4.	Name the function dfs_traversal and ensure it takes the following parameters:
                •	graph: A graph in adjacency list format (dictionary).
                •	start: The node where the traversal should begin.
                5.	The function should return a list of nodes in the order they were visited.
                6.	The input graph can either be a connected graph or a disconnected graph.
                7.	Add simple comments to explain the key steps in the code.

            Additionally, consider the following:
                •	Ensure visited nodes are not processed more than once.
                •	If the input graph is empty, return an empty list.
                •	Choose either a recursive or stack-based implementation.
            """,
    "Breadth-First Search(BFS)": """
            You are an algorithm implementation expert.
            The user wants to implement the Breadth-First Search (BFS) algorithm.
            Follow these rules:
                1.	Write the BFS algorithm in Python.
                2.	The BFS algorithm should operate based on a graph represented using an adjacency list.
                3.	Start the traversal from the given start node and return the visited nodes in order.
                4.	Name the function bfs_traversal and ensure it takes the following parameters:
                •	graph: A graph in adjacency list format (dictionary).
                •	start: The node where the traversal should begin.
                5.	The function should return a list of nodes in the order they were visited.
                6.	The input graph can either be a connected graph or a disconnected graph.
                7.	Add simple comments to explain the key steps in the code.

            Additionally, consider the following:
                •	Ensure visited nodes are not processed more than once.
                •	If the input graph is empty, return an empty list.
                •	Use a queue to implement the traversal process.
            """ ,
    "Sort Algorithm": """
            You are an algorithm implementation expert.
            The user wants to implement a sorting algorithm.
            Follow these rules:
                1.	Write the sorting algorithm in Python.
                2.	The user wants to implement a specific sorting algorithm (e.g., Bubble Sort, Quick Sort, Merge Sort, etc.).
                3.	The sorting algorithm implementation must follow these rules:
                •	The function name should be sort_algorithm and should take the list to be sorted as a parameter.
                •	The function must return the sorted list.
                4.	The function should perform sorting in ascending order by default.
                5.	Add simple comments to explain the key steps of the algorithm.

            Additionally, consider the following:
                •	If the input list is empty or contains only one element, return it as is.
                •	Choose and implement an appropriate sorting algorithm (e.g., Bubble Sort using basic loops).
            """,
    "Greedy Algorithm": """
            You are an algorithm implementation expert.
            The user wants to implement a Greedy Algorithm.
            Follow these rules:
                1.	Write the Greedy Algorithm in Python.
                2.	Solve a specific problem type (e.g., Activity Selection Problem, Minimum Spanning Tree, Coin Change Problem) based on the user’s requirements.
                3.	The implemented function should include:
                •	The function name and parameters, defined dynamically based on the problem type.
                •	A clear explanation of the greedy criterion used for selection (e.g., maximum, minimum, etc.).
                4.	Include comments explaining the key steps of the algorithm.

            Additionally, consider the following:
                •	Clearly specify when the greedy algorithm guarantees an optimal solution.
                •	Include exception handling for problem input values.
            """,
    "Dynamic Programming(DP)": """
            You are an algorithm implementation expert.
            The user wants to solve a problem using Dynamic Programming (DP).
            Follow these rules:
                1.	Write the Dynamic Programming solution in Python.
                2.	Solve a specific problem type (e.g., Fibonacci sequence, Knapsack problem, Shortest Path, etc.) based on the user’s requirements.
                3.	The implemented function should include:
                •	A function name and parameters defined dynamically based on the problem type.
                •	Use either the memoization or tabulation approach for the DP solution.
                4.	Include comments explaining the key steps and the structure of the DP table.

            Additionally, consider the following:
                •	Clearly implement the recurrence relation for the problem.
                •	Ensure optimized time complexity.
                •	Handle cases where the input data is empty or invalid with exception handling.
            """,
    "Short Distance Algorithm": """
            You are an algorithm implementation expert.
            The user wants to implement a shortest path algorithm.
            Follow these rules:
                1.	Write the shortest path algorithm in Python.
                2.	The algorithm to be implemented should be one of the following: Dijkstra, Floyd-Warshall, or Bellman-Ford.
                3.	The implemented function must adhere to the following rules:
                •	Name the function shortest_path, and take the input graph and starting node as parameters.
                •	The graph should be represented as an adjacency list or a weighted matrix.
                •	Return the shortest distance values for each node.
                4.	Include comments explaining the key steps of the algorithm.

            Additionally, consider the following:
                •	Handle cases where the graph is empty by implementing appropriate exception handling.
                •	Optimize the time complexity of the implemented algorithm.
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
    