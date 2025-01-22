import streamlit as st
from back.llm_service import get_huggingface_response
from back.chat_storage import save_chat_history
from back.captum_utils import generate_heatmap  # Utility for Captum heatmap
from io import BytesIO
from PIL import Image
import torch
import gc
from datetime import datetime
import os
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
                2.	Solve a specific problem type (e.g., Activity Selection Problem, Minimum Spanning Tree, Coin Change Problem) based on the user's requirements.
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
                2.	Solve a specific problem type (e.g., Fibonacci sequence, Knapsack problem, Shortest Path, etc.) based on the user's requirements.
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
# GPU 메모리 및 캐시 초기화 함수
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
def render_main_page():
    clear_gpu_cache()
    """중앙 메인 페이지 구현"""
    st.header(f"Conversation {st.session_state['current_page']}")

    # 안내문구 표시 (처음 한 번만)
    if not st.session_state.get("greetings", False):
        with st.chat_message("assistant"):
            intro = """
            Welcome to **Prompt Explainer**! 🤵🏻‍♀️\n
            This tool is designed to help you leverage LLMs (Large Language Models) more effectively when **solving algorithm problems**. ⛳️\n
            By visually highlighting which **parts of the prompt the LLM focuses on**, you can craft **better prompts** and receive **higher-quality response codes**. 🎲\n
            When you input a prompt, we will visualize the emphasized sections based on **SHAP values**. This allows you to learn better **prompt-writing strategies** and **maximize the utility of LLMs** in your workflow. 🎞️\n 
            Give it a try and enhance your experience in solving algorithmic problems! 🎸
            """
            st.markdown(intro)
            st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True
        st.rerun()

    # 상태 관리: 버튼이 눌리지 않았을 때
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None

    # 대화 메시지 출력
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "heatmap" in message:
                # base64 문자열을 이미지 데이터로 변환
                image_bytes = base64.b64decode(message["heatmap"])
                st.image(image_bytes)  # 바이트 데이터로 직접 전달

    # 알고리즘 버튼 출력
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None

    # 버튼이 눌리지 않았을 때 알고리즘 버튼 출력
    if st.session_state.button_pressed is None:
        cols = st.columns(3)
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
    # if user_input:  # 사용자가 입력을 하면
    #     if user_input.strip():
    #         try:
    #             response = get_huggingface_response(st.session_state["model"], user_input)
    #             if response:
    #                 # 메시지 기록 추가
    #                 st.session_state.messages.append({"role": "user", "content": user_input})
    #                 st.session_state.messages.append({"role": "assistant", "content": response})
                # 사용자 입력 메시지 즉시 표시
                # st.session_state.messages.append({"role": "user", "content": user_input})
                
                # LLM 응답 생성
                # response = get_huggingface_response(st.session_state["model"], user_input)
                
                # if response:
                    # 응답 메시지 즉시 표시
                    # st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # 히트맵 생성
                    # with st.spinner('Generating attribution heatmap...'):
                    #     heatmap_buffer = generate_heatmap(st.session_state["model"], user_input, response)
                        
                    #     if heatmap_buffer:
                    #         # 히트맵 이미지 저장
                    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    #         save_path = f"heatmaps/heatmap_{timestamp}.png"
                    #         os.makedirs("heatmaps", exist_ok=True)
                            
                    #         # BytesIO를 이미지로 변환하고 저장
                    #         heatmap_image = Image.open(heatmap_buffer)
                    #         heatmap_image.save(save_path)
                        
            #         # 히트맵 표시
            #         st.image(heatmap_image, caption="Prompt Attribution Heatmap", use_column_width=True)
                    
            #         # 대화 기록 저장
            #         chat_history = st.session_state["chat_history"]
            #         current_page = st.session_state["current_page"] - 1
                    
            #         if current_page < len(chat_history):
            #             chat_history[current_page] = {"messages": st.session_state["messages"]}
            #         else:
            #             chat_history.append({"messages": st.session_state["messages"]})

            #         save_chat_history(chat_history)
                    
            #         st.rerun()

            # except Exception as e:
            #     st.error(f"An error occurred: {str(e)}")
        
    # if user_input:  # 사용자가 프롬프트를 입력하면
    #     # if user_input.strip():
    #     if user_input is not None and user_input.strip():
    #         try:
    #             response_lines = get_huggingface_response(st.session_state["model"], user_input)
    #             if response_lines:
    #                 # 메시지 기록 추가
    #                 st.session_state.messages.append({"role": "user", "content": user_input})
    #                 st.session_state.messages.append({"role": "assistant", "content": "\n".join(response_lines)})

    #                 # Captum 기여도 계산
    #                 json_path = generate_heatmap(st.session_state["model"], user_input, response_lines)

    #                 if json_path:
    #                     # JSON 파일을 기반으로 히트맵 생성 및 표시
    #                     st.markdown("### Prompt Attribution Heatmap")
    #                     with open(json_path, "r") as json_file:
    #                         attribution_data = json.load(json_file)
    #                     # 각 줄의 히트맵 생성 (여기서 구현 필요)
    #                     for idx, data in attribution_data.items():
    #                         st.markdown(f"**{data['line']}**")
    #                         # 히트맵 시각화 코드 삽입
    #                         st.image(f"heatmaps/heatmap_{idx}.png", use_column_width=True)

    #                 # save_chat_history(st.session_state["chat_history"])
    #                 # st.rerun()
    #             # 대화 기록 관리
    #                 chat_history = st.session_state["chat_history"]
    #                 current_page = st.session_state["current_page"] - 1

    #                 if current_page < len(chat_history):
    #                     # 기존 페이지 업데이트
    #                     chat_history[current_page] = {"messages": st.session_state["messages"]}
    #                 else:
    #                     # 새로운 페이지 추가
    #                     chat_history.append({"messages": st.session_state["messages"]})

    #                 # 대화 기록 저장
    #                 save_chat_history(chat_history)

    #                 # UI 업데이트
    #                 st.rerun()

    #         except Exception as e:
    #             st.error(f"An error occurred: {str(e)}")

    # if user_input:  # 사용자가 프롬프트를 입력하면
    #     # if user_input.strip():
    #     if user_input is not None and user_input.strip():
    #         try:
    #             response_lines = get_huggingface_response(st.session_state["model"], user_input)
    #             if response_lines:
    #                 # 메시지 기록 추가
    #                 st.session_state.messages.append({"role": "user", "content": user_input})
    #                 # st.session_state.messages.append({"role": "assistant", "content": "\n".join(response_lines)})
    #                 st.session_state.messages.append({"role": "assistant", "content": response_lines})

    #                 # Captum 기여도 계산
    #                 with st.spinner('Generating attribution heatmap...'):
    #                     heatmap_buffer = generate_heatmap(st.session_state["model"], user_input, response_lines)
                        
    #                     if heatmap_buffer:
    #                         # 임시 파일로 저장
    #                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                         temp_path = f"./heatmaps/heatmap_{timestamp}.png"
    #                         os.makedirs("./heatmaps", exist_ok=True)
                            
    #                         with open(temp_path, "wb") as f:
    #                             f.write(heatmap_buffer.getvalue())
                            
    #                         # 파일을 base64로 인코딩
    #                         with open(temp_path, "rb") as f:
    #                             heatmap_base64 = base64.b64encode(f.read()).decode()
                            
    #                         # 응답과 히트맵을 함께 메시지에 저장
    #                         st.session_state.messages.append({
    #                             "role": "assistant", 
    #                             "content": response_lines,
    #                             "heatmap": heatmap_base64
    #                         })
                            
    #                         # 임시 파일 삭제
    #                         os.remove(temp_path)

    #                 # 히트맵 표시
    #                 st.image(f"heatmaps/heatmap_{timestamp}.png", caption="Prompt Attribution Heatmap", use_column_width=True)

    #                 # 대화 기록 관리
    #                 chat_history = st.session_state["chat_history"]
    #                 current_page = st.session_state["current_page"] - 1

    #                 if current_page < len(chat_history):
    #                     # 기존 페이지 업데이트
    #                     chat_history[current_page] = {"messages": st.session_state["messages"]}
    #                 else:
    #                     # 새로운 페이지 추가
    #                     chat_history.append({"messages": st.session_state["messages"]})

    #                 # 대화 기록 저장
    #                 save_chat_history(chat_history)

    #                 # UI 업데이트
    #                 st.rerun()

    #         except Exception as e:
    #             st.error(f"An error occurred: {str(e)}")
    if user_input:
        if user_input.strip():
            try:
                # LLM 응답 생성
                response = get_huggingface_response(st.session_state["model"], user_input)
                if response:
                    # Captum 기여도 계산
                    heatmap_buffer = generate_heatmap(st.session_state["model"], user_input, response)

                    if heatmap_buffer:
                        # Streamlit에 이미지 표시
                        st.markdown("### Prompt Attribution Heatmap")
                        st.image(heatmap_buffer, caption="Attribution Heatmap", use_column_width=True)

                        # 채팅 메시지 기록에 추가
                        heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode("utf-8")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "heatmap": heatmap_base64  # Base64 인코딩된 이미지 저장
                        })

                        # 채팅 기록 업데이트
                        chat_history = st.session_state["chat_history"]
                        current_page = st.session_state["current_page"] - 1

                        if current_page < len(chat_history):
                            chat_history[current_page] = {"messages": st.session_state["messages"]}
                        else:
                            chat_history.append({"messages": st.session_state["messages"]})

                        save_chat_history(chat_history)
                        st.rerun()
              
      


            except Exception as e:
                st.error(f"An error occurred: {str(e)}")