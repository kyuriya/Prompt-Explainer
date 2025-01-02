import streamlit as st

# Page Config
st.set_page_config(page_title="이름 뭘로 하지", layout="wide", page_icon="👩‍💻")




# 중앙과 오른쪽 영역 레이아웃 비율
col2, col3 = st.columns([8, 3])

# 중앙 섹션
with col2:
    st.title("이름 뭘로 하지")

    def display_chat_messages() -> None:
        """Print message history
        @returns None
        """
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    def highlight_prompt_with_contributions(prompt, contribution_scores):
        tokens = prompt.split()
        highlighted_prompt = ""
        for token in tokens:
            contribution = contribution_scores.get(token, 0)  # 기여도가 없는 경우 기본값 0
            color_intensity = int(contribution * 255)  # 기여도에 따른 색상 강도 (0~255)
            highlighted_prompt += f'<span style="background-color: rgba(255, 0, 0, {contribution});">{token}</span> '
        return f"<p>{highlighted_prompt}</p>"


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.greetings = False

    # Display chat messages from history on app rerun
    display_chat_messages()
    

    # Greet user
    if not st.session_state.greetings:
        with st.chat_message("assistant"):
            intro = "안내문구"
            st.markdown(intro)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": intro})
            st.session_state.greetings = True

    button_pressed = ""
    system_prompt = ""

    # 버튼 데이터
    button_data = [
        # 시스템 프롬프트 예시
        {"text": "Depth-First Search(DFS)", 
        "system_prompt": """
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
            """},
        {"text": "Breadth-First Search(BFS)", "system_prompt": "BFS 시스템 프롬프트"},
        {"text": "Dynamic Programming(DP)", "system_prompt": "DP 시스템 프롬프트"},
        {"text": "Sort Algorithm", "system_prompt": "Sort 시스템 프롬프트"},
        {"text": "Greedy Algorithm", "system_prompt": "Greedy 시스템 프롬프트"},
        {"text": "최단 경로 알고리즘", "system_prompt": "최단경로 시스템 프롬프트"},
    ]
    
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None

    #사용자 프롬프트에 대한 페이지 기록
    if "page_history" not in st.session_state:
        st.session_state.page_history = []  # 사용자 프롬프트 기록 초기화


    if st.session_state.button_pressed is None:
        cols = st.columns(3)
        for idx, button in enumerate(button_data):
            col = cols[idx % 3]  # 3열 반복
            with col:
                if st.button(button["text"], key=f"button_{idx}"):
                    st.session_state.button_pressed = button["text"]
                    st.session_state.system_prompt = button["system_prompt"]
                    st.rerun()

    else:
        col = st.columns(1)[0]  # 중앙 정렬을 위해 1열 사용
        with col:
            if st.button(st.session_state.button_pressed, key="selected_button"):
                st.session_state.button_pressed = None  # 상태 초기화
                st.session_state.system_prompt = None

        if st.button("다시 선택"):
            st.session_state.button_pressed = None  # 상태 초기화
            st.session_state.system_prompt = None
            st.rerun()

    
if prompt := (st.chat_input("프롬프트를 입력하세요.")):
    prompt = prompt.replace('"', "").replace("'", "")

    if prompt != "":
        query = prompt.strip()
        #llm_response = API
        #contribution_scores = 기여도 계산 점수

        # highlighted_prompt = highlight_prompt_with_contributions(prompt, contribution_scores)

        # Display user message in chat message container
        # with st.chat_message("user"):
        #     st.markdown(highlighted_prompt, unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.page_history.append(prompt)
        
        # sorted_scores = sorted(contribution_scores.items(), key=lambda x: x[1], reverse=True) #scores 딕셔너리
        # top_10 = sorted_scores[:10]
        # bottom_10 = sorted_scores[-10:]

        # with st.chat_message("assistant"): #LLM 응답 출력
        #     st.markdown(f"LLM Response: {llm_response}")
        #     st.session_state.messages.append({"role": "assistant", "content": llm_response})
        #     st.markdown('<hr style="border: 1px solid #ccc;">', unsafe_allow_html=True) #중앙 구분선

        #     cols = st.columns(len(top_10))
        #     for idx, (word, score) in enumerate(top_10):
        #         with cols[idx]:
        #             st.markdown(
        #                 f'<button title="기여도: {score:.2f}" style="background-color: rgba(255, 0, 0, {score}); '
        #                 'border: none; border-radius: 8px; padding: 8px; color: white; font-weight: bold; cursor: pointer;">'
        #                 f'{word}</button>',
        #                 unsafe_allow_html=True,
        #             )

        #     cols = st.columns(len(bottom_10))
        #     for idx, (word, score) in enumerate(bottom_10):
        #         with cols[idx]:
        #             st.markdown(
        #                 f'<button title="기여도: {score:.2f}" style="background-color: rgba(0, 0, 255, {score}); '
        #                 'border: none; border-radius: 8px; padding: 8px; color: white; font-weight: bold; cursor: pointer;">'
        #                 f'{word}</button>',
        #                 unsafe_allow_html=True,
        #             )
            
        #     st.session_state.messages.append({
        #         "role": "assistant",
        #         "content": {
        #             "llm_response": llm_response,
        #             "highlighted_prompt": highlighted_prompt,
        #             "top_10": top_10,
        #             "bottom_10": bottom_10,
        #         },
        #     })

        st.rerun()


# 오른쪽 섹션 - 수정된 프롬프트 추천
with col3:
    st.header("대체 프롬프트 추천")

    def suggest_alternative_prompt(original_prompt):
        """
        간단한 대체 프롬프트 추천 로직.
        사용자의 입력 프롬프트를 기반으로 더 나은 대안을 제시.
        """
        # 여기에서 모듈이나 API를 사용해 수정된 프롬프트 생성 가능
        if "DFS" in original_prompt:
            return "DFS 구현 방법을 설명해주세요."
        elif "정렬" in original_prompt:
            return "퀵 정렬과 병합 정렬의 차이점을 설명해주세요."
        elif len(original_prompt) < 10:
            return "더 구체적인 질문을 입력해주세요."
        else:
            return f"수정된 프롬프트: {original_prompt} (내용 보강)"

    # 수정된 프롬프트 표시
    if st.session_state.page_history:
        last_prompt = st.session_state.page_history[-1]
        alternative_prompt = suggest_alternative_prompt(last_prompt)

        # st.write("입력된 프롬프트:")
        # st.info(last_prompt)

        st.write("추천된 프롬프트:")
        st.success(alternative_prompt)
    else:
        st.write("아직 입력된 프롬프트가 없습니다.")




# 왼쪽 사이드바 - Page History & 시스템 프롬프트

# 왼쪽 사이드바 스타일 적용
sidebar_style = """
    <style>
        .custom-sidebar {
            margin-top: 50%; /* 화면의 중간쯤으로 이동 */
        }
        .custom-box {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)
with st.sidebar:
    # Page History 섹션
    st.title("Page History")
    if st.session_state.page_history:
        for idx, history in enumerate(reversed(st.session_state.page_history[-10:]), 1):  # 최신 10개만 표시
            st.write(f"{idx}. {history}")

    # 시스템 프롬프트 섹션
    # st.markdown('<div class="custom-sidebar">', unsafe_allow_html=True)
    st.title("시스템 프롬프트")
    system_prompt_content = st.session_state.system_prompt
    st.markdown(
        f"""
        <div class="custom-box">
            {system_prompt_content}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
