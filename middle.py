import streamlit as st
import time
import sys
import os

# Title
st.title("이름 뭘로 하지")

# Connection to Weaviate thorugh Connector
#conn = st.connection(
#   "weaviate",
#    type=#데이터베이스,
#    url=os.getenv("WEAVIATE_URL"),
#    api_key=os.getenv("WEAVIATE_API_KEY"),
#    additional_headers={"X-OpenAI-Api-Key": openai_key},
#)

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

#사이드바 추가 

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
    {"text": "Depth-First Search(DFS)", "system_prompt": "DFS 시스템 프롬프트"},
    {"text": "Breadth-First Search(BFS)", "system_prompt": "BFS 시스템 프롬프트"},
    {"text": "Dynamic Programming(DP)", "system_prompt": "DP 시스템 프롬프트"},
    {"text": "Sort Algorithm", "system_prompt": "Sort 시스템 프롬프트"},
    {"text": "Greedy Algorithm", "system_prompt": "Greedy 시스템 프롬프트"},
    {"text": "최단 경로 알고리즘", "system_prompt": "최단경로 시스템 프롬프트"},
]

if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = None
    st.session_state.system_prompt = None

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

        highlighted_prompt = highlight_prompt_with_contributions(prompt, contribution_scores)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(highlighted_prompt, unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        sorted_scores = sorted(contribution_scores.items(), key=lambda x: x[1], reverse=True) #scores 딕셔너리
        top_10 = sorted_scores[:10]
        bottom_10 = sorted_scores[-10:]

        with st.chat_message("assistant"): #LLM 응답 출력
            st.markdown(f"LLM Response: {llm_response}")
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
            st.markdown('<hr style="border: 1px solid #ccc;">', unsafe_allow_html=True) #중앙 구분선

            cols = st.columns(len(top_10))
            for idx, (word, score) in enumerate(top_10):
                with cols[idx]:
                    st.markdown(
                        f'<button title="기여도: {score:.2f}" style="background-color: rgba(255, 0, 0, {score}); '
                        'border: none; border-radius: 8px; padding: 8px; color: white; font-weight: bold; cursor: pointer;">'
                        f'{word}</button>',
                        unsafe_allow_html=True,
                    )

            cols = st.columns(len(bottom_10))
            for idx, (word, score) in enumerate(bottom_10):
                with cols[idx]:
                    st.markdown(
                        f'<button title="기여도: {score:.2f}" style="background-color: rgba(0, 0, 255, {score}); '
                        'border: none; border-radius: 8px; padding: 8px; color: white; font-weight: bold; cursor: pointer;">'
                        f'{word}</button>',
                        unsafe_allow_html=True,
                    )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "llm_response": llm_response,
                    "highlighted_prompt": highlighted_prompt,
                    "top_10": top_10,
                    "bottom_10": bottom_10,
                },
            })

            st.rerun()
