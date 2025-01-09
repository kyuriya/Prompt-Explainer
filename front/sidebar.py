import streamlit as st
from backend.chat_storage import save_chat_history, load_chat_history

def render_sidebar():
    """왼쪽 사이드바 구현"""
    st.sidebar.title("Conversations")

    # Make New Page 버튼
    if st.sidebar.button("Make New Page"):
        chat_history = load_chat_history()
        chat_history.append({"messages": []})  # 새 대화는 빈 메시지 리스트로 추가
        save_chat_history(chat_history)
        st.session_state["current_page"] = len(chat_history)  # 새 페이지로 이동
        st.session_state["messages"] = []  # 현재 메시지 초기화
        st.rerun()

    # 현재 페이지 표시 및 이동 버튼
    chat_history = st.session_state["chat_history"]
    for idx in range(1, len(chat_history) + 1):
        if st.sidebar.button(f"Conversation {idx}"):
            st.session_state["current_page"] = idx
            st.session_state["messages"] = chat_history[idx - 1]["messages"]  # 페이지에 해당하는 메시지 로드
            st.rerun()

    # 왼쪽 사이드바에 현재 선택된 시스템 프롬프트 표시
    if "current_prompt" in st.session_state and st.session_state["current_prompt"]:
        # st.sidebar.subheader("System Prompt")
        st.sidebar.title("System Prompt")
        st.sidebar.info(st.session_state["current_prompt"])