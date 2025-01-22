import streamlit as st

def render_layout():
    """좌측, 중앙, 우측 레이아웃 정의"""
    col2, col3 = st.columns([8, 3])
    return col2, col3