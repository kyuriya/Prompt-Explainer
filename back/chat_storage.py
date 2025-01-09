import json
import os

CHAT_HISTORY_FILE = "../chat_history.json"

def initialize_chat_storage():
    """대화 기록 JSON 파일 초기화"""
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump([], f)

def load_chat_history():
    """대화 기록 불러오기"""
    if not os.path.exists(CHAT_HISTORY_FILE):
        initialize_chat_storage()
    with open(CHAT_HISTORY_FILE, "r") as f:
        return json.load(f)

def save_chat_history(history):
    """대화 기록 저장"""
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)