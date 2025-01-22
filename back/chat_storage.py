import json
import os

CHAT_HISTORY_FILE = "back/chat_history.json"

def initialize_chat_storage():
    """대화 기록 JSON 파일 초기화"""
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump([], f)

def load_chat_history():
    """채팅 기록을 로드"""
    try:
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # JSON 파일이 손상되었거나 없는 경우 빈 리스트 반환
        return []

def save_chat_history(chat_history):
    """채팅 기록을 저장"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")