import redis
import json
import os
from typing import Optional, List, Dict, Any

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
SESSION_EXPIRE = int(os.getenv("SESSION_EXPIRE", 3600))  # 默认1小时

_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True  # 自动解码字符串
        )
    return _redis_client

def get_session(session_id: str) -> Optional[List[Dict[str, str]]]:
    """获取会话历史"""
    client = get_redis_client()
    key = f"session:{session_id}"
    data = client.get(key)
    if data:
        return json.loads(data)
    return []

def save_session(session_id: str, messages: List[Dict[str, str]]) -> None:
    """保存会话历史"""
    client = get_redis_client()
    key = f"session:{session_id}"
    client.setex(key, SESSION_EXPIRE, json.dumps(messages))

def add_message(session_id: str, role: str, content: str) -> None:
    """添加一条消息到会话"""
    messages = get_session(session_id) or []
    messages.append({"role": role, "content": content})
    save_session(session_id, messages)

def clear_session(session_id: str) -> None:
    """清除会话"""
    client = get_redis_client()
    client.delete(f"session:{session_id}")