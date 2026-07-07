from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    # 用户原始问题
    user_query: str
    # 识别出的意图
    intent: Optional[str]  # "job" | "plan" | "qa" | "exam" | "grading" | "ops"
    # 对话历史（用于多轮）
    messages: List[Dict[str, str]]
    # 检索到的文档片段
    retrieved_docs: List[str]
    # 各 Agent 的中间结果
    intermediate: Dict[str, Any]
    # 最终回答
    final_answer: str