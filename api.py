from fastapi import FastAPI
from pydantic import BaseModel
from agents.supervisor import supervisor_workflow
from agents.state import AgentState
from langsmith import traceable
from core.redis_manager import get_session
import uuid
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str = None  # 可选，如果不提供则自动生成


@app.post("/chat")  # 改为 /chat，更符合 Agent 语义
@traceable(run_type="chain", name="chat_endpoint")
def chat(request: QueryRequest):
    # 生成或使用 session_id
    session_id = request.session_id or str(uuid.uuid4())
    # 获取历史消息（用于状态初始化）
    history = get_session(session_id) or []
    # 初始化状态
    state = AgentState(
        user_query=request.question,
        intent=None,
        messages=history,   # 加载历史
        session_id=session_id,
        retrieved_docs=[],
        intermediate={},
        final_answer=""
    )

    # 执行主管工作流
    result = supervisor_workflow.invoke(state)

    return {
        "session_id": session_id,
        "answer": result["final_answer"],
        "intent": result.get("intent"),
        "details": result.get("intermediate", {})
    }


# 保留旧的 /qa 接口（兼容）
@app.post("/qa")
def answer(request: QueryRequest):
    return chat(request)

@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    """查看会话历史（调试用）"""
    history = get_session(session_id)
    return {"session_id": session_id, "history": history}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    from core.redis_manager import clear_session
    clear_session(session_id)
    return {"status": "cleared"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "gongkao-agent"}