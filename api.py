from fastapi import FastAPI
from pydantic import BaseModel
from agents.supervisor import supervisor_workflow
from fastapi.responses import StreamingResponse

import asyncio
import os
import json
from core.retriever import get_retriever
from langchain_anthropic import ChatAnthropic
from agents.state import AgentState
from langsmith import traceable, run_trees
from core.redis_manager import get_session
from agents.supervisor import classify_intent
from agents.qa_agent import qa_agent_stream
from agents.exam_agent import exam_agent_stream
from agents.plan_agent import plan_agent_stream
from agents.grading_agent import grading_agent_stream
from agents.job_agent import job_agent_stream
from agents.ops_agent import ops_agent_stream
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


@app.post("/chat_stream")
async def chat_stream(request: QueryRequest):
    query = request.question
    session_id = request.session_id or str(uuid.uuid4())
    history = get_session(session_id) or []
    # 初始化状态
    state = AgentState(
        user_query=query,
        messages=history,  # ← 加载历史
        session_id=session_id,  # ← 传递 session_id
        intent=None,
        retrieved_docs=[],
        intermediate={},
        final_answer=""
    )

    async def generate():
        # 使用 LangGraph 的 astream，自动保持调用链
        async for event in supervisor_workflow.astream(state):
            # event 结构: {node_name: node_output}
            for node_name, node_output in event.items():
                # 如果是 Agent 节点，提取 final_answer 并流式输出
                if node_name in ["qa_agent", "ops_agent", "exam_agent", "plan_agent", "grading_agent", "job_agent"]:
                    # 注意：这里需要你的 Agent 内部支持流式
                    # 如果 Agent 内部是同步的，这里无法逐字输出
                    # 可以在这里调用对应的流式版本
                    pass
                # 如果节点输出包含 final_answer，一次性输出
                if "final_answer" in node_output:
                    answer = node_output["final_answer"]
                    if answer:
                        # 为了流式效果，逐字发送
                        for char in answer:
                            yield f"data: {json.dumps({'token': char})}\n\n"
                            await asyncio.sleep(0.01)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")