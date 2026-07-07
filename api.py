from fastapi import FastAPI
from pydantic import BaseModel
from agents.supervisor import supervisor_workflow
from agents.state import AgentState

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.post("/chat")  # 改为 /chat，更符合 Agent 语义
def chat(request: QueryRequest):
    # 初始化状态
    state = AgentState(
        user_query=request.question,
        intent=None,
        messages=[],
        retrieved_docs=[],
        intermediate={},
        final_answer=""
    )

    # 执行主管工作流
    result = supervisor_workflow.invoke(state)

    return {
        "answer": result["final_answer"],
        "intent": result.get("intent"),
        "details": result.get("intermediate", {})
    }


# 保留旧的 /qa 接口（兼容）
@app.post("/qa")
def answer(request: QueryRequest):
    return chat(request)