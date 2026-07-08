import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.qa_agent import qa_agent
from agents.exam_agent import exam_agent
from agents.plan_agent import plan_agent
from agents.grading_agent import grading_agent
from agents.job_agent import job_agent
load_dotenv()

# 初始化 LLM（用于意图识别）
llm = ChatAnthropic(
    model="claude-sonnet-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30
)


# ===== 意图识别函数 =====
def classify_intent(state: AgentState) -> AgentState:
    """识别用户意图，只输出一个词"""
    query = state["user_query"]
    prompt = f"""判断用户问题属于以下哪个类别，只输出一个词：
    - job: 选岗、职位推荐、岗位匹配
    - plan: 学习规划、备考计划、时间安排
    - qa: 答疑、知识点解释、政策咨询
    - exam: 出题、组卷、生成练习题
    - grading: 批改、评分、写作批改
    - ops: 内容运营、热点资讯、备考素材

    问题：{query}
    类别："""

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        # 如果是列表，提取所有文本并合并
        content = " ".join([block.text if hasattr(block, 'text') else str(block) for block in content])
    intent = content.strip().lower()

    # 安全兜底
    if intent not in ["job", "plan", "qa", "exam", "grading", "ops"]:
        intent = "qa"  # 默认走答疑

    state["intent"] = intent
    print(f"🔍 意图识别结果: {intent}")
    return state


# ===== 路由函数 =====
def route_to_agent(state: AgentState) -> str:
    """根据意图路由到对应的子 Agent"""
    intent = state.get("intent", "qa")
    routing_map = {
        "qa": "qa_agent",
        "job": "job_agent",  # 暂时先用 qa_agent 占位
        "plan": "plan_agent",  # 后续替换为真实的子 Agent
        "exam": "exam_agent",
        "grading": "grading_agent",
        "ops": "qa_agent",

    }
    return routing_map.get(intent, "qa_agent")


# ===== 构建主管图 =====
supervisor_graph = StateGraph(AgentState)

# 添加节点
supervisor_graph.add_node("classify", classify_intent)
supervisor_graph.add_node("qa_agent", qa_agent)  # 目前只有一个，后续会扩展
supervisor_graph.add_node("exam_agent", exam_agent)
supervisor_graph.add_node("plan_agent", plan_agent)
supervisor_graph.add_node("grading_agent", grading_agent)
supervisor_graph.add_node("job_agent", job_agent)

# 设置入口
supervisor_graph.set_entry_point("classify")

# 添加条件边：从 classify 路由到对应 Agent
supervisor_graph.add_conditional_edges(
    "classify",
    route_to_agent,
    {
        "qa_agent": "qa_agent",
        "exam_agent": "exam_agent",
        "plan_agent": "plan_agent",
        "grading_agent": "grading_agent",
        "job_agent": "job_agent",
        # 后续添加更多 Agent 时在这里扩展
    }
)

# 所有子 Agent 结束后进入 END
supervisor_graph.add_edge("qa_agent", END)
supervisor_graph.add_edge("exam_agent", END)
supervisor_graph.add_edge("plan_agent", END)
supervisor_graph.add_edge("grading_agent", END)
supervisor_graph.add_edge("job_agent", END)
# 编译
supervisor_workflow = supervisor_graph.compile()