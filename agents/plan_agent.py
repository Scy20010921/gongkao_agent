import os
import time
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from core.retriever import get_retriever
from agents.state import AgentState
from core.redis_manager import get_session
load_dotenv()


def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=120.0,      # 增加超时时间到 120 秒
        max_retries=1,      # 减少重试次数，避免总等待时间过长
        # max_retries=0     # 也可以完全禁用重试
    )


def extract_plan_params(query: str) -> dict:
    """简单提取用户提到的关键信息（可后续用LLM优化）"""
    params = {"target": "国考", "time": "3个月", "daily_hours": 2, "level": "零基础"}
    if "省考" in query:
        params["target"] = "省考"
    if "6个月" in query:
        params["time"] = "6个月"
    elif "1年" in query:
        params["time"] = "1年"
    if "每天" in query:
        match = re.search(r"每天(\d+)小时", query)
        if match:
            params["daily_hours"] = int(match.group(1))
    if "基础" in query:
        if "零基础" in query:
            params["level"] = "零基础"
        elif "有基础" in query:
            params["level"] = "有基础"
    return params


def plan_agent(state: AgentState) -> AgentState:
    query = state["user_query"]
    session_id = state.get("session_id", "default")
    history = get_session(session_id) or []
    history_text = ""
    if history:
        recent = history[-6:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    try:
        print(f"📋 [Plan Agent] 收到规划请求: {query}")

        # 1. 提取规划参数
        params = extract_plan_params(query)
        print(f"📌 规划参数: {params}")

        # 2. 检索通用备考知识（行测/申论备考策略）
        retriever = get_retriever(top_k=4)
        nodes = retriever.retrieve("行测 申论 备考 计划 方法")
        context = "\n".join([n.text for n in nodes])
        print(f"📄 检索到 {len(nodes)} 个相关片段")

        # 3. 调用 Claude 生成规划
        prompt = f"""你是一名公考备考规划专家。请根据用户需求，制定一份详细的学习计划。

        【对话历史】
        {history_text if history_text else "无"}

        【用户信息】
        - 目标考试：{params['target']}
        - 备考时间：{params['time']}
        - 每天学习时长：{params['daily_hours']}小时
        - 基础水平：{params['level']}

        【参考备考知识】
        {context}

        请输出规划（按以下结构）：
        ### 总体目标
        ### 阶段划分
        ### 每日时间分配建议
        ### 行测各模块学习方法
        ### 申论备考建议
        ### 推荐资料
        ### 注意事项
        """
        llm = get_llm()
        print("🤖 正在生成学习规划...")
        t0 = time.time()
        response = llm.invoke(prompt)
        t1 = time.time()
        print(f"⏱️ 规划生成耗时: {t1 - t0:.2f}s")

        state["final_answer"] = response.content
        state["intermediate"]["plan"] = {
            "params": params,
            "plan_text": response.content
        }
        state["retrieved_docs"] = [n.text for n in nodes]
        print("✅ 学习规划生成成功")
        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["final_answer"] = f"生成规划失败: {str(e)}"
        state["intermediate"]["plan"] = {}
        return state

async def plan_agent_stream(state: AgentState):
    query = state["user_query"]
    session_id = state.get("session_id", "default")
    history = get_session(session_id) or []
    history_text = ""
    if history:
        recent = history[-6:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    prompt = f"""你是一名公考备考规划专家。请根据用户需求制定详细学习计划。
    【对话历史】
    {history_text if history_text else "无"}
    
    【用户需求】
    {query}
        
    【参考知识】
    {context}
    请按以下结构输出：
    ### 总体目标
    ### 阶段划分
    ### 每日时间分配
    ### 各模块学习方法
    ### 推荐资料"""

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        streaming=True,
        max_tokens=4096
    )
    async for chunk in llm.astream(prompt):
        content = chunk.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    yield block.get('text', '')
                elif hasattr(block, 'type') and block.type == 'text':
                    yield block.text
        else:
            yield str(content)