import os
import time
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from core.retriever import get_retriever
from agents.state import AgentState

load_dotenv()


def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=90  # 批改可能需要更长生成时间
    )


def extract_essay(query: str) -> str:
    """从用户问题中提取申论作文内容（简单实现，可后续优化）"""
    # 目前假设用户直接粘贴作文，或者用引号包围
    # 如果问题中明确包含“作文：”或“我的文章：”，则提取后面内容
    for prefix in ["作文：", "我的文章：", "申论：", "文章内容："]:
        if prefix in query:
            return query.split(prefix, 1)[1].strip()
    # 否则直接返回原问题（假设就是作文）
    return query


def grading_agent(state: AgentState) -> AgentState:
    query = state["user_query"]
    try:
        print(f"📝 [Grading Agent] 收到批改请求")

        # 1. 提取作文内容
        essay = extract_essay(query)
        print(f"📄 待批改作文长度: {len(essay)} 字符")
        if len(essay) < 20:
            state["final_answer"] = "请提供完整的申论作文内容。"
            state["intermediate"]["grading"] = {}
            return state

        # 2. 检索评分标准和范文
        retriever = get_retriever(top_k=4)
        nodes = retriever.retrieve("申论 评分标准 写作技巧 范文")
        context = "\n".join([n.text for n in nodes])
        print(f"📚 检索到 {len(nodes)} 个相关参考片段")

        # 3. 调用 Claude 进行批改
        prompt = f"""你是一名资深的申论批改专家。请对以下作文进行批改，并给出详细的评分报告。

**评分标准参考**（基于公考申论评分规则）：
{context}

**用户作文**：
{essay}

请按以下格式输出批改报告：
### 总体评分
（给出一个总分，如 65/100，并简要说明）

### 评分维度拆解
- 内容充实度（满分25）：得分 X，理由...
- 逻辑结构（满分25）：得分 X，理由...
- 语言表达（满分25）：得分 X，理由...
- 思想深度/观点（满分25）：得分 X，理由...

### 主要优点
（列出2-3个亮点）

### 主要问题
（列出2-3个不足，并具体指出问题所在）

### 改进建议
（给出具体的修改方向，可举例）

### 参考范文片段
（从资料库中摘录一段与题目相关的优秀范文片段，供用户参考）
"""
        llm = get_llm()
        print("🤖 正在批改作文...")
        t0 = time.time()
        response = llm.invoke(prompt)
        t1 = time.time()
        print(f"⏱️ 批改耗时: {t1 - t0:.2f}s")

        state["final_answer"] = response.content
        state["intermediate"]["grading"] = {
            "essay": essay,
            "report": response.content
        }
        state["retrieved_docs"] = [n.text for n in nodes]
        print("✅ 批改完成")
        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["final_answer"] = f"批改失败: {str(e)}"
        state["intermediate"]["grading"] = {}
        return state