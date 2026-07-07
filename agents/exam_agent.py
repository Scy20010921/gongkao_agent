import os
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from core.retriever import get_retriever
from agents.state import AgentState

load_dotenv()

def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=30
    )

def exam_agent(state: AgentState) -> AgentState:
    """出题 Agent：基于知识库生成练习题"""
    query = state["user_query"]
    try:
        print(f"📝 [Exam Agent] 收到出题请求: {query}")
        t0 = time.time()

        # 1. 检索相关知识点（用于出题素材）
        retriever = get_retriever(top_k=5)
        nodes = retriever.retrieve(query)
        print(f"⏱️ 检索到 {len(nodes)} 个相关片段")

        if not nodes:
            state["final_answer"] = "未找到相关知识点，无法生成题目。"
            state["intermediate"]["exam"] = ""
            return state

        context = "\n".join([n.text for n in nodes])
        print(f"📄 上下文长度: {len(context)} 字符")

        # 2. 调用 Claude 生成题目
        prompt = f"""你是一名公考出题专家。请基于以下资料，生成一道公考风格的题目。

要求：
- 题型：选择题（单选）
- 难度：中等
- 包含：题干、选项（A/B/C/D）、正确答案、详细解析
- 选项之间要有区分度

资料：
{context}

请按以下格式输出：
【题目】...
【A】...
【B】...
【C】...
【D】...
【正确答案】...
【解析】...
"""

        llm = get_llm()
        print("🤖 正在生成题目...")
        t3 = time.time()
        response = llm.invoke(prompt)
        t4 = time.time()
        print(f"⏱️ 出题耗时: {t4 - t3:.2f}s")

        # 3. 组装返回结果
        state["final_answer"] = response.content
        state["intermediate"]["exam"] = {
            "question": response.content,
            "source_context": context[:200] + "..."
        }
        state["retrieved_docs"] = [n.text for n in nodes]

        print("✅ 题目生成成功")
        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["final_answer"] = f"出题失败: {str(e)}"
        state["intermediate"]["exam"] = ""
        return state