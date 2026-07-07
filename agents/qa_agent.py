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

def qa_agent(state: AgentState) -> AgentState:
    """答疑 Agent：检索 + Claude 生成"""
    query = state["user_query"]
    try:
        print(f"📥 [QA Agent] 收到问题: {query}")
        t0 = time.time()

        retriever = get_retriever(top_k=3)
        nodes = retriever.retrieve(query)
        print(f"⏱️ 检索到 {len(nodes)} 个片段")

        if not nodes:
            state["final_answer"] = "未找到相关文档。"
            state["intermediate"]["qa"] = ""
            return state

        context = "\n".join([n.text for n in nodes])
        prompt = f"基于以下资料回答用户问题：\n资料：{context}\n问题：{query}\n回答："

        llm = get_llm()
        print("🤖 正在调用 Claude...")
        response = llm.invoke(prompt)
        print("✅ 回答生成成功")

        state["final_answer"] = response.content
        state["intermediate"]["qa"] = response.content
        state["retrieved_docs"] = [n.text for n in nodes]

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["final_answer"] = f"错误: {str(e)}"
        state["intermediate"]["qa"] = ""

    return state