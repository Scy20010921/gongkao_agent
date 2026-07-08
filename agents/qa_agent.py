import os
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from core.retriever import get_retriever
from agents.state import AgentState
from langsmith import traceable
from core.redis_manager import get_session, add_message
load_dotenv()

async def qa_agent_stream(state: AgentState):
    query = state["user_query"]
    retriever = get_retriever(top_k=3)
    nodes = retriever.retrieve(query)
    context = "\n".join([n.text for n in nodes])
    prompt = f"基于以下资料回答：\n资料：{context}\n问题：{query}\n回答："

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

def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=30,
        max_retries=1
    )
@traceable(run_type="chain", name="qa_agent")
def qa_agent(state: AgentState) -> AgentState:
    """答疑 Agent：检索 + Claude 生成"""
    query = state["user_query"]
    session_id = state.get("session_id", "default")
    try:
        print(f"📥 [QA Agent] 收到问题: {query}, session: {session_id}")

        # 1. 获取历史对话（最近3轮）
        history = get_session(session_id) or []
        history_text = ""
        if history:
            # 只取最近3轮（6条消息）
            recent = history[-6:]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
            print(f"📚 历史上下文: {len(recent)} 条消息")

        # 2. 检索相关文档
        retriever = get_retriever(top_k=3)
        nodes = retriever.retrieve(query)
        context = "\n".join([n.text for n in nodes])

        # 3. 构建包含历史上下文的 prompt
        prompt = f"""你是一名公考助手。请基于以下信息回答用户问题。

    【对话历史】
    {history_text if history_text else "无"}

    【参考资料】
    {context}

    【当前问题】
    {query}

    【回答】
    """
        llm = get_llm()
        print("🤖 正在调用 Claude...")
        response = llm.invoke(prompt)
        answer = response.content

        # 4. 保存对话到 Redis
        add_message(session_id, "user", query)
        add_message(session_id, "assistant", answer)

        state["final_answer"] = answer
        state["intermediate"]["qa"] = answer
        state["retrieved_docs"] = [n.text for n in nodes]
        state["messages"] = get_session(session_id) or []

        print("✅ 回答生成成功，已保存会话")
        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["final_answer"] = f"错误: {str(e)}"
        state["intermediate"]["qa"] = ""
        return state