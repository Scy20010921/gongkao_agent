    import os
    import time
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    from core.retriever import get_retriever
    from agents.state import AgentState
    from core.redis_manager import get_session
    load_dotenv()


    def get_llm():
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=90,
            max_retries=1
        )


    def ops_agent(state: AgentState) -> AgentState:
        query = state["user_query"]
        session_id = state.get("session_id", "default")
        history = get_session(session_id) or []
        history_text = ""
        if history:
            recent = history[-6:]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        try:
            print(f"📢 [Ops Agent] 收到运营请求: {query}")

            # 判断任务类型
            if "每日一练" in query or "练习题" in query:
                task_type = "daily_exercise"
            elif "热点" in query or "话题" in query:
                task_type = "hot_topic"
            elif "金句" in query or "素材" in query:
                task_type = "golden_sentence"
            elif "一周计划" in query:
                task_type = "weekly_plan"
            else:
                task_type = "general"

            # 检索相关知识（如果需要）
            retriever = get_retriever(top_k=3)
            nodes = retriever.retrieve(query)
            context = "\n".join([n.text for n in nodes]) if nodes else ""
            print(f"📄 检索到 {len(nodes)} 个相关片段")

            # 构建 prompt
            if task_type == "daily_exercise":
                prompt = f"""你是一名公考出题专家。请生成一套行测每日一练（5道选择题），涵盖不同模块（言语、数量、判断、资料、常识），并附上答案和解析。
    
                要求：
                1. 题目用中文输出，不要用任何代码或 JSON 格式。
                2. 每道题格式为：【第1题】...
                3. 最后附上【答案】和【解析】。
                【对话历史】
                {history_text if history_text else "无"}
                资料参考（可选）：
                {context}
                【当前需求】
                {query}
                请直接输出题目内容，不要输出其他无关信息。
                """
            elif task_type == "hot_topic":
                prompt = f"""你是一名公考申论热点分析师。请提供3个当前适合公考的热点话题，每个话题包括：
                - 话题名称
                - 背景简述
                - 可能考察角度（如政策、民生、科技等）
                - 适用主题（如乡村振兴、科技创新等）
                 【对话历史】
                {history_text if history_text else "无"}
                参考资料（如有）：
                {context}
                【当前需求】
                {query}
                """
            elif task_type == "golden_sentence":
                prompt = f"""你是一名公考素材整理师。请提供10条申论金句，涵盖以下主题：民生、法治、创新、担当、奋斗、廉洁、改革、生态、文化、梦想。每条金句注明出处或适用主题。
                
                 【对话历史】
                {history_text if history_text else "无"}
                参考资料（如有）：
                {context}
                【当前需求】
                {query}
                """
            elif task_type == "weekly_plan":
                prompt = f"""你是一名公考备考规划专家。请制定一份一周备考计划，包含：
                - 每日学习内容（行测各模块 + 申论）
                - 每日题量推荐
                - 周末复盘建议
                
                 【对话历史】
                {history_text if history_text else "无"}
                参考资料（如有）：
                {context}
                【当前需求】
                {query}
                """
            else:
                prompt = f"""你是一名公考内容运营专家。请根据用户需求生成相关备考素材。
    
                【对话历史】
                {history_text if history_text else "无"}
                参考资料（如有）：
                {context}
                【当前需求】
                {query}
                
                请提供高质量、可直接使用的学习内容。
                """

            llm = get_llm()
            print("🤖 正在生成内容...")
            t0 = time.time()
            response = llm.invoke(prompt)
            t1 = time.time()
            print(f"⏱️ 内容生成耗时: {t1 - t0:.2f}s")

            # ===== 兼容多种返回格式提取文本 =====
            # ===== 提取纯文本内容 =====
            content_parts = response.content
            content = ""

            if isinstance(content_parts, list):
                for block in content_parts:
                    if isinstance(block, dict):
                        # 只取 type 为 'text' 的块
                        if block.get('type') == 'text' and 'text' in block:
                            content += block['text'] + "\n"
                    elif hasattr(block, 'type') and block.type == 'text':
                        content += block.text + "\n"
                    elif hasattr(block, 'text'):
                        content += block.text + "\n"
                    else:
                        content += str(block) + "\n"
            else:
                content = str(content_parts)

            content = content.strip()
            if not content:
                content = "（未生成有效内容，请重试）"

            print(f"📄 生成内容长度: {len(content)} 字符")
            print(f"📄 内容预览: {content[:200]}...")

            state["final_answer"] = content
            state["intermediate"]["ops"] = {
                "task_type": task_type,
                "generated_content": content
            }
            state["retrieved_docs"] = [n.text for n in nodes] if nodes else []
            print("✅ 内容生成成功")
            return state

        except Exception as e:
            import traceback
            traceback.print_exc()
            state["final_answer"] = f"内容生成失败: {str(e)}"
            state["intermediate"]["ops"] = {}
            return state


    async def ops_agent_stream(state: AgentState):
        """ops_agent 的流式版本"""
        query = state["user_query"]
        session_id = state.get("session_id", "default")
        history = get_session(session_id) or []
        history_text = ""
        if history:
            recent = history[-6:]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        # 判断任务类型（复用 ops_agent 的逻辑）
        if "每日一练" in query or "练习题" in query:
            task_type = "daily_exercise"
        elif "热点" in query or "话题" in query:
            task_type = "hot_topic"
        elif "金句" in query or "素材" in query:
            task_type = "golden_sentence"
        elif "一周计划" in query:
            task_type = "weekly_plan"
        else:
            task_type = "general"

        # 检索相关知识
        retriever = get_retriever(top_k=3)
        nodes = retriever.retrieve(query)
        context = "\n".join([n.text for n in nodes]) if nodes else ""

        # 构建 prompt（与 ops_agent 相同）
        if task_type == "daily_exercise":
            prompt = f"""你是一名公考出题专家。请生成一套行测每日一练（5道选择题），涵盖不同模块（言语、数量、判断、资料、常识），并附上答案和解析。
    要求：
    1. 题目用中文输出，不要用任何代码或 JSON 格式。
    2. 每道题格式为：【第1题】...
    3. 最后附上【答案】和【解析】。
    资料参考（可选）：
    {context}
    请直接输出题目内容，不要输出其他无关信息。
    """
        elif task_type == "hot_topic":
            prompt = f"""你是一名公考申论热点分析师。请提供3个当前适合公考的热点话题...
    参考资料（如有）：{context}"""
        elif task_type == "golden_sentence":
            prompt = f"""你是一名公考素材整理师。请提供10条申论金句...
    参考素材（如有）：{context}"""
        elif task_type == "weekly_plan":
            prompt = f"""你是一名公考备考规划专家。请制定一份一周备考计划...
    用户信息：{query}
    参考计划模板：{context}"""
        else:
            prompt = f"""你是一名公考内容运营专家。请根据用户需求生成相关备考素材。
    用户需求：{query}
    请提供高质量、可直接使用的学习内容。"""

        # 流式调用 Claude
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",  # 运营任务用 Haiku 更快
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=True,
            max_tokens=4096
        )
        async for chunk in llm.astream(prompt):
            content = chunk.content
            # 处理多块内容，只提取 text
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        yield block.get('text', '')
                    elif hasattr(block, 'type') and block.type == 'text':
                        yield block.text
            else:
                yield str(content)