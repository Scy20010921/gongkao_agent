import os
import time
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from core.job_matcher import match_jobs
from agents.state import AgentState

load_dotenv()


def get_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",  # 选岗推荐用 Haiku 即可，更快更便宜
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=60,
        max_retries=1
    )


def extract_user_info(query: str) -> dict:
    """从用户输入中提取关键信息"""
    info = {
        "major": "不限",
        "education": "本科及以上",
        "location": None,
        "is_应届": None,
        "experience": None
    }

    # 专业提取
    major_keywords = ["经济学", "财政学", "金融学", "会计学", "财务管理", "法学", "计算机", "统计学", "管理学", "中文",
                      "新闻"]
    for m in major_keywords:
        if m in query:
            info["major"] = m
            break

    # 学历提取
    if "硕士" in query or "研究生" in query:
        info["education"] = "硕士研究生及以上"
    elif "本科" in query or "大学" in query:
        info["education"] = "本科及以上"
    elif "大专" in query:
        info["education"] = "大专及以上"

    # 地点提取
    location_keywords = [
        "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "重庆", "西安", "长沙",
        "湖南", "湖北", "广东", "浙江", "四川", "江苏", "山东", "河南", "福建", "河北", "安徽",
        "辽宁", "江西", "广西", "云南", "贵州", "山西", "陕西", "甘肃", "海南", "内蒙古", "新疆",
        "西藏", "青海", "宁夏", "黑龙江", "吉林", "天津", "苏州", "宁波", "厦门", "青岛", "大连"
    ]
    for loc in location_keywords:
        if loc in query:
            info["location"] = loc
            break

    # 应届/往届
    if "应届" in query:
        info["is_应届"] = True
    elif "往届" in query or "社会" in query:
        info["is_应届"] = False

    # 提取工作经历
    if "基层" in query or "工作经历" in query:
        info["experience"] = "二年以上基层工作经历"
    elif "不限" in query or "无限制" in query:
        info["experience"] = "无限制"
    return info


def job_agent(state: AgentState) -> AgentState:
    query = state["user_query"]
    try:
        print(f"💼 [Job Agent] 收到选岗请求: {query}")

        # 1. 提取用户条件
        user_info = extract_user_info(query)
        print(f"📌 提取的用户条件: {user_info}")

        # 2. 匹配岗位
        matched_jobs = match_jobs(
            major=user_info.get("major", "不限"),
            education=user_info.get("education", "本科及以上"),
            location=user_info.get("location"),
            is_应届=user_info.get("is_应届"),
            experience=user_info.get("experience")
        )

        if not matched_jobs:
            state["final_answer"] = "未找到符合条件的岗位。建议放宽条件（如专业、地区）后重试。"
            state["intermediate"]["job"] = {"matched": []}
            return state

        print(f"📊 匹配到 {len(matched_jobs)} 个岗位")

        # 3. 调用 Claude 生成推荐说明
        jobs_text = "\n\n".join([
            f"【{j['部门名称']}】{j['招考职位']}\n"
            f"  地点：{j['工作地点']}\n"
            f"  学历要求：{j['学历']}\n"
            f"  专业要求：{j.get('专业', '不限')}\n"
            f"  招录人数：{j.get('招考人数', '未知')}\n"
            f"  基层经历要求：{j.get('基层工作最低年限', '不限')}\n"
            f"  面试比例：{j.get('面试人员比例', '未知')}\n"
            f"  职位简介：{j.get('职位简介', '')[:50]}...\n"
            f"  匹配度：{j.get('match_score', 0)}分\n"
            f"  匹配原因：{'、'.join(j.get('match_reasons', []))}"
            for j in matched_jobs
        ])

        prompt = f"""你是一名公考选岗专家。请根据以下岗位匹配结果，为用户提供选岗建议。

【用户条件】
专业：{user_info['major']}
学历：{user_info['education']}
地点：{user_info.get('location', '不限')}
身份：{'应届生' if user_info.get('is_应届') else '往届生' if user_info.get('is_应届') is False else '不限'}

【匹配岗位】（按匹配度排序）
{jobs_text}

请输出：
### 推荐岗位（Top 3）
### 选岗建议（结合报录比、进面分、发展前景）
### 备选方向
### 下一步行动建议
"""
        llm = get_llm()
        print("🤖 正在生成选岗建议...")
        t0 = time.time()
        response = llm.invoke(prompt)
        t1 = time.time()
        print(f"⏱️ 选岗建议生成耗时: {t1 - t0:.2f}s")

        state["final_answer"] = response.content
        state["intermediate"]["job"] = {
            "user_info": user_info,
            "matched_jobs": matched_jobs,
            "recommendation": response.content
        }
        state["retrieved_docs"] = [jobs_text]
        print("✅ 选岗推荐完成")
        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["final_answer"] = f"选岗失败: {str(e)}"
        state["intermediate"]["job"] = {}
        return state