import os
import pandas as pd
from typing import List, Dict, Any

# 岗位数据路径（根据实际文件格式修改）
JOB_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/中央机关及其直属机构2026年度考试录用公务员招考简章.xls")
# 如果是 CSV：JOB_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/positions.csv")

_jobs_cache = None


def load_jobs() -> List[Dict[str, Any]]:
    """加载岗位数据（带缓存）"""
    global _jobs_cache
    if _jobs_cache is not None:
        return _jobs_cache

    if not os.path.exists(JOB_DATA_PATH):
        print(f"⚠️ 岗位数据文件不存在: {JOB_DATA_PATH}")
        return []

    try:
        if JOB_DATA_PATH.endswith('.csv'):
            df = pd.read_csv(JOB_DATA_PATH, encoding='utf-8')
        elif JOB_DATA_PATH.endswith('.xlsx'):
            df = pd.read_excel(JOB_DATA_PATH, engine='openpyxl', header=1)  # 使用第2行作为列名
        elif JOB_DATA_PATH.endswith('.xls'):
            df = pd.read_excel(JOB_DATA_PATH, engine='xlrd', header=1)  # 使用第2行作为列名
        else:
            print(f"⚠️ 不支持的文件格式: {JOB_DATA_PATH}")
            return []


        # 将所有 NaN（空值）替换为空字符串，避免 JSON 序列化错误
        df = df.fillna('')
        jobs = df.to_dict(orient='records')
        _jobs_cache = jobs
        print(f"✅ 加载了 {len(jobs)} 个岗位，列名：{df.columns.tolist()}")
        return jobs
    except Exception as e:
        print(f"❌ 加载岗位数据失败: {e}")
        return []


def match_jobs(
        major: str = None,
        education: str = None,
        location: str = None,
        is_应届: bool = None,
        experience: str = None,
        top_k: int = 5
) -> List[Dict[str, Any]]:
    jobs = load_jobs()
    if not jobs:
        return []

    matched = []
    for job in jobs:
        score = 0
        reasons = []

        # 1. 专业匹配
        if major and major != "不限":
            job_majors = str(job.get('专业', '')).replace('，', ',').split(',')
            job_majors = [m.strip() for m in job_majors if m.strip()]
            if any(major in m for m in job_majors):
                score += 30
                reasons.append("专业匹配")
            else:
                score += 5
                reasons.append("专业不完全匹配")
        else:
            score += 10

        # 2. 学历匹配
        if education:
            if education in str(job.get('学历', '')):
                score += 25
                reasons.append("学历符合")
            else:
                score += 5

        # 3. 地点匹配
        if location:
            # 如果用户指定了地点，只匹配工作地点包含该地点的岗位
            job_location = str(job.get('工作地点', ''))
            if location in job_location:
                score += 30  # 提高地点匹配的权重
                reasons.append("地点匹配")
            else:
                # 地点不匹配，大幅降低分数
                score -= 20
                reasons.append("地点不匹配")
        else:
            score += 5

        # 4. 应届身份匹配（通过基层经历判断，或专门字段）
        # 注意：你的数据没有明确的"应届"列，可以通过"基层工作最低年限"判断
        if is_应届 is not None:
            job_exp = str(job.get('基层工作最低年限', ''))
            if is_应届 and '无限制' in job_exp:
                score += 15
                reasons.append("应届生可报")
            elif not is_应届 and '无限制' not in job_exp:
                score += 15
                reasons.append("往届生可报")

        # 5. 基层经历匹配
        if experience:
            job_exp = str(job.get('基层工作最低年限', '不限'))
            if '不限' in job_exp or experience in job_exp:
                score += 10
                reasons.append("经历符合")

        matched.append({
            **job,
            "match_score": score,
            "match_reasons": reasons
        })

    matched.sort(key=lambda x: x["match_score"], reverse=True)
    return matched[:top_k]