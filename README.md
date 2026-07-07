# 公考多智能体助手 (Gongkao Agent)

基于 LangGraph + Chroma + LlamaIndex + Claude 的公考智能问答与出题系统。

## 功能
- 知识库检索（RAG）
- 智能答疑（基于文档）
- 自动出题（选择题）

## 技术栈
- LangGraph (多 Agent 编排)
- Chroma (向量数据库)
- LlamaIndex (文档索引)
- Anthropic Claude (LLM)
- FastAPI (API 服务)

## 快速开始
1. 克隆项目
2. 创建虚拟环境：`python -m venv .venv`
3. 安装依赖：`pip install -r requirements.txt`
4. 复制 `.env.example` 为 `.env` 并填入你的 API Key
5. 运行索引：`python ingestion.py`
6. 启动服务：`uvicorn api:app --reload`

## API 示例