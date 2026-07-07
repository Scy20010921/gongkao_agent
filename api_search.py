from fastapi import FastAPI
from pydantic import BaseModel
from core.retriever import get_retriever

app = FastAPI(title="公考知识检索 API (无LLM)")

class QueryRequest(BaseModel):
    question: str

@app.post("/search")
def search(request: QueryRequest):
    retriever = get_retriever(top_k=3)
    nodes = retriever.retrieve(request.question)
    return {
        "results": [
            {"text": node.text, "score": node.score}
            for node in nodes
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}