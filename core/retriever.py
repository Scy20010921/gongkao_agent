import os
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()

# ===== 设置嵌入模型（百炼）=====
Settings.embed_model = DashScopeEmbedding(
    model_name=os.getenv("BAILIAN_MODEL", "text-embedding-v4"),
    api_key=os.getenv("BAILIAN_API_KEY"),
    embed_batch_size=10,
)

_retriever = None
@traceable(run_type="retriever", name="get_retriever")
def get_retriever(top_k: int = 5):
    global _retriever
    if _retriever is None:
        print("🔄 首次加载 Chroma 索引...")
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("gongkao_knowledge")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        _retriever = index.as_retriever(similarity_top_k=top_k)
        print("✅ 索引加载完成")
    return _retriever