import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
import chromadb
from test_data import TEST_QUERIES

load_dotenv()

# ---------- 配置 ----------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gongkao_knowledge")
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "text-embedding-v4")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
TOP_K = 5  # 检索前5个结果

# ---------- 设置嵌入模型（与索引一致） ----------
print("🔹 使用百炼 Embedding 模型")
Settings.embed_model = DashScopeEmbedding(
    model_name=BAILIAN_MODEL,
    api_key=BAILIAN_API_KEY,
    embed_batch_size=10,
)

# ---------- 连接本地 Chroma ----------
print(f"🔄 连接本地 Chroma (路径: {CHROMA_PERSIST_DIR})...")
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=TOP_K)
print(f"✅ 连接成功，集合中有 {collection.count()} 个文档\n")

# ---------- 评估函数 ----------
def evaluate_retrieval():
    total_queries = len(TEST_QUERIES)
    hits = 0
    total_expected = 0
    found_keywords = 0

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected_keywords"]
        total_expected += len(expected)

        # 执行检索
        results = retriever.retrieve(query)
        retrieved_texts = [r.text for r in results]
        combined_text = " ".join(retrieved_texts).lower()

        # 统计命中的关键词
        hit_count = 0
        missing_keywords = []
        for kw in expected:
            if kw.lower() in combined_text:
                hit_count += 1
            else:
                missing_keywords.append(kw)

        # 如果所有关键词都命中，算一次命中
        if hit_count == len(expected):
            hits += 1

        found_keywords += hit_count

        # 打印每个查询的结果
        print(f"📝 查询: {query}")
        print(f"   ✅ 命中关键词: {hit_count}/{len(expected)}")
        if missing_keywords:
            print(f"   ⚠️ 未命中: {', '.join(missing_keywords)}")
        print(f"   📄 检索到的片段数: {len(results)}")
        if results:
            print(f"   📌 最相关片段: {results[0].text[:100]}...")
        else:
            print("   ❌ 无结果")
        print("-" * 60)

    # 计算指标
    hit_rate = hits / total_queries if total_queries > 0 else 0
    recall = found_keywords / total_expected if total_expected > 0 else 0

    print(f"\n📊 评估结果:")
    print(f"   - 命中率 (Hit Rate): {hit_rate:.2%} ({hits}/{total_queries})")
    print(f"   - 召回率 (Recall): {recall:.2%} ({found_keywords}/{total_expected})")
    print(f"   - 平均每个查询命中的关键词: {found_keywords / total_queries:.2f}")

if __name__ == "__main__":
    evaluate_retrieval()