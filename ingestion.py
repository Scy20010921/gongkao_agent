import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.dashscope import DashScopeEmbedding
import chromadb

load_dotenv()

# ===== 读取配置 =====
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "text-embedding-v4")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gongkao_knowledge")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

DATA_DIR = "./data"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

# ===== 设置嵌入模型 =====
print(f"🔹 使用百炼 Embedding 模型: {BAILIAN_MODEL}")
Settings.embed_model = DashScopeEmbedding(
    model_name=BAILIAN_MODEL,
    api_key=BAILIAN_API_KEY,
    embed_batch_size=10,
)

# ===== 连接本地 Chroma =====
print(f"🔄 连接本地 Chroma (路径: {CHROMA_PERSIST_DIR})...")
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
# 如果集合已存在，先删除确保干净
try:
    client.delete_collection(COLLECTION_NAME)
    print("已删除旧集合")
except ValueError:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)
print("✅ Chroma 连接成功")
print(f"📊 集合中现有文档数量: {collection.count()}")

# ===== 加载文档 =====
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"⚠️ 请将公考资料放入 {DATA_DIR} 目录后重新运行")
    exit(0)

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"📄 加载了 {len(documents)} 个文档片段")

# ===== 分块 =====
parser = SimpleNodeParser.from_defaults(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
nodes = parser.get_nodes_from_documents(documents)
print(f"✂️ 分割为 {len(nodes)} 个文本块")

# 在加载文档后，添加过滤
filtered_nodes = []
for node in nodes:
    text = node.text
    # 过滤包含广告关键词的节点
    if "认准淘宝店铺" in text or "通关达人" in text or "共" in text and "页" in text:
        continue
    filtered_nodes.append(node)
nodes = filtered_nodes
print(f"✂️ 过滤后剩余 {len(nodes)} 个有效文本块")

# ===== 手动生成向量并写入 Chroma =====
print("📊 正在生成向量并写入 Chroma（可能需要几分钟）...")
embeddings = Settings.embed_model.get_text_embedding_batch(
    [node.text for node in nodes],
    show_progress=True
)
ids = [node.node_id for node in nodes]
metadatas = [node.metadata for node in nodes]
documents_text = [node.text for node in nodes]

# 批量添加到集合
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents_text,
)
print(f"✅ 手动添加了 {len(ids)} 个向量到 Chroma")

# ===== 验证写入 =====
print(f"📊 写入后集合中的文档数量: {collection.count()}")

# ===== 从已填充的集合构建索引 =====
print("🔄 从 Chroma 加载索引...")
index = VectorStoreIndex.from_vector_store(vector_store)
print("✅ 索引加载成功")

# ===== 快速测试检索 =====
print("🔍 测试检索...")
retriever = index.as_retriever(similarity_top_k=2)
sample_query = "公务员考试报名条件"
results = retriever.retrieve(sample_query)
if results:
    print("✅ 检索成功，示例结果：")
    for node in results:
        print(f"  - {node.text[:100]}...")
else:
    print("⚠️ 检索无结果，请检查数据内容")

print(f"\n✅ 所有步骤完成！数据已持久化到本地目录: {CHROMA_PERSIST_DIR}")