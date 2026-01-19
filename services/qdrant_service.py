from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
import os

QDRANT_URL = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "credit_data")
EMBEDDING_MODEL = "text-embedding-3-small"

qdrant_client = QdrantClient(url=QDRANT_URL)


vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    content_payload_key="summary",  # 🔥 THIS IS THE FIX
)
