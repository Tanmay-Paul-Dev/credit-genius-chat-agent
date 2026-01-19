# Pinecone Service
# This module provides the Pinecone vector store for the application

import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Config from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_DB_NAME", "credit-genius-production")
EMBEDDING_MODEL = "text-embedding-ada-002"
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "credit_reports")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Get the index
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Create vectorstore with LangChain integration
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    namespace=NAMESPACE,
    text_key="summary",
)
