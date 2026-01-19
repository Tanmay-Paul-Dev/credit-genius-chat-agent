# retriever.py
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue


class MMRRetriever:
    def __init__(
        self,
        vectorstore: VectorStore,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        user_id: str | None = None,
    ):
        """
        lambda_mult:
          0.0 → max diversity
          1.0 → max relevance
        user_id: Filter results by user_id field
        """
        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.user_id = user_id

    def retrieve(self, query: str, user_id: str | None = None) -> list[Document]:
        """
        Retrieve documents with optional user_id filtering.

        Args:
            query: Search query string
            user_id: Optional user_id to filter results (overrides instance user_id)
        """
        # Use provided user_id or fall back to instance user_id
        filter_user_id = user_id or self.user_id

        # Build metadata filter for user_id
        metadata_filter = None
        if filter_user_id:
            metadata_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id", match=MatchValue(value=filter_user_id)
                    )
                ]
            )

        # Create retriever with filter
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult,
                "filter": metadata_filter,
            },
        )

        return retriever.invoke(query)
