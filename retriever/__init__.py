# retriever.py
# MMR Retriever for Pinecone vector store
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from datetime import datetime, timezone

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
        Retrieve documents with Recency Bias (Time Decay).
        Relevant recent docs > Relevant old docs > Irrelevant docs.
        """
        filter_user_id = user_id or self.user_id
        
        metadata_filter = None
        if filter_user_id:
            metadata_filter = {"userId": {"$eq": filter_user_id}}

        # 1. Fetch more docs than needed (fetch_k) to allow re-ranking
        # We use similarity_search_with_score because we need the base score to modify
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.k * 3, # Fetch 3x pool to find recent gems
            filter=metadata_filter
        )

        # 2. Apply Time Decay to Scores
        scored_results = []
        now = datetime.now(timezone.utc)
        decay_rate = 0.01  # Adjustable: Higher = stricter penalty for old docs

        for doc, score in docs_with_scores:
            # Extract timestamp (Default to 'now' if missing so we don't punish docs with no date)
            # Ensure your metadata date format matches! (Here assuming ISO format or unix timestamp)
            ts_value = doc.metadata.get('timestamp')            
            if ts_value:
                # Parse timestamp depending on your format
                # Example for ISO string: doc_date = datetime.fromisoformat(ts_value)
                # Example for Unix float: 
                doc_date = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                
                # Calculate age in days
                days_old = (now - doc_date).days
                days_old = max(0, days_old) # distinct possibility of negative if clocks skew
            else:
                days_old = 0

            # Apply Formula: Score * (1 / (1 + rate * age))
            time_factor = 1 / (1 + (decay_rate * days_old))
            final_score = score * time_factor
            
            scored_results.append((doc, final_score))

        # 3. Sort by New Adjusted Score (Highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 4. Return top K documents (strip scores to match return type)
        return [doc for doc, score in scored_results[:self.k]]