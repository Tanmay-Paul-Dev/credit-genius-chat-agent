from services.pinecone_service import pinecone_index
import os


def fetch_borrower_info_from_vector_db(
    user_id: str,
):
    """
    Fetch borrower info from Pinecone filtered by user_id and category.
    """
    # Query Pinecone with metadata filter
    results = pinecone_index.query(
        vector=[0.0] * 1536,  # Dummy vector for metadata-only query
        filter={
            "user_id": {"$eq": user_id},
            "category": {"$eq": "BORROWER"},
        },
        top_k=5,
        include_metadata=True,
    )
    return results.matches
