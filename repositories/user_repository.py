from services.qdrant_service import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os


def fetch_borrower_info_from_vector_db(
    user_id: str,
):
    query_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            ),
            FieldCondition(
                key="category",
                match=MatchValue(value="BORROWER"),
            ),
        ]
    )
    results = qdrant_client.query_points(
        collection_name=os.getenv("VECTOR_DB_COLLECTION_NAME"),
        query_filter=query_filter,
        limit=5,
        with_payload=True,
    ).points
    return results
