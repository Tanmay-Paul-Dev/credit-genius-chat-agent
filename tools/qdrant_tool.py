from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "credit_data"
EMBEDDING_MODEL = "text-embedding-3-small"

client = QdrantClient(url=QDRANT_URL)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# -----------------------------
# Tool Factory
# -----------------------------
def create_qdrant_search_tool(user_id: str):
    """
    Factory function to create a qdrant_search tool bound to a specific user_id.

    Args:
        user_id (str): The UUID of the user whose data to search

    Returns:
        A tool instance configured for the specific user
    """

    @tool(
        "qdrant_search",
        description=(
            "Search and retrieve user-specific credit information from the Qdrant vector database. "
            "This tool searches through borrower profiles, credit account details, payment history, "
            "credit inquiries, public records, and financial summaries. Use this tool when you need to: "
            "1) Get detailed information about the user's credit profile "
            "2) Retrieve specific credit account information (loans, credit cards, mortgages) "
            "3) Find payment history and credit behavior patterns "
            "4) Access credit score factors and financial summaries "
            "5) Look up borrower personal details and address history. "
            "The user_id is automatically provided - you only need to specify the query."
        ),
    )
    def qdrant_search(
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Retrieve relevant credit documents from Qdrant vector database for the current user.

        Args:
            query (str): Natural language query describing what credit information to retrieve
            limit (int): Maximum number of results to return (default: 5)

        Returns:
            str: Formatted string containing relevant credit information or "No relevant data found."
        """
        logger.info(
            f"🔍 QDRANT_SEARCH tool called with query='{query}', user_id={user_id}, limit={limit}"
        )

        # 1️⃣ Embed the query
        vector = embeddings.embed_query(query)

        # 2️⃣ Filter by user (using the bound user_id)
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

        # 3️⃣ Search Qdrant
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

        logger.info(
            f"✅ QDRANT_SEARCH returned {len(results)} results for user_id={user_id}"
        )

        # 4️⃣ Format response for LLM
        if not results:
            return "No relevant data found."

        formatted = []
        for r in results:
            payload = r.payload
            summary = payload.get("summary", "")
            category = payload.get("category", "")
            topics = payload.get("topics", "")
            formatted.append(f"[{category}] {summary}\nTopics: {topics}")

        return "\n\n---\n\n".join(formatted)

    return qdrant_search
