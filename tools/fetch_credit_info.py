from langchain_core.tools import tool
from nodes.user_info_fetch import user_info_fetch as user_info_fetch_node


@tool
def fetch_credit_info(user_id: str, query: str):
    """
    Fetch comprehensive user credit information using AI-powered classification.

    This tool intelligently determines what credit data to retrieve based on the query.
    It uses OpenAI to classify the query into relevant credit report categories, then
    fetches the corresponding data from the vector database.

    Supported categories include:
    - CREDIT_SCORE: Credit score values, factors, and scoring models
    - BORROWER: Personal details, addresses, date of birth
    - OPEN_CREDIT_LIABILITY: Open loans, late payments, collection accounts
    - CLOSED_CREDIT_LIABILITY: Closed accounts, charge-offs, collections
    - CREDIT_SUMMARY: Overall credit health, tradelines, utilization
    - HARD_CREDIT_INQUIRY: Credit inquiries and applications
    - And many more credit-related categories

    Args:
        user_id: The unique identifier for the user/borrower
        query: The user's natural language query about their credit information

    Returns:
        Dictionary containing classified categories and fetched credit data
    """
    state = {"user_id": user_id, "query": query, "context": {}}

    result_state = user_info_fetch_node(state)

    if not result_state.get("context") or not result_state["context"].get("data"):
        return {
            "error": "No credit information found for the given user_id and query",
            "categories": result_state.get("context", {}).get("categories", []),
        }

    return result_state["context"]
