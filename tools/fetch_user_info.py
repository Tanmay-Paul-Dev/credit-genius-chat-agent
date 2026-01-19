from langchain_core.tools import tool
from repositories.user_repository import fetch_borrower_info_from_vector_db


@tool
def fetch_user_info(user_id: str):
    """
    Fetch borrower information from the vector database.

    Use this tool to retrieve comprehensive borrower profile information including:
    - Personal details (name, contact information)
    - Address history
    - Employment information
    - Identification details

    Args:
        user_id: The unique identifier for the borrower

    Returns:
        List of borrower information records from the vector database
    """
    results = fetch_borrower_info_from_vector_db(user_id)

    if not results:
        return {"error": "No borrower information found for the given user_id"}

    # Extract payloads from results
    borrower_data = [point.payload for point in results]
    return borrower_data
