from langchain_core.tools import tool
from retriever import MMRRetriever
from services.qdrant_service import vectorstore
from langchain_core.runnables import RunnableConfig

retriever = MMRRetriever(
    vectorstore=vectorstore,
    k=5,
    fetch_k=20,
    lambda_mult=0.6,
)


@tool
def retriever_tool(
    query: str,
    config: RunnableConfig,
    required_info: list = None,
    optional_info: list = None,
) -> str:
    """
    Retrieves user-specific personal and credit-related information from the internal knowledge base
    using semantic search.

    This tool is used to fetch data such as:
    - Credit score and credit history
    - Loan records and repayment behavior
    - Personal details (e.g., name, location)
    - Other financial or profile-related attributes linked to the user

    The search is strictly filtered by user_id, and only documents belonging to the
    specified user can be returned.

    Args:
        query (str): Natural language query describing the information to retrieve.
        user_id (str): Unique identifier of the user whose data should be searched.
        required_info (list, optional): List of required information fields to search for
                                       (e.g., ["credit_score", "location"]).
        optional_info (list, optional): List of optional information fields that enhance results.

    Returns:
        str: Retrieved user-specific information as a concatenated string.
             If no data is found, returns a clear "no documents found" message.
    """
    # Combine required_info and optional_info into searchdata
    searchdata = []
    if required_info:
        searchdata.extend(required_info)
    if optional_info:
        searchdata.extend(optional_info)

    # Build enhanced query from searchdata
    enhanced_query = query
    if searchdata:
        search_terms = " ".join(searchdata)
        enhanced_query = f"{query} {search_terms}"

    # Log the search details for debugging
    print(f"[Retriever] Original query: {query}")
    if searchdata:
        print(f"[Retriever] Search data: {searchdata}")
        print(f"[Retriever] Enhanced query: {enhanced_query}")

    docs = retriever.retrieve(
        enhanced_query.strip(), user_id=config["configurable"]["user_id"]
    )

    if not docs:
        return "No relevant documents found for this user."

    print("[Trigerring]::[Tool]: retriever_tool")
    return "\n\n".join(
        f"[SOURCE {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )
