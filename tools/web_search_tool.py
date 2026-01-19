from langchain_community.tools import DuckDuckGoSearchRun

web_search_tool = DuckDuckGoSearchRun(
    description="""
    Searches the public internet using DuckDuckGo to retrieve up-to-date,
    publicly available information.

    This tool MAY use user context (such as location, city, country,
    or generic profile attributes) ONLY to refine search queries
    and improve relevance of public results.

    Appropriate use cases:
    - Finding location-specific loan options, banks, or NBFCs
    - Searching regional interest rates, government schemes, or regulations
    - Retrieving lender policies applicable to the user’s location
    - General financial knowledge when internal data is insufficient

    Restrictions:
    - MUST NOT be used to retrieve or infer user-specific private data
      such as credit score, credit history, identity details, or accounts
    - MUST NOT fabricate or guess missing user information
    - MUST NOT replace internal user data retrieval

    This tool is supplementary and should be used only when
    internal knowledge does not fully answer the query.
    """
)
