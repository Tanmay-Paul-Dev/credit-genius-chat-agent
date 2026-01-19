from langchain_core.tools import tool


@tool
def select_intent(intent: str):
    """
    Select the most appropriate intent for the user's query.

    intent must be one of:
    CREDIT_OVERVIEW
    CREDIT_SCORE
    ACCOUNT_DETAILS
    NEGATIVE_CREDIT_EVENTS
    CREDIT_INQUIRIES
    PERSONAL_PROFILE
    BUREAU_STATUS
    """
    return intent
