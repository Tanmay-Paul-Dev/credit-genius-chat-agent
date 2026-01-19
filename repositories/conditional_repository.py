from states import State
from langgraph.graph import StateGraph, END


def route_after_intent_classification(state: State) -> str:
    """Route based on classified query_type - routes to retriever_node for finance queries"""
    if state.get("error"):
        return "error_node"

    intent_obj = state.get("intent", {})
    query_type = (
        intent_obj.get("query_type", "") if isinstance(intent_obj, dict) else ""
    )

    if query_type == "finance":
        print(f"[Router] query_type=finance, routing to retriever_node")
        return "retriever_node"

    return END


def route_after_retriever(state: State) -> str:
    """Route after retriever node - always goes to rule_builder for finance queries"""
    if state.get("error"):
        return "error_node"

    return "finance_agent_node"


def route_after_rule_builder(state: State) -> str:
    if state.get("error"):
        return "error_node"
    return "finance_agent_node"


def route_after_finance_agent(state: State) -> str:
    if state.get("error"):
        return "error_node"

    missing_info = state["finance_agent"]["missing_info"]

    if missing_info and len(missing_info) > 0:
        print(f"[Router] missing_info found: {missing_info}, routing to chat_node")
        return "chat_node"

    return "chat_node"


def route_after_followup_agent(state: State) -> str:
    if state.get("error"):
        return "error_node"
    return END


def route_on_error(state: State) -> str:
    if state.get("error"):
        return "error_node"
    return "next"


def route_after_memory_retrieval(state: State) -> str:
    if state.get("error"):
        return "error_node"

    if state.get("found_in_memory") == True:
        return "chat_node"

    if state.get("found_in_memory") == False:
        return "intent_classifier"

    return "intent_classifier"
