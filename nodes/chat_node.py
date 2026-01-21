from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.messages import SystemMessage
from services.opanai_service import initialize_model
from utils.prompt_loader import load_prompt
from langchain_core.runnables import RunnableConfig
import asyncio
import uuid


async def save_response_memory_background(
    query: str, response_content: str, user_id: str, store: BaseStore
):
    """Background task to analyze chat response and save relevant data to memory."""
    try:
        llm = initialize_model()
        ns = ("user", user_id, "details")

        # Get existing memories to avoid duplicates
        items = list(store.search(ns))
        existing = (
            "\n".join(f"- {it.value.get('data', '')}" for it in items)
            if items
            else "(empty)"
        )

        # Prompt to determine if response contains storable info
        extract_prompt = f"""Analyze this conversation and determine if there's any important user-specific information worth remembering.

        USER QUERY: {query}

        ASSISTANT RESPONSE: {response_content[:2000]}

        ALREADY STORED:
        {existing}

        If there's NEW factual information about the user (credit scores, account balances, loan amounts, dates, preferences, etc.) that is NOT already stored, extract it as a single short sentence.

            Rules:
            - Return "NONE" if no new information to store
            - Return "NONE" if the info is already stored
            - Otherwise return ONLY a short sentence like:
            "User has credit scores of 802 and 553"
            "User's loan balance is $50,000"
            "User prefers monthly payments"

Keep it under 100 characters. Return ONLY the sentence or "NONE"."""

        decision = await llm.ainvoke([SystemMessage(content=extract_prompt)])

        memory_text = decision.content.strip().strip('"')

        print(f"[Memory Background] LLM decision: '{memory_text}'")

        if memory_text and memory_text.upper() != "NONE" and len(memory_text) < 200:
            store.put(ns, str(uuid.uuid4()), {"data": memory_text})
            print(f"[Memory Background] Stored: {memory_text}")
        else:
            print(f"[Memory Background] Skipped - no new info to store")

    except Exception as e:
        print(f"[Memory Background] Error: {e}")


async def chat_node(state: State, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    query_type = state["intent"]["query_type"]
    user_query = state["query"]
    finance_agent_answer = state.get("finance_agent", {}).get("answer")
    required_info = state.get("retrieved_data", {}).get("required_info")
    optional_info = state.get("retrieved_data", {}).get("optional_info")
    evidence = state.get("retrieved_data", {}).get("retrived_intent_info")
    memories = (
        "\n".join(f"- {it}" for it in state.get("memory_context", []))
        if state.get("memory_context")
        else "(empty)"
    )
    recent_cenversations = state.get("messages", [])[-10:]

    # Build context message for the chat agent
    context_message = f"""
        query_type: {query_type}
        user_query: {user_query}
        finance_agent_response: {finance_agent_answer}
        retrieved_data:
        required_info: {required_info}
        optional_info: {optional_info}
        evidence: {evidence}
        messages: {recent_cenversations}
        memory: {memories}
    """

    print(context_message)

    # Invoke the chat agent with structured input
    chat_llm = initialize_model()
    response = await chat_llm.ainvoke(
        [
            SystemMessage(content=load_prompt("chat_prompt")),
            {"role": "user", "content": context_message},
        ]
    )

    # Add assistant message to short-term memory (user message already added at graph invoke)
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": response.content})
    messages = messages[-10:]  # Keep only last 10 messages

    # 🔥 Save memory (await to ensure it completes)
    print("[Memory] Starting memory save...")
    await save_response_memory_background(
        state["query"], response.content, user_id, store
    )

    return {"final_answer": response.content, "messages": messages}
