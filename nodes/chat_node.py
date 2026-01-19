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
    chat_llm = initialize_model()
    user_id = config["configurable"]["user_id"]

    ns = ("user", user_id, "details")

    # Get existing memories
    items = list(store.search(ns))
    existing = (
        "\n".join(f"- {it.value.get('data', '')}" for it in items)
        if items
        else "(empty)"
    )

    # Check if we need to ask user for specific info
    finance_agent_response = state.get("finance_agent", {})
    missing_info = (
        finance_agent_response.get("missing_info", [])
        if isinstance(finance_agent_response, dict)
        else []
    )

    if missing_info and len(missing_info) > 0:
        # We need to ask the user for these fields
        required_fields = ", ".join(missing_info)
        ask_user_prompt = f"""You need to ask the user for the following information to help answer their query.
        
        User's original query: {state["query"]}

        Required information from user: {required_fields}

        Generate a friendly, conversational response asking the user for this information. Be specific about what you need and why it will help answer their question.
        Do not make up any information - just ask for what's needed."""

        response = await chat_llm.ainvoke(
            [
                SystemMessage(
                    content="You are a helpful assistant. Ask the user for the required information in a friendly way."
                ),
                {"role": "user", "content": ask_user_prompt},
            ]
        )

        print(f"[Chat] Asking user for: {missing_info}")

        # Add assistant message to short-term memory (user message already added at graph invoke)
        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": response.content})
        messages = messages[-10:]  # Keep only last 10 messages

        print(f"[Chat] Updated messages: {len(messages)} messages")

        return {"final_answer": response.content, "messages": messages}

    # Normal chat flow - when we have finance agent response or just answering a query
    system_msg = SystemMessage(
        content=load_prompt("chat_prompt").format(
            user_details_content=existing or "(empty)",
            finance_agent_response=state.get("finance_agent", "(not available)"),
        )
    )

    # Build user prompt - only include finance_agent if available
    finance_agent = state.get("finance_agent")
    if finance_agent:
        chat_node_user_prompt = f"""
        FINANCE AGENT RESPONSE: {finance_agent}
        USER QUERY: {state["query"]}
        """
    else:
        chat_node_user_prompt = f"""
        USER QUERY: {state["query"]}
        """

    # Get response from LLM
    response = await chat_llm.ainvoke(
        [system_msg]
        + [
            {
                "role": "user",
                "content": chat_node_user_prompt,
            }
        ]
    )

    print(f"[Chat] Response: {response.content}")

    # Add assistant message to short-term memory (user message already added at graph invoke)
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": response.content})
    messages = messages[-10:]  # Keep only last 10 messages

    print(f"[Chat] Updated messages: {len(messages)} messages")

    # 🔥 Save memory (await to ensure it completes)
    print("[Memory] Starting memory save...")
    await save_response_memory_background(
        state["query"], response.content, user_id, store
    )

    return {"final_answer": response.content, "messages": messages}
