from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.messages import SystemMessage
from services.opanai_service import small_model, large_model
from utils.prompt_loader import load_prompt
from langchain_core.runnables import RunnableConfig
import asyncio
import uuid


async def save_response_memory_background(
    user_id: str,
    store: BaseStore,
    retrieved_data: dict = None,
):
    """
    Use LLM to determine which retrieved data is important enough to store in memory for future use.
    Only stores key user information that will be needed in future conversations.
    """
    try:
        if not retrieved_data:
            print("[Memory] No retrieved data to save")
            return

        ns = ("user", user_id, "details")

        # Get existing memories to check for duplicates
        items = list(store.search(ns))
        existing_memories = (
            "\n".join(f"- {it.value.get('data', '')}" for it in items)
            if items
            else "(empty)"
        )

        # Get required_info and optional_info from retrieved_data
        required_info = retrieved_data.get("required_info", {}) or {}
        optional_info = retrieved_data.get("optional_info", {}) or {}

        # Combine all data for LLM analysis
        all_data = {**required_info, **optional_info}
        
        # Filter out None values
        all_data = {k: v for k, v in all_data.items() if v is not None and v != "" and v != "null"}

        if not all_data:
            print("[Memory] No valid data to analyze")
            return

        # Ask LLM to determine what's important to store
        prompt = f"""Analyze the following user data and determine which items are IMPORTANT to remember for future conversations.

        RETRIEVED USER DATA:
        {all_data}

        ALREADY STORED IN MEMORY:
        {existing_memories}

        TASK:
        Select the data that is IMPORTANT for personalization and future financial decisions:
        1. User's name (ALWAYS store if available)
        2. User's location/city (important for loan eligibility)
        3. Credit scores
        4. Income and employment status
        6. Any other personal factual data about the user

        DO NOT store:
        - Data that's already stored in memory
        - Generic descriptions or long summaries
        - Temporary or transient information
        - Duplicate information (even if worded differently)
        - Future Goals like (loan amount/purpose)

        For each important item, return it as a short sentence.
        Return ONLY a JSON array of strings, each being a memory to store.
        If nothing NEW is important enough to store, return an empty array [].

        Example output:
        ["User's name is Henry Vander", "User is located in Charlotte, NC", "User's credit score is 716", "User's income is $5000 per month"]

        Return ONLY the JSON array, nothing else."""

        response = await small_model.ainvoke([SystemMessage(content=prompt)])
        
        # Parse the response
        import json
        try:
            # Clean up the response - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            memories_to_store = json.loads(content)
        except json.JSONDecodeError:
            print(f"[Memory] Could not parse LLM response: {response.content}")
            return

        if not memories_to_store or not isinstance(memories_to_store, list):
            print("[Memory] LLM determined no important data to store")
            return

        # Store the important memories
        stored_count = 0
        existing_lower = set(m.lower() for m in existing_memories.split("\n") if m.strip())
        
        for memory_text in memories_to_store:
            if memory_text and isinstance(memory_text, str):
                # Check if already stored
                if memory_text.lower() not in existing_lower:
                    store.put(ns, str(uuid.uuid4()), {"data": memory_text})
                    print(f"[Memory] Stored: {memory_text}")
                    stored_count += 1
                else:
                    print(f"[Memory] Skipped (duplicate): {memory_text}")

        if stored_count == 0:
            print("[Memory] No new important data to store")
        else:
            print(f"[Memory] Total stored: {stored_count} items")

    except Exception as e:
        print(f"[Memory] Error: {e}")


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

    # Invoke the chat agent with structured input
    response = await large_model.ainvoke(
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
        user_id=user_id,
        store=store,
        retrieved_data=state.get("retrieved_data"),
    )

    return {"final_answer": response.content, "messages": messages}
