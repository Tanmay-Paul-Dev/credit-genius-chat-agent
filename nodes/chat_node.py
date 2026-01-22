from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.messages import SystemMessage
from services.opanai_service import large_model
from utils.prompt_loader import load_prompt
from langchain_core.runnables import RunnableConfig
import asyncio
import uuid

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
    # print("[Memory] Starting memory save...")
    # await save_response_memory_background(
    #     user_id=user_id,
    #     store=store,
    #     retrieved_data=state.get("retrieved_data"),
    # )

    return {"final_answer": response.content, "messages": messages}
