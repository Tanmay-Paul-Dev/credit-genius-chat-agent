from services.opanai_service import small_model
from langgraph.graph import StateGraph, START, END, MessagesState
import uuid
from langchain_core.messages import SystemMessage
from utils.prompt_loader import load_prompt
from langgraph.store.base import BaseStore
from states import State, MemoryDecision, MemoryLookupResult
from langchain_core.runnables import RunnableConfig


MEMORY_LOOKUP_PROMPT = """You are a memory lookup assistant.

The user has stored the following information about themselves:
{user_details_content}

The user is now asking: "{query}"

Determine if you can answer the user's question DIRECTLY from the stored memories.
- If yes, set found_in_memory=true and extract the relevant memory as the answer.
- If no (the question requires external data or is not about stored info), set found_in_memory=false.
"""


async def memory_node(state: State, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Get existing memories
    items = list(store.search(ns))
    existing = (
        "\n".join(f"- {it.value.get('data', '')}" for it in items)
        if items
        else "(empty)"
    )

    print(f"[Memory] Existing memories:\n{existing}")

    query = state["query"]

    # Ensure messages list exists
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    # 1️⃣ Check if query can be answered from memory
    if items:  # Only check if there are memories
        memory_lookup = small_model.with_structured_output(MemoryLookupResult)
        lookup_result: MemoryLookupResult = await memory_lookup.ainvoke(
            [
                SystemMessage(
                    content=MEMORY_LOOKUP_PROMPT.format(
                        user_details_content=existing, query=query
                    )
                ),
            ]
        )

        print(f"[Memory] Lookup: found_in_memory={lookup_result.found_in_memory}")

        if lookup_result.found_in_memory and lookup_result.memories:
            print(f"[Memory] Retrieved: {lookup_result.memories}")
            return {
                "messages": state["messages"],
                "found_in_memory": True,
                "memory_context": state["memory_context"],
            }

    # 2️⃣ Check if we should store new memories
    memory_extractor = small_model.with_structured_output(MemoryDecision)
    decision: MemoryDecision = await memory_extractor.ainvoke(
        [
            SystemMessage(
                content=load_prompt("memory_prompt").format(
                    user_details_content=existing
                )
            ),
            {"role": "user", "content": query},
        ]
    )

    print(f"[Memory] Store decision: should_write={decision.should_write}")

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
                print(f"[Memory] Stored: {mem.text.strip()}")

    return {
        "messages": state["messages"],
        "found_in_memory": False,
        "memory_context": state["memory_context"],
    }
