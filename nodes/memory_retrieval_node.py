from services.opanai_service import initialize_model
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore
from states import State, MemoryLookupResult
from langchain_core.runnables import RunnableConfig


MEMORY_LOOKUP_PROMPT = """You are a memory lookup assistant.

The user has stored the following information about themselves:
{user_details_content}

The user is now asking: "{query}"

Determine if you can answer the user's question DIRECTLY from the stored memories.
- If yes, set found_in_memory=true and extract ALL relevant memories as a list.
- If no (the question requires external data or is not about stored info), set found_in_memory=false.
"""


async def memory_retrieval_node(state: State, config: RunnableConfig, store: BaseStore):
    """
    Retrieves relevant memories from the store based on user query.
    Sets found_in_memory=True and memory_context if relevant memories exist.
    """
    model = initialize_model()

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Get existing memories
    items = list(store.search(ns))
    existing = (
        "\n".join(f"- {it.value.get('data', '')}" for it in items)
        if items
        else "(empty)"
    )

    print(f"[Memory Retrieval] Existing memories:\n{existing}")

    query = state["query"]

    # Ensure messages list exists
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    # Check if query can be answered from memory
    if items:
        memory_lookup = model.with_structured_output(MemoryLookupResult)
        lookup_result: MemoryLookupResult = await memory_lookup.ainvoke(
            [
                SystemMessage(
                    content=MEMORY_LOOKUP_PROMPT.format(
                        user_details_content=existing, query=query
                    )
                ),
            ]
        )

        print(f"[Memory Retrieval] found_in_memory={lookup_result.found_in_memory}")

        if lookup_result.found_in_memory and lookup_result.memories:
            print(f"[Memory Retrieval] Retrieved: {lookup_result.memories}")
            return {
                "messages": state["messages"],
                "found_in_memory": True,
                "memory_context": state.get("memory_context", []),
            }

    return {
        "messages": state["messages"],
        "found_in_memory": False,
        "memory_context": state.get("memory_context", []),
    }
