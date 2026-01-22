import uuid
from services.opanai_service import small_model
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.runnables import RunnableConfig
from utils.prompt_loader import load_prompt


async def memory_creator_node(state: State, config: RunnableConfig, store: BaseStore):
    """
    Extracts and stores new user memories from the conversation.
    Should be called after processing to capture new user information.
    """
    memory_extractor = small_model.with_structured_output(MemoryDecision)

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Get existing memories to avoid duplicates
    items = list(store.search(ns))
    existing = (
        "\n".join(f"- {it.value.get('data', '')}" for it in items)
        if items
        else "(empty)"
    )

    query = state["query"]

    print(f"[Memory Creator] Checking for new memories to store...")

    # Check if we should store new memories
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

    print(f"[Memory Creator] should_write={decision.should_write}")

    stored_count = 0
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
                print(f"[Memory Creator] Stored: {mem.text.strip()}")
                stored_count += 1

    print(f"[Memory Creator] Stored {stored_count} new memories")

    return state  # Pass through unchanged
