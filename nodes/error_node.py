from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.runnables import RunnableConfig

MAX_RETRIES = 0  # Set to 0 to use external retry only


async def error_node(state: State, config: RunnableConfig, store: BaseStore) -> State:
    """
    Error node that logs errors and manages retry state.

    Since MAX_RETRIES is 0, this node just logs and preserves error state.
    """
    error = state.get("error")

    if not error:
        # No error in state, just return state as-is
        return state

    print(
        f"🚨 Error in {error.get('node', 'unknown')} | "
        f"type={error.get('type', 'unknown')} | "
        f"attempt={error.get('attempt', 0)}"
    )

    if error.get("retryable") and error.get("attempt", 0) < MAX_RETRIES:
        print("🔁 Retrying...")
        # Clear error but KEEP node name for retry
        return {**state, "error": {**error, "retry": True}}

    print("❌ Permanent failure - preserving error state")
    # Keep error in state so external retry can see it
    return state
