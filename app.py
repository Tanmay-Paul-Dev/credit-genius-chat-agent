from dotenv import load_dotenv
load_dotenv()  # Must be called before importing modules that use env vars

import asyncio
import json
from pymongo import MongoClient

from states import State, ErrorType
from nodes.intent_classifier_node import intent_classifier_agent_node
from nodes.rule_builder_node import rule_builder_node
from nodes.finance_agent_node import finance_agent_node
from nodes.followup_agent_node import followup_agent_node
from nodes.chat_node import chat_node
from nodes.error_node import error_node
from nodes.memory_retrieval_node import memory_retrieval_node
from nodes.retriver_node import retriever_node

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

from states import State, MemoryDecision
from langchain_core.runnables import RunnableConfig

from repositories.conditional_repository import (
    route_after_intent_classification,
    route_after_retriever,
    route_after_rule_builder,
    route_after_finance_agent,
    route_after_followup_agent,
    route_after_memory_retrieval,
)

# ----------------------------------------
# ENV & DB
# ----------------------------------------
client = MongoClient("mongodb://localhost:27017")

DB_URI = "postgresql://root:Admin%40008@localhost:5432/postgres?sslmode=disable"


# ----------------------------------------
# ERROR HANDLER WRAPPER
# ----------------------------------------
def with_error_handling(node_name: str, error_type: ErrorType, retryable=True):
    def wrap(fn):
        async def wrapper(state: State, config: RunnableConfig, store: BaseStore):
            try:
                return await fn(state, config, store)
            except Exception as e:
                print(
                    f"🚨 Error in {node_name} | type={error_type} | attempt={state.get('error', {}).get('attempt', 0) + 1}"
                )
                previous_attempt = state.get("error", {}).get("attempt", 0)
                return {
                    **state,
                    "error": {
                        "node": node_name,
                        "message": str(e),
                        "type": error_type,
                        "retryable": retryable,
                        "attempt": previous_attempt + 1,
                    },
                }

        return wrapper

    return wrap


# ----------------------------------------
# BUILD GRAPH
# ----------------------------------------
def build_graph(store: BaseStore = None):
    graph = StateGraph(State)

    # graph.add_node(
    #     "memory_retrieval_node",
    #     with_error_handling("memory_retrieval_node", "LLM")(memory_retrieval_node),
    # )
    graph.add_node(
        "intent_classifier",
        with_error_handling("intent_classifier", "LLM")(intent_classifier_agent_node),
    )
    graph.add_node(
        "retriever_node",
        with_error_handling("retriever_node", "LLM")(retriever_node),
    )

    graph.add_node(
        "finance_agent_node",
        with_error_handling("finance_agent_node", "LLM")(finance_agent_node),
    )

    graph.add_node(
        "chat_node",
        with_error_handling("chat_node", "LLM")(chat_node),
    )

    graph.add_node("error_node", error_node)

    graph.set_entry_point("intent_classifier")

    # graph.add_conditional_edges(
    #     "memory_retrieval_node",
    #     route_after_memory_retrieval,
    #     {
    #         "chat_node": "chat_node",
    #         "intent_classifier": "intent_classifier",
    #         "error_node": "error_node",
    #     },
    # )

    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent_classification,
        {
            "retriever_node": "retriever_node",
            "error_node": "error_node",
        },
    )

    graph.add_conditional_edges(
        "retriever_node",
        route_after_retriever,
        {
            "finance_agent_node": "finance_agent_node",
            "error_node": "error_node",
        },
    )

    graph.add_conditional_edges(
        "finance_agent_node",
        route_after_finance_agent,
        {
            "chat_node": "chat_node",
            "error_node": "error_node",
        },
    )

    graph.add_edge("error_node", END)

    # Compile with store if provided
    if store:
        return graph.compile(store=store)
    return graph.compile()


# ----------------------------------------
# TERMINAL CHAT LOOP
# ----------------------------------------
async def chat():
    # Use PostgresStore for memory persistence
    with PostgresStore.from_conn_string(DB_URI) as store:
        # Setup the store (run once for new database)
        store.setup()

        # Build graph with store
        app = build_graph(store=store)

        user_id = "916f1a62-4e4a-4258-9440-3f1619083f3f"
        thread_id = f"terminal-chat-{user_id}"
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        print("\n💬 Finance Agent Terminal Chat (with Postgres Memory)")
        print("Type 'exit' or 'quit' to stop\n")

        # Initialize messages list
        messages = []
        memory_context = []

        # Load any existing memories for this user
        memory_ns = ("user", user_id, "details")
        existing_memories = list(store.search(memory_ns))
        if existing_memories:
            print("📝 Loaded existing memories:")
            for mem in existing_memories:
                print(f"   - {mem.value.get('data', '')}")
                memory_context.append(mem.value.get("data", ""))
            print()

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    print("\n👋 Goodbye!")
                    break

                # Add user message before invoking graph
                messages.append({"role": "user", "content": user_input})
                messages = messages[-10:]  # Keep only last 10 messages

                # Build graph state with current values
                graph_state = {
                    "query": user_input,
                    "is_profile_complete": True,
                    "intent": {
                        "intent": "",
                        "required_info": [],
                        "retrievable_info": [],
                        "user_provided_info": [],
                    },
                    "tier": "PAID",
                    "final_answer": "",
                    "rule_prompt": "",
                    "messages": messages,
                    "memory_context": memory_context,
                }

                result = await app.ainvoke(
                    graph_state,
                    config=config,
                )

                if result.get("error"):
                    print("\n❌ Error:")
                    print(json.dumps(result["error"], indent=2))
                else:
                    print("\nAgent:", result.get("final_answer", ""))
                    print("-" * 50)

                # Update messages for next iteration
                messages = result.get("messages", messages)

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Bye!")
                break
            except Exception as e:
                print(f"\n🔥 Fatal Error: {str(e)}")
                break

        # Show stored memories on exit
        print("\n📝 Stored Memories (from Postgres):")
        for mem in store.search(memory_ns):
            print(f"   - {mem.value.get('data', '')}")


# ----------------------------------------
# ENTRY POINT
# ----------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

