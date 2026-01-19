"""
AWS Lambda Handler for Credit Genius Finance Agent
This module provides the Lambda entry point for the LangGraph-based finance agent.
"""

import os
import json
import asyncio

# Load environment variables (for local testing - Lambda uses environment config)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed in Lambda

from states import State, ErrorType
from nodes.intent_classifier_node import intent_classifier_agent_node
from nodes.finance_agent_node import finance_agent_node
from nodes.chat_node import chat_node
from nodes.error_node import error_node
from nodes.retriver_node import retriever_node

from langgraph.graph import StateGraph, END
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

from states import State
from langchain_core.runnables import RunnableConfig

from repositories.conditional_repository import (
    route_after_intent_classification,
    route_after_retriever,
    route_after_finance_agent,
)

# ----------------------------------------
# Configuration
# ----------------------------------------
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://root:Admin%40008@localhost:5432/postgres?sslmode=disable"
)


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

    if store:
        return graph.compile(store=store)
    return graph.compile()


# ----------------------------------------
# ASYNC HANDLER LOGIC
# ----------------------------------------
async def process_message(
    user_id: str,
    message: str,
    messages: list = None,
    memory_context: list = None,
) -> dict:
    """
    Process a single message through the finance agent graph.
    
    Args:
        user_id: The user's unique identifier
        message: The user's message/query
        messages: Previous conversation messages (optional)
        memory_context: User's memory context (optional)
    
    Returns:
        dict with 'answer', 'messages', and optionally 'error'
    """
    messages = messages or []
    memory_context = memory_context or []
    
    with PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()
        app = build_graph(store=store)
        
        thread_id = f"lambda-chat-{user_id}"
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        
        # Load existing memories if memory_context is empty
        if not memory_context:
            memory_ns = ("user", user_id, "details")
            existing_memories = list(store.search(memory_ns))
            for mem in existing_memories:
                memory_context.append(mem.value.get("data", ""))
        
        # Add user message
        messages.append({"role": "user", "content": message})
        messages = messages[-10:]  # Keep only last 10 messages
        
        # Build graph state
        graph_state = {
            "query": message,
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
        
        # Invoke the graph
        result = await app.ainvoke(graph_state, config=config)
        
        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "messages": result.get("messages", messages),
            }
        
        return {
            "success": True,
            "answer": result.get("final_answer", ""),
            "messages": result.get("messages", messages),
            "memory_context": memory_context,
        }


# ----------------------------------------
# LAMBDA HANDLER
# ----------------------------------------
def lambda_handler(event, context):
    """
    AWS Lambda handler for the finance agent.
    
    Expected event format (API Gateway):
    {
        "body": {
            "user_id": "uuid-string",
            "message": "user's question",
            "messages": [...],  # optional: conversation history
            "memory_context": [...]  # optional: memory context
        }
    }
    
    Or direct invocation:
    {
        "user_id": "uuid-string",
        "message": "user's question",
        ...
    }
    """
    try:
        # Parse the request body
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        elif isinstance(event.get("body"), dict):
            body = event["body"]
        else:
            # Direct invocation format
            body = event
        
        user_id = body.get("user_id")
        message = body.get("message")
        messages = body.get("messages", [])
        memory_context = body.get("memory_context", [])
        
        # Validate required fields
        if not user_id:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "user_id is required"}),
            }
        
        if not message:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "message is required"}),
            }
        
        # Run the async handler
        result = asyncio.run(
            process_message(
                user_id=user_id,
                message=message,
                messages=messages,
                memory_context=memory_context,
            )
        )
        
        # Return success response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            },
            "body": json.dumps(result),
        }
        
    except Exception as e:
        print(f"Lambda error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": False,
                "error": str(e),
            }),
        }


# ----------------------------------------
# LOCAL TESTING
# ----------------------------------------
if __name__ == "__main__":
    # Test event for local development
    test_event = {
        "user_id": "916f1a62-4e4a-4258-9440-3f1619083f3f",
        "message": "What is my credit score?",
        "messages": [],
        "memory_context": [],
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
