from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from states import State, IntentClassifierState
from utils.prompt_loader import load_prompt
from services.opanai_service import initialize_model

# Initialize model with structured output
model = initialize_model()
intent_classifier_model = model.with_structured_output(
    IntentClassifierState, method="function_calling"
)


async def intent_classifier_agent_node(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> Dict[str, Any]:
    query: str = state.get("query", "")
    previous_messages = state.get("messages", [])
    memory_context = state.get("memory_context", [])

    # Load system prompt
    system_prompt = load_prompt("intent_classifier_prompt")

    # Build full prompt with context
    full_prompt = f"""{system_prompt}

[USER STORED MEMORIES]
{memory_context}

[PREVIOUS MESSAGES]
{previous_messages}

[CURRENT USER QUERY]
{query}
"""

    print(f"\n🧠 Intent Classifier - Query: {query}")
    print(f"🧠 Memory Context: {memory_context}")

    # Call LLM with structured output
    result: IntentClassifierState = await intent_classifier_model.ainvoke(full_prompt)

    # Convert to dict for state
    response = {
        "query_type": result.query_type,
        "intent": result.intent,
        "required_info": result.required_info,
        "optional_info": result.optional_info,
    }

    # Debug output
    print("\n📌 Intent Classification Result")
    print(f"Query          : {query}")
    print(f"Query Type     : {response['query_type']}")
    print(f"Intent         : {response['intent']}")
    print(f"Required Info  : {response['required_info']}")
    print(f"Optional Info  : {response['optional_info']}\n")

    return {
        **state,
        "intent": response,
    }
