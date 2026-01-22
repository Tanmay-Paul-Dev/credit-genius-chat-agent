from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from states import State, RequirementExtractorState
from utils.prompt_loader import load_prompt
from services.opanai_service import large_model

# Initialize model with structured output
intent_classifier_model = large_model.with_structured_output(
    RequirementExtractorState, method="function_calling"
)


async def requirement_extractor_node(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> Dict[str, Any]:
    query: str = state.get("query", "")
    previous_messages = state.get("messages", [])
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Get existing memories to check for duplicates
    items = list(store.search(ns))
    existing_memories = (
        "\n".join(f"- {it.value.get('data', '')}" for it in items)
        if items
        else "(empty)"
    )

    # Load system prompt
    system_prompt = load_prompt("requirements_exteactor_prompt")

    # Build full prompt with context
    full_prompt = f"""{system_prompt}

        [USER STORED MEMORIES]
        {existing_memories}

        [PREVIOUS MESSAGES]
        {previous_messages}

        [CURRENT USER QUERY]
        {query}
    """

    # Call LLM with structured output
    result: RequirementExtractorState = await intent_classifier_model.ainvoke(full_prompt)

    # Debug output
    print("\n📌 Requirements Extraction Result")
    print(f"Intent         : {result.intent}")
    print(f"Required Info  : {result.required_info}")
    print(f"Optional Info  : {result.optional_info}\n")

    # Get existing intent from state and merge
    existing_intent = state.get("intent", {})
    
    return {
        "intent": {
            **existing_intent,  # Preserves query_type from intent classifier
            "intent": result.intent,
            "required_info": result.required_info,
            "optional_info": result.optional_info,
        },
    }
