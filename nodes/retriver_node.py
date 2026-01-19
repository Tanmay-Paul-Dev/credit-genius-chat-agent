"""
Retriever Node
Fetches required_info values from the vector store
and uses an LLM to structure the data.

Strategy:
- Loop through each required_info field individually for accurate retrieval
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field, create_model
from states import State
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from retriever import MMRRetriever
from services.pinecone_service import vectorstore
from services.opanai_service import initialize_model


# Initialize model for structured extraction
model = initialize_model()


async def retriever_node(
    state: State, config: RunnableConfig, store: BaseStore
) -> Dict[str, Any]:
    """
    Retrieves required_info values from the vector store.
    """
    user_id = config["configurable"]["user_id"]

    # Get intent info
    intent_obj = state.get("intent", {})
    intent = intent_obj.get("intent", {})
    required_info_list = intent_obj.get("required_info", [])
    optional_info_list = intent_obj.get("optional_info", [])

    # Create retrieved_data structure with null values
    retrieved_data = {
        "required_info": {field: None for field in required_info_list},
        "optional_info": {field: None for field in optional_info_list},
    }

    print(f"[Retriever] Retrieved Data Structure: {retrieved_data}")
    print(f"[Retriever] Fetching data for user: {user_id}")

    # Initialize retriever
    retriever = MMRRetriever(
        vectorstore=vectorstore,
        k=5,
        fetch_k=20,
        lambda_mult=0.6,
        user_id=user_id
    )

    recent_messages = state["messages"]
    memory = state["memory_context"]

    # Loop through each required field individually for accurate retrieval
    for field in required_info_list:
        print(f"\n[Retriever] 🔍 Retrieving field: {field}")

        # Retrieve documents for this specific field
        docs = retriever.retrieve(field, user_id=user_id)

        if not docs:
            print(f"[Retriever] ❌ No documents found for {field}")
            continue

        print(f"[Retriever] Found {len(docs)} documents for {field}")

        # Build context from retrieved documents
        context = "\n\n".join(
            f"[DOCUMENT {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )

        # Use LLM to extract this specific field
        field_data = await _extract_single_field(context, field, recent_messages, memory)

        if field_data is not None:
            retrieved_data["required_info"][field] = field_data
            print(f"[Retriever] ✅ Found {field}: {field_data}")
        else:
            print(f"[Retriever] ❌ Could not extract {field}")

    # Retrive The Intent Information
    intent_data = retriever.retrieve(intent, user_id=user_id)

    print("====>",intent_data)

    if intent_data:
        intent_data = intent_data[0].page_content
    else:
        intent_data = ""

    retrieved_data = {**retrieved_data, "retrived_intent_info": intent_data}

    print(f"\n[Retriever] Final retrieved data: {retrieved_data}")

    return {
        **state,
        "retrieved_data": retrieved_data,
    }


async def _extract_single_field(context: str, field: str, recent_messages: List[Dict[str, Any]], memory: str) -> Any:
    """
    Use LLM to extract a single field value from document context.
    """
    # Create a simple model for single field extraction
    SingleFieldModel = create_model(
        "SingleFieldExtraction",
        value=(Any, Field(default=None, description=f"The value for {field}")),
    )

    # Create structured output model
    extractor = model.with_structured_output(
        SingleFieldModel, method="function_calling"
    )

    # Build prompt
    prompt = f"""Extract the value for "{field}" from the user data documents below.
    If not found, set value to null.

    TASK:
    - Keep the value as a short atomic sentence or value.
    - Extract only the most relevant/recent value.
    - If the value is not found in the user data documents, use "Recent Messages* and *Memory* to extract the value

    User Data:
    {context}

    Recent Messages:
    {recent_messages}

    Memory:
    {memory}

    Extract the value for: {field}"""

    # Invoke LLM
    try:
        result = await extractor.ainvoke(prompt)
        return result.value
    except Exception as e:
        print(f"[Retriever] LLM extraction error for {field}: {e}")
        return None
