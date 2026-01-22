"""
Retriever Node
Fetches required_info values from the vector store
and uses an LLM to structure the data.

Strategy:
- Loop through each required_info field individually for accurate retrieval
"""

import asyncio
import uuid
from typing import Dict, Any, List
from pydantic import BaseModel, Field, create_model
from states import State
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore
from retriever import MMRRetriever
from services.pinecone_service import vectorstore
from services.opanai_service import large_model, small_model


async def save_retrieved_data_to_memory_background(
    user_id: str,
    store: BaseStore,
    retrieved_data: dict,
):
    """
    Use LLM to determine which retrieved data is important enough to store in memory.
    Only stores key user information that will be needed in future conversations.
    Runs in background without blocking the main flow.
    """
    try:
        if not retrieved_data:
            print("[Memory] No retrieved data to save")
            return

        ns = ("user", user_id, "details")

        # Get existing memories to check for duplicates
        items = list(store.search(ns))
        existing_memories = (
            "\n".join(f"- {it.value.get('data', '')}" for it in items)
            if items
            else "(empty)"
        )

        # Get required_info and optional_info from retrieved_data
        required_info = retrieved_data.get("required_info", {}) or {}
        optional_info = retrieved_data.get("optional_info", {}) or {}

        # Combine all data for LLM analysis
        all_data = {**required_info, **optional_info}

        # Filter out None values
        all_data = {
            k: v
            for k, v in all_data.items()
            if v is not None and v != "" and v != "null"
        }

        if not all_data:
            print("[Memory] No valid data to analyze")
            return

        # Ask LLM to determine what's important to store
        prompt = f"""Analyze the following user data and determine which items are IMPORTANT to remember for future conversations.

        RETRIEVED USER DATA:
        {all_data}

        ALREADY STORED IN MEMORY:
        {existing_memories}

        TASK:
        Select the data that is IMPORTANT for personalization and future financial decisions:
        1. User's name (ALWAYS store if available)
        2. User's location/city (important for loan eligibility)
        3. Credit scores
        4. Income and employment status
        5. Any other personal factual data about the user

        DO NOT store:
        - Data that's already stored in memory (even if worded differently - CHECK SEMANTICALLY)
        - Generic descriptions or long summaries
        - Temporary or transient information
        - Duplicate information (even if worded differently)
        - Future Goals like (loan amount/purpose)

        For each important item, return it as a short sentence.
        Return ONLY a JSON array of strings, each being a memory to store.
        If nothing NEW is important enough to store, return an empty array [].

        Example output:
        ["User's name is Henry Vander", "User is located in Charlotte, NC", "User's credit score is 716", "User's income is $5000 per month"]

        Return ONLY the JSON array, nothing else."""

        response = await small_model.ainvoke([SystemMessage(content=prompt)])

        # Parse the response
        import json

        try:
            # Clean up the response - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            memories_to_store = json.loads(content)
        except json.JSONDecodeError:
            print(f"[Memory] Could not parse LLM response: {response.content}")
            return

        if not memories_to_store or not isinstance(memories_to_store, list):
            print("[Memory] LLM determined no important data to store")
            return

        # Store the important memories
        stored_count = 0
        existing_lower = set(
            m.lower() for m in existing_memories.split("\n") if m.strip()
        )

        for memory_text in memories_to_store:
            if memory_text and isinstance(memory_text, str):
                # Check if already stored (basic string check)
                if memory_text.lower() not in existing_lower:
                    store.put(ns, str(uuid.uuid4()), {"data": memory_text})
                    print(f"[Memory] Stored: {memory_text}")
                    stored_count += 1
                else:
                    print(f"[Memory] Skipped (duplicate): {memory_text}")

        if stored_count == 0:
            print("[Memory] No new important data to store")
        else:
            print(f"[Memory] Total stored: {stored_count} items")

    except Exception as e:
        print(f"[Memory] Error: {e}")


def build_composite_query(missing_fields: List[str]) -> str:
    """Build a descriptive composite query for vector store retrieval."""
    return ", ".join(missing_fields)


async def retriever_node(
    state: State, config: RunnableConfig, store: BaseStore
) -> Dict[str, Any]:
    """
    Retrieves required_info and optional_info values.
    Strategy: Extract all fields in one go from memory/messages first,
    then batch retrieve remaining fields from vector store.
    """
    user_id = config["configurable"]["user_id"]

    # Get intent info
    intent_obj = state.get("intent", {})
    intent = intent_obj.get("intent", {})
    required_info_list = intent_obj.get("required_info", [])
    optional_info_list = intent_obj.get("optional_info", [])
    all_fields = required_info_list + optional_info_list

    # Create retrieved_data structure with null values
    retrieved_data = {
        "required_info": {field: None for field in required_info_list},
        "optional_info": {field: None for field in optional_info_list},
    }

    recent_messages = state["messages"]
    memory = state["memory_context"]

    # Initialize retriever (only used if memory/messages don't have the info)
    retriever = MMRRetriever(
        vectorstore=vectorstore, k=5, fetch_k=20, lambda_mult=0.6, user_id=user_id
    )

    # Step 1 & Step 3: Run in PARALLEL
    # - Step 1: Extract fields from memory/messages
    # - Step 3: Retrieve intent info from vector store
    print(f"\n[Retriever] 🔍 Extracting all fields from memory/messages: {all_fields}")
    print(f"[Retriever] ⚡ Running Steps 1 & 3 in parallel...")



    # Define async wrapper for sync retriever call (Step 3)
    async def retrieve_intent_async():
        return retriever.retrieve(intent, user_id=user_id)

    # Run Step 1 and Step 3 in parallel
    memory_extracted, intent_docs = await asyncio.gather(
        _extract_all_fields_from_memory(all_fields, recent_messages, memory),
        retrieve_intent_async(),
    )

    # Process intent data from Step 3
    intent_data = intent_docs[0].page_content if intent_docs else ""

    # Update retrieved_data with memory-extracted values from Step 1
    missing_fields = []
    for field in required_info_list:
        if memory_extracted.get(field) is not None:
            retrieved_data["required_info"][field] = memory_extracted[field]
            print(
                f"[Retriever] ✅ Found {field} in memory/messages: {memory_extracted[field]}"
            )
        else:
            missing_fields.append(field)

    for field in optional_info_list:
        if memory_extracted.get(field) is not None:
            retrieved_data["optional_info"][field] = memory_extracted[field]
            print(
                f"[Retriever] ✅ Found {field} in memory/messages: {memory_extracted[field]}"
            )
        else:
            missing_fields.append(field)

    # Step 2: For missing fields ONLY, batch retrieve from vector store using composite query
    if missing_fields:
        print(
            f"\n[Retriever] 🔄 Missing fields, retrieving from vector store: {missing_fields}"
        )

        # Create composite query from all missing fields
        composite_query = build_composite_query(missing_fields)

        print(f"[Retriever] 🔍 Composite query: {composite_query}")
        all_docs = retriever.retrieve(composite_query, user_id=user_id)

        if all_docs:
            # Build context from retrieved documents (deduplicate by content)
            seen_content = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)

            context = "\n\n".join(
                f"[DOCUMENT {i + 1}]\n{doc.page_content}"
                for i, doc in enumerate(unique_docs[:10])  # Limit to 10 docs
            )

            # Extract all missing fields from vector store docs in one go
            vector_extracted = await _extract_all_fields_from_context(
                missing_fields, context
            )

            # Update retrieved_data with vector-extracted values
            for field in missing_fields:
                if vector_extracted.get(field) is not None:
                    if field in required_info_list:
                        retrieved_data["required_info"][field] = vector_extracted[field]
                    else:
                        retrieved_data["optional_info"][field] = vector_extracted[field]
                    print(
                        f"[Retriever] ✅ Found {field} in vector store: {vector_extracted[field]}"
                    )
                else:
                    print(f"[Retriever] ❌ Could not extract {field}")

            # 🔥 Save vector store extracted data to memory in background
            asyncio.create_task(
                save_retrieved_data_to_memory_background(
                    user_id=user_id,
                    store=store,
                    retrieved_data={
                        "required_info": {k: v for k, v in vector_extracted.items() if k in required_info_list},
                        "optional_info": {k: v for k, v in vector_extracted.items() if k in optional_info_list},
                    },
                )
            )
        else:
            print(f"[Retriever] ❌ No documents found for missing fields")

    # Add intent data to retrieved_data
    retrieved_data = {**retrieved_data, "retrived_intent_info": intent_data}

    return {
        **state,
        "retrieved_data": retrieved_data,
    }


async def _extract_all_fields_from_memory(
    fields: List[str], recent_messages: List[Dict[str, Any]], memory: List[str]
) -> Dict[str, Any]:
    """
    Use LLM to extract ALL field values from memory and recent messages in one go.
    Returns a dict with field names as keys and extracted values (or None if not found).
    """
    if not fields:
        return {}

    # Create a dynamic model with all fields
    field_definitions = {
        field: (Any, Field(default=None, description=f"The value for {field}"))
        for field in fields
    }
    MultiFieldModel = create_model("MultiFieldExtraction", **field_definitions)

    # Create structured output model
    extractor = large_model.with_structured_output(
        MultiFieldModel, method="function_calling"
    )

    # Format memory as string
    memory_str = "\n".join(f"- {m}" for m in memory) if memory else "(empty)"

    # Build prompt
    fields_list = ", ".join(fields)
    prompt = f"""Extract ALL of the following fields from the memory and recent messages below.
    For each field, if it is NOT clearly present, set its value to null.

    FIELDS TO EXTRACT: {fields_list}

    TASK:
    - Keep each value as a short atomic sentence or value.
    - Extract only if explicitly mentioned.
    - Return null for any field not found or unclear.

    Memory:
    {memory_str}

    Recent Messages:
    {recent_messages}

    Extract values for: {fields_list}"""

    # Invoke LLM
    try:
        result = await extractor.ainvoke(prompt)
        # Convert result to dict and filter out null/empty values
        extracted = {}
        for field in fields:
            value = getattr(result, field, None)
            if value is not None and value != "" and value != "null":
                extracted[field] = value
        return extracted
    except Exception as e:
        print(f"[Retriever] LLM memory extraction error: {e}")
        return {}


async def _extract_all_fields_from_context(
    fields: List[str], context: str
) -> Dict[str, Any]:
    """
    Use LLM to extract ALL field values from vector store context in one go.
    Returns a dict with field names as keys and extracted values (or None if not found).
    """
    if not fields:
        return {}

    # Create a dynamic model with all fields
    field_definitions = {
        field: (Any, Field(default=None, description=f"The value for {field}"))
        for field in fields
    }
    MultiFieldModel = create_model("MultiFieldExtraction", **field_definitions)

    # Create structured output model
    extractor = large_model.with_structured_output(
        MultiFieldModel, method="function_calling"
    )

    # Build prompt
    fields_list = ", ".join(fields)
    prompt = f"""Extract ALL of the following fields from the user data documents below.
    For each field, if it is NOT found, set its value to null.

    FIELDS TO EXTRACT: {fields_list}

    TASK:
    - Keep each value as a short atomic sentence or value.
    - Extract only the most relevant/recent value for each field.
    - Return null for any field not found.

    User Data:
    {context}

    Extract values for: {fields_list}"""

    # Invoke LLM
    try:
        result = await extractor.ainvoke(prompt)
        # Convert result to dict
        extracted = {}
        for field in fields:
            value = getattr(result, field, None)
            if value is not None and value != "" and value != "null":
                extracted[field] = value
        return extracted
    except Exception as e:
        print(f"[Retriever] LLM extraction error: {e}")
        return {}
